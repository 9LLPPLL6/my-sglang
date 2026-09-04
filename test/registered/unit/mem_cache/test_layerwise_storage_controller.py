"""End-to-end test of the layerwise storage controller on real page files.

Covers the whole chain a streaming request walks: publish pages through the
write-through writer, discover them, stream them into private host staging in
layer groups, hand each agreed group to the H2D session in order, and release
ownership exactly once.  The device side is faked because it is the only part
that needs a GPU; everything below it is the real implementation.
"""

import shutil
import tempfile
import time
import unittest

import torch

from sglang.srt.mem_cache.layerwise_storage.aio_engine import (
    AlignmentProfile,
    probe_alignment,
)
from sglang.srt.mem_cache.layerwise_storage.consensus import SingleRankConsensus
from sglang.srt.mem_cache.layerwise_storage.controller import (
    LayerwiseControllerConfig,
    LayerwiseStorageController,
)
from sglang.srt.mem_cache.layerwise_storage.file_backend import LayerwiseFileBackend
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    PageIdentity,
    model_fingerprint,
)
from sglang.srt.mem_cache.layerwise_storage.page_writer import PageFileWriter
from sglang.srt.mem_cache.layerwise_storage.state_machine import TransactionState
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ALIGNMENT = 4096
_LAYERS = 8
_PAGE_SIZE = 16
_HEADS = 8
_HEAD_DIM = 64
_PAGES = 4


class _FakeStreamingHandle:
    def __init__(self, producer_id: int):
        self.producer_id = producer_id


class _FakeCacheController:
    """Records the layer ranges the pipeline hands to the H2D stream."""

    def __init__(self):
        self.ranges: list[tuple[int, int]] = []
        self.finished = 0
        self.started = 0

    def start_streaming_load(self, *, host_indices, device_indices, node_ids):
        self.started += 1
        return _FakeStreamingHandle(producer_id=self.started - 1)

    def submit_streaming_load_range(self, handle, *, layer_start, layer_end):
        self.ranges.append((layer_start, layer_end))

    def finish_streaming_load(self, handle):
        self.finished += 1


def _make_host() -> MHATokenToKVPoolHost:
    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host.layout = "page_first_direct"
    host.page_num = _PAGES
    host.layer_num = _LAYERS
    host.page_size = _PAGE_SIZE
    host.head_num = _HEADS
    host.head_dim = _HEAD_DIM
    host.size = _PAGES * _PAGE_SIZE
    host.dtype = torch.float16
    host.kv_buffer = torch.zeros(
        (2, _PAGES, _LAYERS, _PAGE_SIZE, _HEADS, _HEAD_DIM), dtype=torch.float16
    )
    return host


def _make_identity() -> PageIdentity:
    return PageIdentity(
        fingerprint=model_fingerprint(
            model_name="controller-test",
            dtype_name="float16",
            layer_num=_LAYERS,
            page_size=_PAGE_SIZE,
            local_kv_heads=_HEADS,
            head_dim=_HEAD_DIM,
            tp_size=1,
        ),
        tp_size=1,
        tp_rank=0,
        dtype_name="float16",
        layer_num=_LAYERS,
        page_size=_PAGE_SIZE,
        local_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
        element_size=2,
    )


class _ControllerFixture:
    def __init__(self, *, first_group_layers=1, group_size=2, read_ahead_groups=2):
        self.root = tempfile.mkdtemp(prefix="sglang-layerwise-ctl-")
        probed = probe_alignment(self.root, require_direct=False)
        profile = AlignmentProfile(
            memory_alignment=_ALIGNMENT,
            offset_alignment=_ALIGNMENT,
            length_alignment=_ALIGNMENT,
            max_transfer_nbytes=probed.max_transfer_nbytes,
            direct_io_available=probed.direct_io_available,
        )
        identity = _make_identity()
        self.host = _make_host()
        self.cache_controller = _FakeCacheController()
        writer = PageFileWriter(
            root=self.root,
            identity=identity,
            alignment_profile=profile,
            require_direct_io=False,
        )
        backend = LayerwiseFileBackend(
            root=self.root,
            identity=identity,
            queue_depth=64,
            max_inflight_bytes=1 << 26,
            alignment_profile=profile,
            require_direct_io=False,
        )
        self.controller = LayerwiseStorageController(
            host_pool=self.host,
            cache_controller=self.cache_controller,
            config=LayerwiseControllerConfig(
                root=self.root,
                first_group_layers=first_group_layers,
                group_size=group_size,
                read_ahead_groups=read_ahead_groups,
                group_timeout_s=30.0,
                admission_budget_s=0.0,
                max_inflight_bytes=1 << 26,
                slow_fallback="full_wait",
            ),
            identity=identity,
            consensus=SingleRankConsensus(),
            backend=backend,
            writer=writer,
        )

    def seed_pages(self) -> tuple[list[str], torch.Tensor]:
        """Fill the host pool, publish it through write-through, then clear it."""
        reference = torch.randint(
            0,
            200,
            (2, _PAGES, _LAYERS, _PAGE_SIZE, _HEADS, _HEAD_DIM),
            generator=torch.Generator().manual_seed(7),
        ).to(torch.float16)
        self.host.kv_buffer.copy_(reference)
        keys = [f"seed{page:04d}" for page in range(_PAGES)]
        host_indices = torch.arange(_PAGES * _PAGE_SIZE, dtype=torch.int64)
        results = self.controller.write_through(
            page_keys=keys, host_indices=host_indices
        )
        assert all(results)
        self.host.kv_buffer.zero_()
        return keys, reference

    def drive(self, request_id: str, *, timeout_s=20.0) -> None:
        deadline = time.monotonic() + timeout_s
        transaction = self.controller._active[request_id].transaction
        while transaction.machine.state is not TransactionState.STORAGE_COMPLETE:
            self.controller.poll()
            if transaction.aborted:
                raise AssertionError(f"transaction aborted: {transaction.error}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"state stuck at {transaction.machine.state}")

    def close(self):
        self.controller.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)


class TestLayerwiseStorageController(CustomTestCase):
    def test_write_through_then_stream_reproduces_the_kv_exactly(self):
        fixture = _ControllerFixture()
        try:
            keys, reference = fixture.seed_pages()
            self.assertEqual(fixture.controller.query_hit_pages(keys), len(keys))

            host_indices = torch.arange(_PAGES * _PAGE_SIZE, dtype=torch.int64)
            fixture.controller.begin(
                request_id="req-0", page_keys=keys, host_indices=host_indices
            )

            deadline = time.monotonic() + 20.0
            while not fixture.controller.admission_ready("req-0"):
                fixture.controller.poll()
                if time.monotonic() > deadline:
                    raise TimeoutError("group 0 never became admission ready")
            self.assertEqual(
                fixture.cache_controller.ranges,
                [],
                "no layer may reach the device before admission",
            )

            fixture.controller.attach_device(
                request_id="req-0",
                device_indices=torch.arange(_PAGES * _PAGE_SIZE, dtype=torch.int64),
                node_ids=[1],
            )
            fixture.drive("req-0")

            self.assertEqual(
                fixture.cache_controller.ranges,
                [(0, 1), (1, 3), (3, 5), (5, 7), (7, 8)],
                "layer ranges must be contiguous and in order",
            )
            self.assertEqual(fixture.cache_controller.finished, 1)
            torch.testing.assert_close(
                fixture.host.kv_buffer, reference, rtol=0, atol=0
            )

            fixture.controller.note_forward_complete("req-0")
            self.assertTrue(fixture.controller.try_release("req-0", committed=True))
            self.assertNotIn("req-0", fixture.controller._active)
        finally:
            fixture.close()

    def test_query_stops_at_the_first_missing_page(self):
        fixture = _ControllerFixture()
        try:
            keys, _ = fixture.seed_pages()
            probed = list(keys[:2]) + ["absent", keys[2]]
            self.assertEqual(fixture.controller.query_hit_pages(probed), 2)
        finally:
            fixture.close()

    def test_a_missing_page_aborts_without_touching_the_device(self):
        fixture = _ControllerFixture()
        try:
            keys, _ = fixture.seed_pages()
            host_indices = torch.arange(_PAGES * _PAGE_SIZE, dtype=torch.int64)
            fixture.controller.begin(
                request_id="req-miss",
                page_keys=list(keys[:-1]) + ["absent"],
                host_indices=host_indices,
            )
            deadline = time.monotonic() + 20.0
            transaction = fixture.controller._active["req-miss"].transaction
            while not transaction.aborted:
                fixture.controller.poll()
                if time.monotonic() > deadline:
                    raise TimeoutError("a missing page never aborted the transaction")

            self.assertEqual(fixture.cache_controller.ranges, [])
            deadline = time.monotonic() + 20.0
            while not fixture.controller.try_release("req-miss", committed=False):
                if time.monotonic() > deadline:
                    raise TimeoutError("staging never became safe to release")
            self.assertNotIn("req-miss", fixture.controller._active)
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
