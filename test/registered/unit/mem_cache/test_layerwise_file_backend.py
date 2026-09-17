"""End-to-end tests for the layerwise HiCache page-file storage tier.

These cover the contract the streaming pipeline depends on: a page published by
the write path is readable as independent layer ranges, an aligned extent lands
directly in the host KV buffer, an unaligned one is bounced without corrupting
neighbouring layers, and no completion is reported while the kernel can still
write to a target.
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
from sglang.srt.mem_cache.layerwise_storage.file_backend import LayerwiseFileBackend
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    PageIdentity,
    build_page_layout,
    model_fingerprint,
)
from sglang.srt.mem_cache.layerwise_storage.page_writer import PageFileWriter
from sglang.srt.mem_cache.layerwise_storage.plan_builder import (
    build_read_plan,
    split_layer_groups,
)
from sglang.srt.mem_cache.layerwise_storage.types import (
    ExtentCompletionStatus,
    HandleTerminalStatus,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_ALIGNMENT = 4096


def _make_host(
    *,
    page_num: int,
    layer_num: int,
    page_size: int,
    head_num: int,
    head_dim: int,
    dtype: torch.dtype,
) -> MHATokenToKVPoolHost:
    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host.layout = "page_first_direct"
    host.page_num = page_num
    host.layer_num = layer_num
    host.page_size = page_size
    host.head_num = head_num
    host.head_dim = head_dim
    host.size = page_num * page_size
    host.dtype = dtype
    host.kv_buffer = torch.zeros(
        (2, page_num, layer_num, page_size, head_num, head_dim), dtype=dtype
    )
    return host


def _make_identity(host: MHATokenToKVPoolHost, *, tp_rank: int = 0, tp_size: int = 2):
    fingerprint = model_fingerprint(
        model_name="unit-test-model",
        dtype_name=str(host.dtype).removeprefix("torch."),
        layer_num=host.layer_num,
        page_size=host.page_size,
        local_kv_heads=host.head_num,
        head_dim=host.head_dim,
        tp_size=tp_size,
    )
    return PageIdentity(
        fingerprint=fingerprint,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype_name=str(host.dtype).removeprefix("torch."),
        layer_num=host.layer_num,
        page_size=host.page_size,
        local_kv_heads=host.head_num,
        head_dim=host.head_dim,
        element_size=host.dtype.itemsize,
    )


def _reference_page(host: MHATokenToKVPoolHost, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        200,
        (2, host.layer_num, host.page_size, host.head_num, host.head_dim),
        generator=generator,
    ).to(host.dtype)


class _LayerwiseFixture:
    """One temp-directory deployment: writer, backend and a matching host pool."""

    def __init__(
        self,
        *,
        page_size,
        head_num,
        head_dim,
        dtype,
        layer_num=6,
        pages=3,
        io_threads=1,
    ):
        self.root = tempfile.mkdtemp(prefix="sglang-layerwise-")
        probed = probe_alignment(self.root, require_direct=False)
        self.profile = AlignmentProfile(
            memory_alignment=_ALIGNMENT,
            offset_alignment=_ALIGNMENT,
            length_alignment=_ALIGNMENT,
            max_transfer_nbytes=probed.max_transfer_nbytes,
            direct_io_available=probed.direct_io_available,
        )
        self.host = _make_host(
            page_num=pages,
            layer_num=layer_num,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.identity = _make_identity(self.host)
        self.layout = build_page_layout(self.identity, alignment=_ALIGNMENT)
        self.writer = PageFileWriter(
            root=self.root,
            identity=self.identity,
            alignment_profile=self.profile,
            require_direct_io=False,
        )
        self.backend = LayerwiseFileBackend(
            root=self.root,
            identity=self.identity,
            queue_depth=64,
            io_threads=io_threads,
            alignment_profile=self.profile,
            require_direct_io=False,
        )
        self.pages = pages
        self.references = []

    def publish_pages(self) -> list[str]:
        keys = []
        for page in range(self.pages):
            reference = _reference_page(self.host, seed=1000 + page)
            self.references.append(reference)
            key = f"page{page:04d}"
            keys.append(key)
            assert self.writer.write_page_bytes(
                key,
                k_bytes=reference[0].contiguous().view(torch.uint8).numpy().tobytes(),
                v_bytes=reference[1].contiguous().view(torch.uint8).numpy().tobytes(),
            )
        return keys

    def stream(self, *, keys, group_size, timeout_s=20.0):
        """Run one transaction to completion, group by ordered group."""
        host_indices = torch.arange(len(keys) * self.host.page_size, dtype=torch.int64)
        plan, target = build_read_plan(
            host_pool=self.host,
            host_indices=host_indices,
            page_keys=keys,
            layout=self.layout,
            group_size=group_size,
        )
        handle = self.backend.begin_read(
            transaction_id="txn-0", generation=0, plan=plan, target=target
        )
        seen = {}
        for group in plan.groups:
            self.backend.submit_group(
                handle=handle,
                group=group,
                priority=0 if group.group_id == 0 else 2,
                deadline_s=None,
            )
            deadline = time.monotonic() + timeout_s
            done = 0
            while done < len(group.extents):
                for completion in self.backend.poll(handle=handle):
                    seen.setdefault(completion.group_id, []).append(completion)
                    if completion.group_id == group.group_id:
                        done += 1
                if time.monotonic() > deadline:
                    raise TimeoutError(f"group {group.group_id} did not complete")
        return plan, handle, seen

    def close(self):
        self.backend.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)


class TestLayerwiseFileBackend(CustomTestCase):
    def _run_round_trip(self, *, page_size, head_num, head_dim, dtype, expect_bounce):
        fixture = _LayerwiseFixture(
            page_size=page_size, head_num=head_num, head_dim=head_dim, dtype=dtype
        )
        try:
            keys = fixture.publish_pages()
            plan, handle, seen = fixture.stream(keys=keys, group_size=2)

            layer_stride = (
                page_size
                * head_num
                * head_dim
                * torch.tensor([], dtype=dtype).element_size()
            )
            self.assertEqual(layer_stride % _ALIGNMENT == 0, not expect_bounce)

            for completions in seen.values():
                for completion in completions:
                    self.assertIs(completion.status, ExtentCompletionStatus.SUCCEEDED)
            self.assertIs(
                fixture.backend.poll_terminal(handle=handle),
                HandleTerminalStatus.SUCCEEDED,
            )

            for page, reference in enumerate(fixture.references):
                torch.testing.assert_close(
                    fixture.host.kv_buffer[:, page], reference, rtol=0, atol=0
                )
            fixture.backend.close(handle=handle)
        finally:
            fixture.close()

    def test_aligned_extents_land_directly_in_the_host_pool(self):
        self._run_round_trip(
            page_size=16,
            head_num=8,
            head_dim=64,
            dtype=torch.float16,
            expect_bounce=False,
        )

    def test_unaligned_extents_bounce_without_touching_neighbour_layers(self):
        self._run_round_trip(
            page_size=2,
            head_num=3,
            head_dim=4,
            dtype=torch.float32,
            expect_bounce=True,
        )

    def test_partial_layer_range_leaves_later_layers_untouched(self):
        fixture = _LayerwiseFixture(
            page_size=16, head_num=8, head_dim=64, dtype=torch.float16
        )
        try:
            keys = fixture.publish_pages()
            host_indices = torch.arange(
                len(keys) * fixture.host.page_size, dtype=torch.int64
            )
            plan, target = build_read_plan(
                host_pool=fixture.host,
                host_indices=host_indices,
                page_keys=keys,
                layout=fixture.layout,
                group_size=2,
            )
            handle = fixture.backend.begin_read(
                transaction_id="txn-partial", generation=0, plan=plan, target=target
            )
            first = plan.groups[0]
            fixture.backend.submit_group(
                handle=handle, group=first, priority=0, deadline_s=None
            )
            done = 0
            deadline = time.monotonic() + 20.0
            while done < len(first.extents):
                done += len(fixture.backend.poll(handle=handle))
                if time.monotonic() > deadline:
                    raise TimeoutError("group 0 did not complete")

            for page, reference in enumerate(fixture.references):
                torch.testing.assert_close(
                    fixture.host.kv_buffer[:, page, 0:2],
                    reference[:, 0:2],
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(
                    torch.all(fixture.host.kv_buffer[:, page, 2:] == 0),
                    "an unsubmitted layer range must stay untouched",
                )
            fixture.backend.close(handle=handle)
        finally:
            fixture.close()

    def test_missing_page_fails_the_extent_instead_of_raising(self):
        fixture = _LayerwiseFixture(
            page_size=16, head_num=8, head_dim=64, dtype=torch.float16
        )
        try:
            host_indices = torch.arange(fixture.host.page_size, dtype=torch.int64)
            plan, target = build_read_plan(
                host_pool=fixture.host,
                host_indices=host_indices,
                page_keys=["never-written"],
                layout=fixture.layout,
                group_size=8,
            )
            handle = fixture.backend.begin_read(
                transaction_id="txn-missing", generation=0, plan=plan, target=target
            )
            fixture.backend.submit_group(
                handle=handle, group=plan.groups[0], priority=0, deadline_s=None
            )
            completions = fixture.backend.poll(handle=handle)
            self.assertTrue(completions)
            self.assertTrue(
                all(
                    completion.status is ExtentCompletionStatus.FAILED
                    for completion in completions
                )
            )
            self.assertIs(
                fixture.backend.poll_terminal(handle=handle),
                HandleTerminalStatus.FAILED,
            )
            self.assertTrue(torch.all(fixture.host.kv_buffer == 0))
        finally:
            fixture.close()

    def test_close_is_refused_while_an_operation_can_touch_the_target(self):
        fixture = _LayerwiseFixture(
            page_size=16, head_num=8, head_dim=64, dtype=torch.float16
        )
        try:
            keys = fixture.publish_pages()
            host_indices = torch.arange(
                len(keys) * fixture.host.page_size, dtype=torch.int64
            )
            plan, target = build_read_plan(
                host_pool=fixture.host,
                host_indices=host_indices,
                page_keys=keys,
                layout=fixture.layout,
                group_size=8,
            )
            handle = fixture.backend.begin_read(
                transaction_id="txn-close", generation=0, plan=plan, target=target
            )
            fixture.backend.submit_group(
                handle=handle, group=plan.groups[0], priority=0, deadline_s=None
            )
            with self.assertRaisesRegex(RuntimeError, "still has"):
                fixture.backend.close(handle=handle)

            deadline = time.monotonic() + 20.0
            while (
                fixture.backend.poll_terminal(handle=handle)
                is HandleTerminalStatus.ACTIVE
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError("reads never reached a terminal state")
            fixture.backend.close(handle=handle)
        finally:
            fixture.close()

    def test_sharded_reads_reproduce_the_kv_exactly(self):
        """Several submitting threads must land the same bytes as one.

        Each shard owns its own AIO context and reads a disjoint set of pages,
        so a page landing in the wrong shard's target, or two shards racing one
        file, shows up here as wrong KV rather than as a hang."""
        fixture = _LayerwiseFixture(
            page_size=16,
            head_num=8,
            head_dim=64,
            dtype=torch.float16,
            pages=5,
            io_threads=4,
        )
        try:
            keys = fixture.publish_pages()
            fixture.host.kv_buffer.zero_()
            _, handle, seen = fixture.stream(keys=keys, group_size=2)

            for completions in seen.values():
                for completion in completions:
                    self.assertIs(completion.status, ExtentCompletionStatus.SUCCEEDED)
            self.assertIs(
                fixture.backend.poll_terminal(handle=handle),
                HandleTerminalStatus.SUCCEEDED,
            )
            for page, reference in enumerate(fixture.references):
                torch.testing.assert_close(
                    fixture.host.kv_buffer[:, page], reference, rtol=0, atol=0
                )
            fixture.backend.close(handle=handle)
        finally:
            fixture.close()

    def test_every_page_is_owned_by_exactly_one_shard(self):
        """Two shards reading one page file would double-open it and could
        interleave into the same target range."""
        fixture = _LayerwiseFixture(
            page_size=16,
            head_num=8,
            head_dim=64,
            dtype=torch.float16,
            pages=5,
            io_threads=4,
        )
        try:
            keys = fixture.publish_pages()
            host_indices = torch.arange(
                len(keys) * fixture.host.page_size, dtype=torch.int64
            )
            plan, target = build_read_plan(
                host_pool=fixture.host,
                host_indices=host_indices,
                page_keys=keys,
                layout=fixture.layout,
                group_size=2,
            )
            handle = fixture.backend.begin_read(
                transaction_id="txn-shard", generation=0, plan=plan, target=target
            )
            state = fixture.backend._state(handle)
            self.assertEqual(set(state.shard_of_page), set(keys))
            self.assertLessEqual(max(state.shard_of_page.values()), 3)
        finally:
            fixture.close()

    def test_close_reports_what_each_shard_actually_read(self):
        """Without this line the streaming path's bandwidth is unobservable,
        and its share of an exposed TTFT can only be inferred from the other
        read path's numbers."""
        fixture = _LayerwiseFixture(
            page_size=16,
            head_num=8,
            head_dim=64,
            dtype=torch.float16,
            pages=4,
            io_threads=2,
        )
        try:
            keys = fixture.publish_pages()
            _, handle, _ = fixture.stream(keys=keys, group_size=2)
            state = fixture.backend._state(handle)

            self.assertEqual(len(state.shard_of_page), len(keys))
            self.assertGreater(state.bytes_done, 0)
            self.assertGreater(state.extents_done, 0)
            self.assertEqual(sum(state.shard_bytes.values()), state.bytes_done)
            self.assertGreater(state.last_complete_s, state.first_submit_s)

            with self.assertLogs(
                "sglang.srt.mem_cache.layerwise_storage.file_backend", level="INFO"
            ) as logs:
                fixture.backend.close(handle=handle)
            line = "\n".join(logs.output)
            self.assertIn("layerwise_stream read:", line)
            self.assertIn("threads=2", line)
            self.assertIn("shard_bytes=", line)
        finally:
            fixture.close()

    def test_group_split_covers_every_layer_exactly_once(self):
        """Even groups, including the first; only the last may be short."""
        specs = split_layer_groups(layer_num=7, group_size=3)
        self.assertEqual(
            [(spec.layer_start, spec.layer_end) for spec in specs],
            [(0, 3), (3, 6), (6, 7)],
        )
        for previous, current in zip(specs, specs[1:]):
            self.assertEqual(previous.layer_end, current.layer_start)


if __name__ == "__main__":
    unittest.main()
