"""Disk-bounding tests for the layerwise HiCache page store.

Regression guard for a real failure: an unbounded page store filled the device
during a write-through run and the scheduler process was killed.  A page store
must decline a write rather than let the filesystem run out.
"""

import os
import shutil
import tempfile
import unittest

from sglang.srt.mem_cache.layerwise_storage.aio_engine import AlignmentProfile
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    PageIdentity,
    model_fingerprint,
)
from sglang.srt.mem_cache.layerwise_storage.page_store_evictor import PageStoreEvictor
from sglang.srt.mem_cache.layerwise_storage.page_writer import PageFileWriter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ALIGNMENT = 4096


def _identity() -> PageIdentity:
    return PageIdentity(
        fingerprint=model_fingerprint(
            model_name="evictor-test",
            dtype_name="float16",
            layer_num=2,
            page_size=16,
            local_kv_heads=8,
            head_dim=64,
            tp_size=1,
        ),
        tp_size=1,
        tp_rank=0,
        dtype_name="float16",
        layer_num=2,
        page_size=16,
        local_kv_heads=8,
        head_dim=64,
        element_size=2,
    )


def _writer(root: str, evictor: PageStoreEvictor) -> PageFileWriter:
    return PageFileWriter(
        root=root,
        identity=_identity(),
        alignment_profile=AlignmentProfile(
            memory_alignment=_ALIGNMENT,
            offset_alignment=_ALIGNMENT,
            length_alignment=_ALIGNMENT,
            max_transfer_nbytes=1 << 24,
            direct_io_available=False,
        ),
        require_direct_io=False,
        evictor=evictor,
    )


def _count_pages(root: str) -> int:
    return sum(
        1
        for _, _, filenames in os.walk(root)
        for filename in filenames
        if filename.endswith(".kv")
    )


class TestLayerwisePageStoreEvictor(CustomTestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="sglang-page-store-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _publish(self, writer: PageFileWriter, index: int) -> bool:
        region = writer.layout.identity.layer_num * writer.layout.layer_stride
        payload = bytes([index % 251]) * region
        return writer.write_page_bytes(
            f"page{index:04d}", k_bytes=payload, v_bytes=payload
        )

    def test_byte_cap_evicts_least_recently_used_pages(self):
        evictor = PageStoreEvictor(root=self.root, max_bytes=0, min_free_bytes=0)
        writer = _writer(self.root, evictor)
        page_nbytes = writer.layout.file_nbytes

        bounded = PageStoreEvictor(
            root=self.root, max_bytes=3 * page_nbytes, min_free_bytes=0
        )
        writer = _writer(self.root, bounded)
        for index in range(6):
            self.assertTrue(self._publish(writer, index))

        self.assertLessEqual(_count_pages(self.root), 3)
        self.assertFalse(writer.exists("page0000"))
        self.assertTrue(writer.exists("page0005"))

    def test_free_space_watermark_declines_a_write(self):
        # A watermark larger than the device guarantees the "no room" branch
        # without needing to actually fill a filesystem.
        stats = os.statvfs(self.root)
        device_bytes = stats.f_blocks * stats.f_frsize
        evictor = PageStoreEvictor(
            root=self.root, max_bytes=0, min_free_bytes=device_bytes * 2
        )
        writer = _writer(self.root, evictor)

        self.assertFalse(self._publish(writer, 0))
        self.assertEqual(_count_pages(self.root), 0)

    def test_unbounded_store_admits_everything(self):
        evictor = PageStoreEvictor(root=self.root, max_bytes=0, min_free_bytes=0)
        self.assertFalse(evictor.enabled)
        writer = _writer(self.root, evictor)
        for index in range(4):
            self.assertTrue(self._publish(writer, index))
        self.assertEqual(_count_pages(self.root), 4)

    def test_restart_inherits_the_existing_page_accounting(self):
        writer = _writer(
            self.root, PageStoreEvictor(root=self.root, max_bytes=0, min_free_bytes=0)
        )
        for index in range(4):
            self.assertTrue(self._publish(writer, index))
        page_nbytes = writer.layout.file_nbytes

        restarted = PageStoreEvictor(
            root=self.root, max_bytes=4 * page_nbytes, min_free_bytes=0
        )
        self.assertEqual(restarted.used_bytes, 4 * page_nbytes)

        writer = _writer(self.root, restarted)
        self.assertTrue(self._publish(writer, 99))
        self.assertLessEqual(_count_pages(self.root), 4)


if __name__ == "__main__":
    unittest.main()
