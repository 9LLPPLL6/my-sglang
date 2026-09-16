"""Parallel submission in the whole-prefix ``layerwise_file`` read path.

``--hicache-storage-io-threads`` shards one ``batch_get_v1`` across worker
threads, each owning its own AIO context and a disjoint slice of the batch's
pages. These cover what sharding must not change: every page still lands, it
lands in the right host slot, a missing page still truncates the prefix at its
own position rather than a shard boundary, and a batch smaller than the thread
count is still served.
"""

import os
import shutil
import tempfile

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.layerwise.hicache_layerwise_file import (
    HiCacheLayerwiseFile,
    _split_positions,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# A layer stride has to be a multiple of the filesystem's Direct I/O alignment:
# the read lands the logical layer range in place, so an unaligned length is
# refused by the kernel. 4 x 2 x 64 x 2 B = 1 KiB here; real geometries are far
# larger and always aligned.
_PAGE_SIZE = 4
_LAYERS = 6
_HEADS = 2
_HEAD_DIM = 64


def _make_host(page_num: int) -> MHATokenToKVPoolHost:
    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host.layout = "page_first_direct"
    host.page_num = page_num
    host.layer_num = _LAYERS
    host.page_size = _PAGE_SIZE
    host.head_num = _HEADS
    host.head_dim = _HEAD_DIM
    host.size = page_num * _PAGE_SIZE
    host.dtype = torch.bfloat16
    host.kv_buffer = torch.zeros(
        (2, page_num, _LAYERS, _PAGE_SIZE, _HEADS, _HEAD_DIM), dtype=torch.bfloat16
    )
    return host


def _storage_config(root: str) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="io-threads-test",
        extra_config={"layerwise_root": root, "require_direct_io": False},
    )


class TestLayerwiseFileIoThreads(CustomTestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="sglang-io-threads-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _backend(self, host, *, io_threads):
        """Build a backend under a published config, the way a server does."""
        override = get_context().override_server_args(
            hicache_storage_io_threads=io_threads
        )
        override.install()
        self.addCleanup(override.restore)
        backend = HiCacheLayerwiseFile(_storage_config(self.root), host)
        self.addCleanup(backend.close)
        return backend

    def _write_pages(self, backend, host, pages):
        """Fill the host pool with known bytes and publish every page."""
        torch.manual_seed(7)
        host.kv_buffer.copy_(
            torch.randint(0, 200, host.kv_buffer.shape).to(host.dtype)
        )
        keys = [f"page{index:04d}" for index in range(pages)]
        indices = torch.arange(pages * _PAGE_SIZE, dtype=torch.int64)
        self.assertTrue(all(backend.batch_set_v1(keys, indices)))
        return keys, indices, host.kv_buffer.clone()

    def _round_trip(self, *, pages, io_threads):
        host = _make_host(pages)
        backend = self._backend(host, io_threads=io_threads)
        keys, indices, written = self._write_pages(backend, host, pages)
        host.kv_buffer.zero_()
        results = backend.batch_get_v1(keys, indices)
        return backend, host, keys, indices, written, results

    def test_default_is_one_thread_and_needs_no_published_config(self):
        # The pre-existing behavior, and the path a bare unit test takes.
        backend = HiCacheLayerwiseFile(_storage_config(self.root), _make_host(2))
        self.addCleanup(backend.close)
        self.assertEqual(backend.io_threads, 1)
        self.assertIsNone(backend._pool)
        self.assertEqual(len(backend._contexts), 1)

    def test_sharded_read_returns_every_page_byte_for_byte(self):
        pages = 9
        _, host, _, _, written, results = self._round_trip(pages=pages, io_threads=4)
        self.assertEqual(results, [True] * pages)
        self.assertTrue(torch.equal(host.kv_buffer, written))

    def test_sharded_and_serial_reads_agree(self):
        pages = 9
        _, threaded_host, _, _, written, _ = self._round_trip(
            pages=pages, io_threads=4
        )
        serial_host = _make_host(pages)
        serial = self._backend(serial_host, io_threads=1)
        keys = [f"page{index:04d}" for index in range(pages)]
        indices = torch.arange(pages * _PAGE_SIZE, dtype=torch.int64)
        serial_host.kv_buffer.zero_()
        self.assertEqual(serial.batch_get_v1(keys, indices), [True] * pages)
        self.assertTrue(torch.equal(threaded_host.kv_buffer, serial_host.kv_buffer))
        self.assertTrue(torch.equal(serial_host.kv_buffer, written))

    def test_more_threads_than_pages_still_serves_the_batch(self):
        pages = 2
        backend, host, _, _, written, results = self._round_trip(
            pages=pages, io_threads=8
        )
        self.assertEqual(results, [True] * pages)
        self.assertTrue(torch.equal(host.kv_buffer, written))

    def test_a_missing_page_truncates_at_its_own_position(self):
        pages = 8
        host = _make_host(pages)
        backend = self._backend(host, io_threads=4)
        keys, indices, _ = self._write_pages(backend, host, pages)
        # Page 5 sits in the last shard; the prefix must end there, not at the
        # boundary of whichever shard happened to notice.
        os.unlink(backend.writer.page_path(keys[5]))
        results = backend.batch_get_v1(keys, indices)
        self.assertEqual(results, [True] * 5 + [False] * 3)

    def test_split_positions_covers_every_page_exactly_once(self):
        for total, parts in ((9, 4), (2, 8), (128, 8), (1, 1), (7, 7)):
            shards = _split_positions(total, parts)
            covered = [position for shard in shards for position in shard]
            self.assertEqual(covered, list(range(total)), f"{total}/{parts}")
            self.assertTrue(all(len(shard) > 0 for shard in shards))
            self.assertLessEqual(len(shards), max(1, min(parts, total)))


if __name__ == "__main__":
    import unittest

    unittest.main()
