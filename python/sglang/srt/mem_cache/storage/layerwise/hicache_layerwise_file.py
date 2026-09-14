"""Direct I/O page-file HiCache backend (``--hicache-storage-backend layerwise_file``).

The stock ``file`` backend reads a page at a time through buffered I/O, so an
L3 hit is paced by one synchronous ``read`` per page plus a page-cache copy.
This backend keeps the same page identity and the same on-disk payload as the
layerwise streaming tier, but issues every page of a batch as ``O_DIRECT``
extents on one asynchronous queue, landing them straight in the host KV pool.

It deliberately implements the existing whole-prefix ``batch_get_v1`` contract
rather than layer-range streaming: it is the drop-in that makes an L3 hit as
fast as the device allows, and it shares its format with the streaming path, so
pages written by one are readable by the other.

``--hicache-storage-io-threads`` picks the submission model. One thread with one
AIO context is enough to saturate a single local NVMe, but not a parallel
filesystem: there the per-request cost is paid by the submitting thread, so one
thread plateaus far below the aggregate bandwidth however deep its queue gets.
Above one, each worker owns a context and a disjoint slice of the batch's pages,
so a page file is never touched by two threads -- which is what keeps the
filesystem's per-inode serialization out of the picture.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.layerwise_storage.aio_engine import (
    LinuxAioContext,
    probe_alignment,
)
from sglang.srt.mem_cache.layerwise_storage.file_backend import BouncePool
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    PageIdentity,
    model_fingerprint,
)
from sglang.srt.mem_cache.layerwise_storage.page_writer import PageFileWriter
from sglang.srt.runtime_context import get_memory

logger = logging.getLogger(__name__)

_QUEUE_DEPTH = 512


class HiCacheLayerwiseFile(HiCacheStorage):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
    ):
        if mem_pool_host.layout != "page_first_direct":
            raise ValueError(
                "the layerwise_file backend requires "
                f"--hicache-mem-layout=page_first_direct, got {mem_pool_host.layout!r}"
            )
        self.mem_pool_host = mem_pool_host
        self.page_size = mem_pool_host.page_size
        self.root = _resolve_root(storage_config)
        self.require_direct_io = _resolve_require_direct(storage_config)

        dtype_name = str(mem_pool_host.dtype).removeprefix("torch.")
        model_name = storage_config.model_name or "unknown-model"
        identity = PageIdentity(
            fingerprint=model_fingerprint(
                model_name=model_name,
                dtype_name=dtype_name,
                layer_num=mem_pool_host.layer_num,
                page_size=mem_pool_host.page_size,
                local_kv_heads=mem_pool_host.head_num,
                head_dim=mem_pool_host.head_dim,
                tp_size=storage_config.tp_size,
            ),
            tp_size=storage_config.tp_size,
            tp_rank=storage_config.tp_rank,
            dtype_name=dtype_name,
            layer_num=mem_pool_host.layer_num,
            page_size=mem_pool_host.page_size,
            local_kv_heads=mem_pool_host.head_num,
            head_dim=mem_pool_host.head_dim,
            element_size=mem_pool_host.dtype.itemsize,
        )
        profile = probe_alignment(self.root, require_direct=self.require_direct_io)
        self.writer = PageFileWriter(
            root=self.root,
            identity=identity,
            alignment_profile=profile,
            require_direct_io=self.require_direct_io,
        )
        self.layout = self.writer.layout
        self.alignment_profile = profile
        self._region_nbytes = identity.layer_num * self.layout.layer_stride
        self.io_threads = _resolve_io_threads()
        # One context per shard, never shared between threads: a context owns a
        # completion buffer and a dict of live iocbs, neither of which is safe
        # to drive from two threads at once.
        self._contexts = [
            LinuxAioContext(queue_depth=_QUEUE_DEPTH) for _ in range(self.io_threads)
        ]
        self._pool = (
            None
            if self.io_threads == 1
            else ThreadPoolExecutor(
                max_workers=self.io_threads, thread_name_prefix="hicache-l3-read"
            )
        )
        # Today only prefetch_io_aux_thread calls in, so batches never overlap.
        # The lock makes that an invariant of this class instead of an
        # assumption about the caller: a second concurrent batch would hand two
        # threads the same context.
        self._batch_lock = threading.Lock()
        self._bounce = BouncePool(alignment=profile.memory_alignment)
        self._open_flags = os.O_RDONLY | (
            os.O_DIRECT if profile.direct_io_available else 0
        )
        logger.info(
            "HiCacheLayerwiseFile at %s: direct_io=%s alignment=%d page=%.2f MiB "
            "io_threads=%d",
            self.root,
            profile.direct_io_available,
            profile.alignment,
            self.layout.physical_nbytes / (1 << 20),
            self.io_threads,
        )

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Read whole pages into the host pool, one asynchronous batch."""
        if not keys:
            return []
        results = [True] * len(keys)
        shards = _split_positions(len(keys), self.io_threads)
        with self._batch_lock:
            if len(shards) == 1:
                self._read_shard(
                    self._contexts[0], keys, host_indices, shards[0], results
                )
            else:
                futures = [
                    self._pool.submit(
                        self._read_shard,
                        self._contexts[index],
                        keys,
                        host_indices,
                        shard,
                        results,
                    )
                    for index, shard in enumerate(shards)
                ]
                # Drain every future before re-raising: each worker releases its
                # own descriptors in its own finally, but only if it is allowed
                # to finish.
                error = None
                for future in futures:
                    try:
                        future.result()
                    except Exception as exception:  # noqa: BLE001 - re-raised below
                        error = error or exception
                if error is not None:
                    raise error
        return _truncate_at_first_failure(results)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        results = []
        for position, key in enumerate(keys):
            if self.writer.exists(key):
                results.append(True)
                continue
            k_ptr, v_ptr = self._page_pointers(host_indices, position)
            results.append(self.writer.write_page(key, k_ptr=k_ptr, v_ptr=v_ptr))
        return results

    def exists(self, key: str) -> bool:
        return self.writer.exists(key)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        for position, key in enumerate(keys):
            if not self.writer.exists(key):
                return position
        return len(keys)

    def get(self, key, target_location=None, target_sizes=None):
        raise NotImplementedError(
            "layerwise_file serves the zero-copy batch interface only"
        )

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise NotImplementedError(
            "layerwise_file serves the zero-copy batch interface only"
        )

    def set(self, key, value=None, target_location=None, target_sizes=None):
        raise NotImplementedError(
            "layerwise_file serves the zero-copy batch interface only"
        )

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise NotImplementedError(
            "layerwise_file serves the zero-copy batch interface only"
        )

    def close(self) -> None:
        # Under the batch lock so teardown cannot destroy a context that a read
        # in flight is still draining.
        with self._batch_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        for context in self._contexts:
            context.close()

    def _read_shard(
        self,
        context: LinuxAioContext,
        keys: List[str],
        host_indices: torch.Tensor,
        positions: range,
        results: List[bool],
    ) -> None:
        """Read one worker's slice of the batch on its own context.

        ``results`` is shared with the other workers, but the slices are
        disjoint and each worker writes only its own positions, so no entry is
        ever written twice.
        """
        requests, records, failures = self._plan_batch(keys, host_indices, positions)
        for position in failures:
            results[position] = False
        try:
            if requests:
                self._run_batch(context, requests, records, results)
        finally:
            self._release(records)

    def _plan_batch(
        self, keys: List[str], host_indices: torch.Tensor, positions: range
    ):
        """Turn each page into two extents, bouncing only what cannot land."""
        memory_alignment = self.alignment_profile.memory_alignment
        requests = []
        records = {}
        failures = set()
        for position in positions:
            key = keys[position]
            path = self.writer.page_path(key)
            try:
                fd = os.open(path, self._open_flags)
            except OSError:
                failures.add(position)
                continue
            k_ptr, v_ptr = self._page_pointers(host_indices, position)
            for target_ptr, file_offset in (
                (k_ptr, self.layout.k_offset),
                (v_ptr, self.layout.v_offset),
            ):
                user_data = len(records) + 1
                bounce = (
                    None
                    if target_ptr % memory_alignment == 0
                    else self._bounce.acquire(self._region_nbytes)
                )
                records[user_data] = (position, fd, bounce, target_ptr)
                requests.append(
                    (
                        fd,
                        bounce.ptr if bounce is not None else target_ptr,
                        self._region_nbytes,
                        file_offset,
                        user_data,
                    )
                )
        return requests, records, failures

    def _run_batch(self, context, requests, records, results) -> None:
        pending = list(requests)
        outstanding = 0
        while pending or outstanding:
            if pending:
                accepted = context.submit_reads(pending)
                outstanding += accepted
                pending = pending[accepted:]
                if accepted == 0 and outstanding == 0:
                    raise RuntimeError("layerwise_file could not submit any read")
            for completion in context.wait(min_events=1 if outstanding else 0):
                outstanding -= 1
                position, _, bounce, target_ptr = records[completion.user_data]
                if completion.result != self._region_nbytes:
                    logger.warning(
                        "layerwise_file short or failed read for page %d: %s",
                        position,
                        completion.error or completion.result,
                    )
                    results[position] = False
                    continue
                if bounce is not None:
                    bounce.copy_out(
                        src_offset=0, nbytes=self._region_nbytes, dst_ptr=target_ptr
                    )

    def _release(self, records) -> None:
        # The two extents of a page share one descriptor, so close by identity;
        # a double close could land on a number another thread just reused.
        descriptors = set()
        for _, fd, bounce, _ in records.values():
            descriptors.add(fd)
            if bounce is not None:
                self._bounce.release(bounce)
        for fd in descriptors:
            try:
                os.close(fd)
            except OSError:
                pass

    def _page_pointers(self, host_indices: torch.Tensor, position: int):
        first_token = int(host_indices[position * self.page_size])
        page_index = first_token // self.page_size
        k_buffer = self.mem_pool_host.k_buffer
        v_buffer = self.mem_pool_host.v_buffer
        page_stride = k_buffer.stride(0) * k_buffer.element_size()
        offset = page_index * page_stride
        return k_buffer.data_ptr() + offset, v_buffer.data_ptr() + offset


def _split_positions(total: int, parts: int) -> List[range]:
    """Split ``total`` pages into at most ``parts`` contiguous slices.

    Contiguous rather than round-robin so a worker walks the batch in order,
    and never more slices than pages so a small batch does not hand a worker an
    empty one.
    """
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    slices = []
    start = 0
    for index in range(parts):
        stop = start + base + (1 if index < extra else 0)
        slices.append(range(start, stop))
        start = stop
    return slices


def _resolve_io_threads() -> int:
    """Submission width for the read path, from the published config.

    Falls back to one outside a published process (unit tests constructing the
    backend directly), which is the pre-existing behavior.
    """
    try:
        threads = int(get_memory().hicache_storage_io_threads)
    except ValueError:
        # "config namespace 'memory' not published": a bare unit test built the
        # backend without a published process. Every server path has published.
        return 1
    return max(1, threads)


def _truncate_at_first_failure(results: List[bool]) -> List[bool]:
    """A prefix is only usable up to its first gap, so drop everything after."""
    for position, ok in enumerate(results):
        if not ok:
            return results[:position] + [False] * (len(results) - position)
    return results


def _resolve_root(storage_config: HiCacheStorageConfig) -> str:
    extra = storage_config.extra_config or {}
    if extra.get("layerwise_root"):
        return str(extra["layerwise_root"])
    return envs.SGLANG_HICACHE_LAYERWISE_ROOT.get()


def _resolve_require_direct(storage_config: HiCacheStorageConfig) -> bool:
    extra = storage_config.extra_config or {}
    if "require_direct_io" in extra:
        return bool(extra["require_direct_io"])
    return True
