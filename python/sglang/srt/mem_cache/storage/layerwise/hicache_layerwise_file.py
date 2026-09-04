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
"""

from __future__ import annotations

import logging
import os
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
        self._context = LinuxAioContext(queue_depth=_QUEUE_DEPTH)
        self._bounce = BouncePool(alignment=profile.memory_alignment)
        self._open_flags = os.O_RDONLY | (
            os.O_DIRECT if profile.direct_io_available else 0
        )
        logger.info(
            "HiCacheLayerwiseFile at %s: direct_io=%s alignment=%d page=%.2f MiB",
            self.root,
            profile.direct_io_available,
            profile.alignment,
            self.layout.physical_nbytes / (1 << 20),
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
        requests, records, failures = self._plan_batch(keys, host_indices)
        results = [key_index not in failures for key_index in range(len(keys))]
        try:
            if requests:
                self._run_batch(requests, records, results)
        finally:
            self._release(records)
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
        self._context.close()

    def _plan_batch(self, keys: List[str], host_indices: torch.Tensor):
        """Turn each page into two extents, bouncing only what cannot land."""
        memory_alignment = self.alignment_profile.memory_alignment
        requests = []
        records = {}
        failures = set()
        for position, key in enumerate(keys):
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

    def _run_batch(self, requests, records, results) -> None:
        pending = list(requests)
        outstanding = 0
        while pending or outstanding:
            if pending:
                accepted = self._context.submit_reads(pending)
                outstanding += accepted
                pending = pending[accepted:]
                if accepted == 0 and outstanding == 0:
                    raise RuntimeError("layerwise_file could not submit any read")
            for completion in self._context.wait(min_events=1 if outstanding else 0):
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
