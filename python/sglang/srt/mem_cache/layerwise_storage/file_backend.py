"""Direct I/O page-file backend for the layerwise HiCache storage tier.

One logical page is one file whose payload matches the host pool's
``page_first_direct`` flat page, so an aligned layer range on disk maps onto a
compact host slice with no repacking.  Reads land straight in the host KV
buffer whenever the extent is aligned; only an unaligned extent pays for a
bounce buffer, and that buffer comes from a pool rather than the read path.

This is the bring-up backend from the plan's P1/P2: it proves the format, the
alignment rules and the async contract on a local filesystem before the same
interface is pointed at the target parallel filesystem.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
from typing import Any, NamedTuple, Optional

from sglang.srt.mem_cache.layerwise_storage.aio_engine import (
    AlignedBuffer,
    AlignmentProfile,
    DirectIOFileCache,
    LinuxAioContext,
    probe_alignment,
)
from sglang.srt.mem_cache.layerwise_storage.backend import LayerwiseStorageBackend
from sglang.srt.mem_cache.layerwise_storage.io_arbiter import IoArbiter, IoPriority
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    PageIdentity,
    page_relative_path,
)
from sglang.srt.mem_cache.layerwise_storage.types import (
    CancelLevel,
    CancelRequestDisposition,
    CancelRequestResult,
    ExtentCompletionStatus,
    HandleTerminalStatus,
    HostTargetBase,
    LayerGroupPlan,
    LayerwiseBackendCapabilities,
    LayerwiseGroupTicket,
    LayerwiseReadHandle,
    LayerwiseReadPlan,
    LayerwiseStorageCompletion,
    LayerwiseStorageExtent,
    validate_group_against_capabilities,
)

logger = logging.getLogger(__name__)

# How long a read shard sleeps when it has nothing to do. Submissions wake it,
# so this only bounds how late an in-flight completion is noticed.
_WORKER_IDLE_S = 0.0005

_PRIORITY_LEVELS = (
    IoPriority.ADMISSION,
    IoPriority.DEMAND,
    IoPriority.READ_AHEAD,
    IoPriority.WRITE_BACK,
)


class BouncePool:
    """Reusable aligned buffers for extents that cannot land in place.

    Buffers are leased per size class and returned on completion.  Growth only
    happens when a plan asks for more than the pool holds, which is a
    configuration event rather than a per-read cost.
    """

    def __init__(self, *, alignment: int):
        self.alignment = alignment
        self._lock = threading.Lock()
        self._free: dict[int, list[AlignedBuffer]] = {}
        self.allocated_nbytes = 0

    def acquire(self, nbytes: int) -> AlignedBuffer:
        with self._lock:
            pool = self._free.get(nbytes)
            if pool:
                return pool.pop()
            self.allocated_nbytes += nbytes
        return AlignedBuffer(nbytes, alignment=self.alignment)

    def release(self, buffer: AlignedBuffer) -> None:
        with self._lock:
            self._free.setdefault(buffer.nbytes, []).append(buffer)

    def reserve(self, *, nbytes: int, count: int) -> None:
        buffers = [self.acquire(nbytes) for _ in range(count)]
        for buffer in buffers:
            self.release(buffer)


class _InflightExtent(NamedTuple):
    transaction_id: str
    generation: int
    group_id: int
    extent_id: int
    path: str
    io_nbytes: int
    bounce: Optional[AlignedBuffer]
    bounce_src_offset: int
    payload_nbytes: int
    target_ptr: int


class _Shard:
    """One AIO context, its arbiter, and the single thread allowed to drive them.

    A context owns a completion buffer and a dict of live iocbs, neither safe to
    drive from two threads, so a shard is never shared. Sharding is what lifts
    the read off the single-submitter ceiling: one thread batching into a deep
    queue tops out far below what a parallel filesystem can serve, no matter how
    deep the queue is.
    """

    def __init__(self, *, queue_depth: int):
        self.context = LinuxAioContext(queue_depth=queue_depth)
        self.arbiter = IoArbiter(context=self.context)
        self.wake = threading.Event()
        self.thread: Optional[threading.Thread] = None


def _split_positions(total: int, parts: int) -> list[range]:
    """Contiguous, near-equal ranges; a page is never split across shards."""
    if parts <= 1 or total <= 0:
        return [range(total)]
    base, extra = divmod(total, parts)
    ranges = []
    start = 0
    for index in range(parts):
        span = base + (1 if index < extra else 0)
        ranges.append(range(start, start + span))
        start += span
    return ranges


class _ReadHandleState:
    """Per-transaction bookkeeping owned by the backend."""

    def __init__(self, *, plan: LayerwiseReadPlan, target: HostTargetBase):
        self.plan = plan
        self.target = target
        self.shard_of_page: dict[str, int] = {}
        self.submitted_groups: set[int] = set()
        self.pending_extents = 0
        self.completions: list[LayerwiseStorageCompletion] = []
        self.failed = False
        self.cancelled = False
        self.terminal_extents = 0


class LayerwiseFileBackend(LayerwiseStorageBackend):
    def __init__(
        self,
        *,
        root: str,
        identity: PageIdentity,
        queue_depth: int = 128,
        io_threads: int = 1,
        fd_cache_capacity: int = 1024,
        alignment_profile: Optional[AlignmentProfile] = None,
        require_direct_io: bool = True,
    ):
        self.root = os.path.abspath(root)
        self.identity = identity
        os.makedirs(self.root, exist_ok=True)
        self.alignment_profile = alignment_profile or probe_alignment(
            self.root, require_direct=require_direct_io
        )
        if require_direct_io and not self.alignment_profile.direct_io_available:
            raise RuntimeError(
                f"O_DIRECT is required but unavailable under {self.root!r}"
            )

        self._io_threads = max(1, io_threads)
        self._shards = [
            _Shard(queue_depth=queue_depth) for _ in range(self._io_threads)
        ]
        self._stop = threading.Event()
        self._files = DirectIOFileCache(capacity=fd_cache_capacity)
        self._bounce = BouncePool(alignment=self.alignment_profile.memory_alignment)
        self._capabilities = LayerwiseBackendCapabilities(
            required_alignment=self.alignment_profile.alignment,
            supports_range_read=True,
            supports_direct_to_host=True,
            max_inflight_groups=max(1, queue_depth // 2),
            max_inflight_extents=queue_depth,
            max_iov=queue_depth,
            cancel_level=CancelLevel.BOUNDED_TERMINAL,
        )

        self._lock = threading.Lock()
        self._handles: dict[str, _ReadHandleState] = {}
        self._inflight: dict[int, _InflightExtent] = {}
        self._user_data = itertools.count(1)

        # One shard keeps the old inline path exactly: the caller submits and
        # drains on its own thread, and no worker exists to race it.
        if self._io_threads > 1:
            for index, shard in enumerate(self._shards):
                shard.thread = threading.Thread(
                    target=self._worker,
                    args=(shard,),
                    name=f"layerwise-l3-read-{index}",
                    daemon=True,
                )
                shard.thread.start()
        logger.info(
            "LayerwiseFileBackend at %s: io_threads=%d queue_depth=%d",
            self.root,
            self._io_threads,
            queue_depth,
        )

    def capabilities(self) -> LayerwiseBackendCapabilities:
        return self._capabilities

    def begin_read(
        self,
        *,
        transaction_id: str,
        generation: int,
        plan: LayerwiseReadPlan,
        target: Any,
    ) -> LayerwiseReadHandle:
        if not isinstance(target, HostTargetBase):
            raise TypeError("target must be a HostTargetBase")
        for group in plan.groups:
            validate_group_against_capabilities(
                group=group, capabilities=self._capabilities
            )
        state = _ReadHandleState(plan=plan, target=target)
        # Assign every page to one shard up front, in plan order: a page file is
        # then touched by one thread only, so its fd is not opened twice and two
        # contexts never read the same file.
        pages = list(
            dict.fromkeys(
                extent.storage_key for group in plan.groups for extent in group.extents
            )
        )
        for shard_index, positions in enumerate(
            _split_positions(len(pages), self._io_threads)
        ):
            for position in positions:
                state.shard_of_page[pages[position]] = shard_index
        with self._lock:
            if transaction_id in self._handles:
                raise RuntimeError(f"transaction {transaction_id!r} is already open")
            self._handles[transaction_id] = state
        return LayerwiseReadHandle(
            transaction_id=transaction_id,
            generation=generation,
            backend_token=None,
        )

    def submit_group(
        self,
        *,
        handle: LayerwiseReadHandle,
        group: LayerGroupPlan,
        priority: int,
        deadline_s: Optional[float],
    ) -> LayerwiseGroupTicket:
        state = self._state(handle)
        with self._lock:
            if group.group_id in state.submitted_groups:
                raise RuntimeError(f"group {group.group_id} was already submitted")
            state.submitted_groups.add(group.group_id)
            state.pending_extents += len(group.extents)

        io_priority = _PRIORITY_LEVELS[min(priority, len(_PRIORITY_LEVELS) - 1)]
        for extent in group.extents:
            self._enqueue_extent(
                handle=handle,
                state=state,
                group_id=group.group_id,
                extent=extent,
                priority=io_priority,
            )
        self._wake_or_pump()
        return LayerwiseGroupTicket(
            handle=handle, group_id=group.group_id, backend_token=None
        )

    def poll(
        self,
        *,
        handle: LayerwiseReadHandle,
        max_completions: Optional[int] = None,
    ) -> tuple[LayerwiseStorageCompletion, ...]:
        self._drain_if_inline()
        state = self._state(handle)
        with self._lock:
            if max_completions is None:
                drained = state.completions
                state.completions = []
            else:
                drained = state.completions[:max_completions]
                state.completions = state.completions[max_completions:]
        return tuple(drained)

    def request_cancel(
        self,
        *,
        handle: LayerwiseReadHandle,
        group_ids: tuple[int, ...],
    ) -> tuple[CancelRequestResult, ...]:
        state = self._state(handle)
        wanted = set(group_ids)
        with self._lock:
            state.cancelled = True
            queued_ids = [
                user_data
                for user_data, extent in self._inflight.items()
                if extent.transaction_id == handle.transaction_id
                and extent.group_id in wanted
            ]
        removed = set()
        for shard in self._shards:
            removed.update(shard.arbiter.cancel_queued(queued_ids))
        for user_data in removed:
            self._retire_extent(user_data, status=ExtentCompletionStatus.CANCELLED)

        results = []
        for group_id in group_ids:
            with self._lock:
                still_inflight = any(
                    extent.transaction_id == handle.transaction_id
                    and extent.group_id == group_id
                    for extent in self._inflight.values()
                )
            results.append(
                CancelRequestResult(
                    group_id=group_id,
                    disposition=(
                        CancelRequestDisposition.ACCEPTED
                        if still_inflight
                        else CancelRequestDisposition.ALREADY_TERMINAL
                    ),
                )
            )
        return tuple(results)

    def poll_terminal(self, *, handle: LayerwiseReadHandle) -> HandleTerminalStatus:
        self._drain_if_inline()
        state = self._state(handle)
        with self._lock:
            if state.pending_extents > 0:
                return HandleTerminalStatus.ACTIVE
            if state.failed:
                return HandleTerminalStatus.FAILED
            if state.cancelled:
                return HandleTerminalStatus.CANCELLED
            return HandleTerminalStatus.SUCCEEDED

    def close(self, *, handle: LayerwiseReadHandle) -> None:
        with self._lock:
            state = self._handles.get(handle.transaction_id)
            if state is None:
                return
            if state.pending_extents > 0:
                raise RuntimeError(
                    f"transaction {handle.transaction_id!r} still has "
                    f"{state.pending_extents} operations that can touch its target"
                )
            del self._handles[handle.transaction_id]

    def shutdown(self) -> None:
        self._stop.set()
        for shard in self._shards:
            shard.wake.set()
        for shard in self._shards:
            if shard.thread is not None:
                shard.thread.join(timeout=30)
        for shard in self._shards:
            shard.context.close()
        self._files.close()

    def page_path(self, page_key: str) -> str:
        return os.path.join(
            self.root,
            page_relative_path(
                fingerprint=self.identity.fingerprint,
                tp_size=self.identity.tp_size,
                tp_rank=self.identity.tp_rank,
                page_key=page_key,
            ),
        )

    def _state(self, handle: LayerwiseReadHandle) -> _ReadHandleState:
        with self._lock:
            state = self._handles.get(handle.transaction_id)
        if state is None:
            raise KeyError(f"unknown transaction {handle.transaction_id!r}")
        return state

    def _enqueue_extent(
        self,
        *,
        handle: LayerwiseReadHandle,
        state: _ReadHandleState,
        group_id: int,
        extent: LayerwiseStorageExtent,
        priority: IoPriority,
    ) -> None:
        path = self.page_path(extent.storage_key)
        target_ptr = state.target.base_for(extent.kv_part) + extent.target_offset
        memory_alignment = self.alignment_profile.memory_alignment
        needs_bounce = (
            extent.payload_offset != 0
            or extent.payload_nbytes != extent.io_nbytes
            or target_ptr % memory_alignment != 0
        )
        try:
            fd = self._files.acquire(
                path, direct=self.alignment_profile.direct_io_available
            )
        except OSError as error:
            self._record_completion(
                state=state,
                transaction_id=handle.transaction_id,
                generation=handle.generation,
                group_id=group_id,
                extent_id=extent.extent_id,
                status=ExtentCompletionStatus.FAILED,
                error=f"open failed: {error}",
            )
            return

        bounce = self._bounce.acquire(extent.io_nbytes) if needs_bounce else None
        user_data = next(self._user_data)
        record = _InflightExtent(
            transaction_id=handle.transaction_id,
            generation=handle.generation,
            group_id=group_id,
            extent_id=extent.extent_id,
            path=path,
            io_nbytes=extent.io_nbytes,
            bounce=bounce,
            bounce_src_offset=extent.payload_offset,
            payload_nbytes=extent.payload_nbytes,
            target_ptr=target_ptr,
        )
        with self._lock:
            self._inflight[user_data] = record
        shard = self._shards[state.shard_of_page.get(extent.storage_key, 0)]
        shard.arbiter.enqueue(
            fd=fd,
            ptr=bounce.ptr if bounce is not None else target_ptr,
            nbytes=extent.io_nbytes,
            offset=extent.io_offset,
            user_data=user_data,
            priority=priority,
        )

    def _wake_or_pump(self) -> None:
        """Hand the new work to the shard threads, or submit it here if alone."""
        if self._io_threads == 1:
            self._shards[0].arbiter.pump()
            return
        for shard in self._shards:
            shard.wake.set()

    def _drain_if_inline(self) -> None:
        # With workers running, only they may touch a context; the caller reads
        # what they have already recorded.
        if self._io_threads == 1:
            self._drain_shard(self._shards[0])

    def _worker(self, shard: _Shard) -> None:
        """Submit and drain one shard, and nothing else, for this shard's life."""
        while not self._stop.is_set():
            shard.wake.wait(_WORKER_IDLE_S)
            shard.wake.clear()
            try:
                shard.arbiter.pump()
                self._drain_shard(shard)
            except Exception:
                # A worker that dies here would strand its shard's extents with
                # no completion ever recorded, and the pipeline would only see a
                # group timeout with nothing to point at.
                logger.exception("layerwise read shard failed")

    def _drain_shard(self, shard: _Shard) -> None:
        for completion in shard.arbiter.poll():
            if completion.failed:
                self._retire_extent(
                    completion.user_data,
                    status=ExtentCompletionStatus.FAILED,
                    error=str(completion.error),
                )
                continue
            self._finish_extent(completion.user_data, transferred=completion.result)

    def _finish_extent(self, user_data: int, *, transferred: int) -> None:
        with self._lock:
            record = self._inflight.get(user_data)
        if record is None:
            return
        if transferred != record.io_nbytes:
            self._retire_extent(
                user_data,
                status=ExtentCompletionStatus.FAILED,
                error=f"short read: {transferred} of {record.io_nbytes} bytes",
                bytes_transferred=transferred,
            )
            return
        if record.bounce is not None:
            record.bounce.copy_out(
                src_offset=record.bounce_src_offset,
                nbytes=record.payload_nbytes,
                dst_ptr=record.target_ptr,
            )
        self._retire_extent(
            user_data,
            status=ExtentCompletionStatus.SUCCEEDED,
            bytes_transferred=transferred,
        )

    def _retire_extent(
        self,
        user_data: int,
        *,
        status: ExtentCompletionStatus,
        error: Optional[str] = None,
        bytes_transferred: int = 0,
    ) -> None:
        with self._lock:
            record = self._inflight.pop(user_data, None)
            if record is None:
                return
            state = self._handles.get(record.transaction_id)
        self._files.release(record.path)
        if record.bounce is not None:
            self._bounce.release(record.bounce)
        if state is None:
            return
        self._record_completion(
            state=state,
            transaction_id=record.transaction_id,
            generation=record.generation,
            group_id=record.group_id,
            extent_id=record.extent_id,
            status=status,
            error=error,
            bytes_transferred=bytes_transferred,
        )

    def _record_completion(
        self,
        *,
        state: _ReadHandleState,
        transaction_id: str,
        generation: int,
        group_id: int,
        extent_id: int,
        status: ExtentCompletionStatus,
        error: Optional[str] = None,
        bytes_transferred: int = 0,
    ) -> None:
        with self._lock:
            state.pending_extents -= 1
            state.terminal_extents += 1
            if status is ExtentCompletionStatus.FAILED:
                state.failed = True
            elif status is ExtentCompletionStatus.CANCELLED:
                state.cancelled = True
            state.completions.append(
                LayerwiseStorageCompletion(
                    transaction_id=transaction_id,
                    generation=generation,
                    group_id=group_id,
                    extent_id=extent_id,
                    status=status,
                    bytes_transferred=bytes_transferred,
                    error=error,
                )
            )
