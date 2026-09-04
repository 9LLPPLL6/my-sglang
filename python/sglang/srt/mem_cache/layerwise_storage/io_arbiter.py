"""Priority arbitration for the shared HiCache L3 I/O queue.

Demand reads and write-through traffic contend for the same device queue.
Without arbitration a burst of write-through pages can sit in front of the one
read that a request is blocked on, which shows up as a TTFT outlier rather than
a bandwidth problem.  The arbiter therefore owns the only submission path into
the AIO context and applies four rules:

* the admission-critical first group keeps reserved queue slots, so it never
  waits behind read-ahead that no one is blocked on;
* read-ahead yields to demand reads;
* write-through is capped to a small share of the queue while any read is
  pending, and may use the whole queue when no read is;
* a write that has waited longer than the aging threshold outranks read-ahead,
  so a steady read stream cannot starve the write backlog forever.

Byte accounting runs alongside slot accounting because one layer group can
expand into hundreds of extents: a queue-depth limit alone does not bound how
much memory the kernel has in flight.
"""

from __future__ import annotations

import enum
import threading
import time
from collections import deque
from typing import NamedTuple, Optional

from sglang.srt.mem_cache.layerwise_storage.aio_engine import (
    AioCompletion,
    LinuxAioContext,
)


class IoPriority(enum.IntEnum):
    """Lower values are submitted first."""

    ADMISSION = 0
    DEMAND = 1
    READ_AHEAD = 2
    WRITE_BACK = 3


class _QueuedIo(NamedTuple):
    fd: int
    ptr: int
    nbytes: int
    offset: int
    user_data: int
    priority: IoPriority
    is_write: bool
    enqueued_at: float


class ArbiterStats(NamedTuple):
    queued: int
    inflight: int
    inflight_nbytes: int
    submitted: int
    completed: int


class IoArbiter:
    """Bounded, priority-ordered submission in front of one AIO context."""

    def __init__(
        self,
        *,
        context: LinuxAioContext,
        max_inflight_bytes: int,
        admission_reserved_slots: int = 2,
        write_share: float = 0.25,
        write_aging_s: float = 0.5,
    ):
        if max_inflight_bytes <= 0:
            raise ValueError(
                f"max_inflight_bytes must be positive, got {max_inflight_bytes}"
            )
        if not 0.0 < write_share <= 1.0:
            raise ValueError(f"write_share must be in (0, 1], got {write_share}")
        if admission_reserved_slots < 0:
            raise ValueError("admission_reserved_slots must be non-negative")

        self._context = context
        self.max_inflight_bytes = max_inflight_bytes
        self.admission_reserved_slots = min(
            admission_reserved_slots, context.queue_depth
        )
        self.write_share = write_share
        self.write_aging_s = write_aging_s

        self._lock = threading.Lock()
        self._queues = {priority: deque() for priority in IoPriority}
        self._inflight_nbytes = 0
        self._inflight_writes = 0
        self._inflight_by_id: dict[int, tuple[int, bool]] = {}
        self._submitted = 0
        self._completed = 0

    def enqueue(
        self,
        *,
        fd: int,
        ptr: int,
        nbytes: int,
        offset: int,
        user_data: int,
        priority: IoPriority,
        is_write: bool = False,
    ) -> None:
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        if nbytes > self.max_inflight_bytes:
            raise ValueError(
                f"a single {nbytes}-byte operation cannot fit the "
                f"{self.max_inflight_bytes}-byte in-flight budget"
            )
        item = _QueuedIo(
            fd=fd,
            ptr=ptr,
            nbytes=nbytes,
            offset=offset,
            user_data=user_data,
            priority=priority,
            is_write=is_write,
            enqueued_at=time.monotonic(),
        )
        with self._lock:
            self._queues[priority].append(item)

    def pump(self) -> int:
        """Submit as many queued operations as the current budget allows."""
        reads, writes = self._take_submittable()
        submitted = 0
        if reads:
            submitted += self._submit_batch(reads, is_write=False)
        if writes:
            submitted += self._submit_batch(writes, is_write=True)
        return submitted

    def poll(self, *, max_events: Optional[int] = None) -> tuple[AioCompletion, ...]:
        """Drain completions, release their budget, then refill the queue."""
        completions = self._context.poll(max_events=max_events)
        if completions:
            with self._lock:
                for completion in completions:
                    nbytes, is_write = self._inflight_by_id.pop(
                        completion.user_data, (0, False)
                    )
                    self._inflight_nbytes -= nbytes
                    if is_write:
                        self._inflight_writes -= 1
                self._completed += len(completions)
        self.pump()
        return completions

    def cancel_queued(self, user_data_values) -> tuple[int, ...]:
        """Drop not-yet-submitted operations; in-flight ones are untouched.

        Returns the ids actually removed.  An id that is absent was already
        submitted, and its buffer stays owned by the kernel until it completes.
        """
        wanted = set(user_data_values)
        removed = []
        with self._lock:
            for priority in IoPriority:
                queue = self._queues[priority]
                kept = deque()
                for item in queue:
                    if item.user_data in wanted:
                        removed.append(item.user_data)
                    else:
                        kept.append(item)
                self._queues[priority] = kept
        return tuple(removed)

    def stats(self) -> ArbiterStats:
        with self._lock:
            queued = sum(len(queue) for queue in self._queues.values())
            return ArbiterStats(
                queued=queued,
                inflight=self._context.inflight,
                inflight_nbytes=self._inflight_nbytes,
                submitted=self._submitted,
                completed=self._completed,
            )

    def _take_submittable(self) -> tuple[list, list]:
        now = time.monotonic()
        reads = []
        writes = []
        with self._lock:
            slots = self._context.free_slots
            byte_budget = self.max_inflight_bytes - self._inflight_nbytes
            read_pending = any(
                self._queues[priority]
                for priority in (
                    IoPriority.ADMISSION,
                    IoPriority.DEMAND,
                    IoPriority.READ_AHEAD,
                )
            )
            write_slots = self._write_slot_budget_locked(read_pending=read_pending)

            for priority in self._submission_order_locked(now=now):
                queue = self._queues[priority]
                while queue and slots > 0:
                    item = queue[0]
                    if item.nbytes > byte_budget:
                        break
                    if item.is_write and write_slots <= 0:
                        break
                    if (
                        not item.is_write
                        and priority is not IoPriority.ADMISSION
                        and slots <= self._admission_reserve_locked()
                    ):
                        break
                    queue.popleft()
                    slots -= 1
                    byte_budget -= item.nbytes
                    if item.is_write:
                        write_slots -= 1
                        writes.append(item)
                    else:
                        reads.append(item)
        return reads, writes

    def _submission_order_locked(self, *, now: float) -> tuple[IoPriority, ...]:
        aged_write = self._queues[IoPriority.WRITE_BACK] and (
            now - self._queues[IoPriority.WRITE_BACK][0].enqueued_at
            >= self.write_aging_s
        )
        if aged_write:
            return (
                IoPriority.ADMISSION,
                IoPriority.DEMAND,
                IoPriority.WRITE_BACK,
                IoPriority.READ_AHEAD,
            )
        return tuple(IoPriority)

    def _admission_reserve_locked(self) -> int:
        """Slots held back so a later admission read never finds a full queue."""
        if self._queues[IoPriority.ADMISSION]:
            return 0
        return self.admission_reserved_slots

    def _write_slot_budget_locked(self, *, read_pending: bool) -> int:
        depth = self._context.queue_depth
        cap = depth if not read_pending else max(1, int(depth * self.write_share))
        return max(0, cap - self._inflight_writes)

    def _submit_batch(self, items: list, *, is_write: bool) -> int:
        requests = [
            (item.fd, item.ptr, item.nbytes, item.offset, item.user_data)
            for item in items
        ]
        if is_write:
            accepted = self._context.submit_writes(requests)
        else:
            accepted = self._context.submit_reads(requests)

        with self._lock:
            for item in items[:accepted]:
                self._inflight_by_id[item.user_data] = (item.nbytes, is_write)
                self._inflight_nbytes += item.nbytes
                if is_write:
                    self._inflight_writes += 1
            self._submitted += accepted
            # Anything the kernel refused goes back to the head of its own
            # queue, so back pressure never demotes an admission read.
            for item in reversed(items[accepted:]):
                self._queues[item.priority].appendleft(item)
        return accepted
