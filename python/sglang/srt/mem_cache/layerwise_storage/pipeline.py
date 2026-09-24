"""Drives one storage-backed prefix hit through the layerwise pipeline.

The pipeline is what makes an L3 hit look like an L2 hit to the layers above:
while the GPU computes layer group ``g-1`` and the H2D stream copies group
``g``, storage is already reading group ``g+1``.  Everything here runs on the
scheduler thread and never blocks; ``advance`` is a single non-blocking step
that drains what finished and submits what the budget allows.

Three orderings are load bearing and deliberately not relaxed:

* storage completions may arrive in any order, but cross-rank agreement is
  started strictly in group order, or ranks deadlock disagreeing about which
  collective comes next;
* a group reaches the H2D stream only after it is agreed everywhere, so a rank
  that lost a page can never publish partly-correct KV;
* group 0 is submitted alone and gates admission, because it is the one group
  no computation can hide.

Ownership is resolved exactly once per allocation.  A failed read does not
release its staging until the backend reports the operation terminal, so a
freed host page can never be overwritten by an I/O the kernel had not finished.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, NamedTuple, Optional

from sglang.srt.mem_cache.layerwise_storage.backend import LayerwiseStorageBackend
from sglang.srt.mem_cache.layerwise_storage.consensus import GroupConsensus
from sglang.srt.mem_cache.layerwise_storage.state_machine import (
    GroupState,
    LayerwiseTransactionStateMachine,
    OwnershipState,
    PrivateBufferOwnership,
    TransactionState,
)
from sglang.srt.mem_cache.layerwise_storage.types import (
    CancelLevel,
    HandleTerminalStatus,
    LayerwiseBackendCapabilities,
    LayerwiseReadHandle,
    LayerwiseReadPlan,
)

logger = logging.getLogger(__name__)


_ADMISSION_PRIORITY = 0
_READ_AHEAD_PRIORITY = 2


class PipelineConfig(NamedTuple):
    group_timeout_s: float
    admission_budget_s: float


class LayerwiseTransaction:
    """Runtime state of one storage-backed hit, owned by the scheduler thread."""

    def __init__(
        self,
        *,
        transaction_id: str,
        generation: int,
        plan: LayerwiseReadPlan,
        handle: LayerwiseReadHandle,
        host_ownership: PrivateBufferOwnership,
    ):
        self.transaction_id = transaction_id
        self.generation = generation
        self.plan = plan
        self.handle = handle
        self.machine = LayerwiseTransactionStateMachine(
            transaction_id=transaction_id, generation=generation, plan=plan
        )
        self.host_ownership = host_ownership
        self.device_ownership: Optional[PrivateBufferOwnership] = None
        self.started_at = time.monotonic()
        self.admitted_at: Optional[float] = None
        self.error: Optional[str] = None
        self.next_submit_group_id = 0
        self.next_consensus_group_id = 0
        self.next_h2d_group_id = 0
        self.group_deadlines: dict[int, float] = {}
        # Wall clock of the few state changes that decide when the request is
        # admitted. The design admits on group 0, but measured admission lands
        # with the whole read instead, and that gap cannot be read off the
        # backend's own report: it is a question about when the scheduler
        # thread got around to looking, not about the I/O. Same clock as the
        # per-request time stats, so the two logs subtract.
        self.marks: dict[str, float] = {}

    def mark(self, event: str) -> None:
        """First occurrence wins; a later group must not overwrite group 0's."""
        self.marks.setdefault(event, time.perf_counter())

    @property
    def group_count(self) -> int:
        return len(self.plan.groups)

    @property
    def admission_ready(self) -> bool:
        return self.machine.state not in (
            TransactionState.NEW,
            TransactionState.QUERYING_L3,
            TransactionState.HOST_PRIVATE_ALLOCATED,
            TransactionState.READING_GROUP0,
            TransactionState.ABORTING,
            TransactionState.ABORTED,
        )

    @property
    def aborted(self) -> bool:
        return self.machine.state in (
            TransactionState.ABORTING,
            TransactionState.ABORTED,
        )


class LayerwiseStoragePipeline:
    """Non-blocking scheduler-side driver for layerwise storage transactions."""

    def __init__(
        self,
        *,
        backend: LayerwiseStorageBackend,
        consensus: GroupConsensus,
        config: PipelineConfig,
        submit_h2d_range: Callable[[LayerwiseTransaction, int, int], None],
    ):
        capabilities = backend.capabilities()
        _reject_unusable_backend(capabilities)

        self._backend = backend
        self._consensus = consensus
        self._config = config
        self._submit_h2d_range = submit_h2d_range
        self._capabilities = capabilities

    def begin(
        self,
        *,
        transaction_id: str,
        generation: int,
        plan: LayerwiseReadPlan,
        target,
        host_resource_id: str,
    ) -> LayerwiseTransaction:
        """Open a transaction and hand the backend the whole read plan.

        Group 0 goes out at admission priority and the rest at read-ahead, all
        at once. Submitting the rest only after group 0 lands looks like
        prudence and is not: nothing but the scheduler thread can notice that
        group 0 landed, and it only looks once per iteration, so the disks sat
        idle for a whole scheduler pass between the two submissions -- measured
        at ~132 ms of a 171 ms read, which is also why that read reported 7.3
        GiB/s when the device does 35. The arbiter keeps group 0 ahead of the
        rest by priority and reserves submission slots for it, so ordering does
        not need a barrier to enforce it.
        """
        handle = self._backend.begin_read(
            transaction_id=transaction_id,
            generation=generation,
            plan=plan,
            target=target,
        )
        transaction = LayerwiseTransaction(
            transaction_id=transaction_id,
            generation=generation,
            plan=plan,
            handle=handle,
            host_ownership=PrivateBufferOwnership(resource_id=host_resource_id),
        )
        transaction.machine.advance(TransactionState.QUERYING_L3)
        transaction.machine.advance(TransactionState.HOST_PRIVATE_ALLOCATED)
        transaction.machine.advance(TransactionState.READING_GROUP0)
        self._submit_group(transaction, group_id=0, priority=_ADMISSION_PRIORITY)
        self._submit_read_ahead(transaction)
        return transaction

    def _submit_read_ahead(self, transaction: LayerwiseTransaction) -> None:
        while transaction.next_submit_group_id < transaction.group_count:
            self._submit_group(
                transaction,
                group_id=transaction.next_submit_group_id,
                priority=_READ_AHEAD_PRIORITY,
            )
        transaction.mark("rest_submitted")

    def advance(self, transaction: LayerwiseTransaction) -> None:
        """One non-blocking step: drain completions, agree, hand off to H2D.

        Submission is not here: the whole plan went to the arbiter in
        ``begin``, so a slow visit from the scheduler thread delays the
        admission decision but no longer stalls the reader.
        """
        if transaction.machine.state in (
            TransactionState.ABORTED,
            TransactionState.DONE,
        ):
            return
        transaction.marks["advances"] = transaction.marks.get("advances", 0) + 1
        self._drain_completions(transaction)
        if transaction.aborted:
            return
        self._enforce_deadlines(transaction)
        if transaction.aborted:
            return
        self._advance_consensus(transaction)
        self._advance_h2d(transaction)
        self._advance_transaction_state(transaction)

    def note_device_allocated(
        self, transaction: LayerwiseTransaction, *, device_resource_id: str
    ) -> None:
        transaction.device_ownership = PrivateBufferOwnership(
            resource_id=device_resource_id
        )
        transaction.machine.advance(TransactionState.DEVICE_PRIVATE_ALLOCATED)
        transaction.machine.advance(TransactionState.STREAMING)

    def note_forward_complete(self, transaction: LayerwiseTransaction) -> None:
        transaction.machine.advance(TransactionState.FORWARD_COMPLETE)

    def abort(self, transaction: LayerwiseTransaction, *, reason: str) -> None:
        """Stop submitting, ask for cancellation, and record why."""
        if transaction.aborted:
            return
        transaction.error = reason
        transaction.machine.advance(TransactionState.ABORTING)
        pending = tuple(
            group.plan.group_id
            for group in transaction.machine.groups
            if group.state is GroupState.SUBMITTED
        )
        planned = tuple(
            group
            for group in transaction.machine.groups
            if group.state is GroupState.PLANNED
        )
        for group in planned:
            group.cancel_unsubmitted()
        if pending:
            self._backend.request_cancel(handle=transaction.handle, group_ids=pending)
        logger.warning(
            "Layerwise storage transaction %s aborting: %s",
            transaction.transaction_id,
            reason,
        )

    def is_release_safe(self, transaction: LayerwiseTransaction) -> bool:
        """True once no backend operation can still write to the staging."""
        self._drain_completions(transaction)
        return (
            self._backend.poll_terminal(handle=transaction.handle)
            is not HandleTerminalStatus.ACTIVE
        )

    def resolve_ownership(
        self,
        transaction: LayerwiseTransaction,
        *,
        host_state: OwnershipState,
        device_state: Optional[OwnershipState] = None,
    ) -> None:
        """Resolve each private allocation exactly once, then close the handle."""
        operation_terminal = self.is_release_safe(transaction)
        transaction.host_ownership.resolve(
            target=host_state, operation_terminal=operation_terminal
        )
        if device_state is not None and transaction.device_ownership is not None:
            # Device slots are never touched by storage, so their disposition
            # does not wait on the storage handle.
            transaction.device_ownership.resolve(
                target=device_state, operation_terminal=True
            )
        for group in transaction.machine.groups:
            self._consensus.release(
                transaction_id=transaction.transaction_id,
                group_id=group.plan.group_id,
            )
        if operation_terminal:
            self._backend.close(handle=transaction.handle)

    def commit(self, transaction: LayerwiseTransaction) -> None:
        transaction.machine.advance(TransactionState.L2_COMMITTED)
        transaction.machine.advance(TransactionState.DONE)

    def _submit_group(
        self, transaction: LayerwiseTransaction, *, group_id: int, priority: int
    ) -> None:
        group_machine = transaction.machine.group(group_id)
        group_machine.submit()
        if group_id == 0:
            transaction.mark("g0_submitted")
        self._backend.submit_group(
            handle=transaction.handle,
            group=transaction.plan.groups[group_id],
            priority=priority,
            deadline_s=transaction.group_deadlines.get(group_id),
        )
        transaction.next_submit_group_id = max(
            transaction.next_submit_group_id, group_id + 1
        )

    def _drain_completions(self, transaction: LayerwiseTransaction) -> None:
        for completion in self._backend.poll(handle=transaction.handle):
            if completion.generation != transaction.generation:
                continue
            transaction.machine.apply_completion(completion)
        failed = [
            group
            for group in transaction.machine.groups
            if group.state in (GroupState.FAILED, GroupState.CANCELLED)
        ]
        if failed and not transaction.aborted:
            self.abort(
                transaction,
                reason=f"group {failed[0].plan.group_id} did not complete locally",
            )

    def _enforce_deadlines(self, transaction: LayerwiseTransaction) -> None:
        """Watchdog the group at the retirement frontier, and only that one.

        With continuous read-ahead every remaining group is submitted at once,
        so a submit-time clock would fire on groups that are merely queued
        behind a healthy frontier rather than stuck. The frontier group is the
        one nothing can hide, and every group becomes the frontier in turn, so
        arming it there still catches a stalled read -- just not before the
        reader actually needs that group.
        """
        frontier_id = transaction.machine.next_retire_group_id
        if frontier_id >= transaction.group_count:
            return
        group = transaction.machine.group(frontier_id)
        if group.state is not GroupState.SUBMITTED:
            return
        now = time.monotonic()
        deadline = transaction.group_deadlines.get(frontier_id)
        if deadline is None:
            deadline = now + self._config.group_timeout_s
            transaction.group_deadlines[frontier_id] = deadline
        if now > deadline:
            self.abort(
                transaction,
                reason=f"group {frontier_id} exceeded its storage timeout",
            )

    def _advance_consensus(self, transaction: LayerwiseTransaction) -> None:
        """Start agreement in group order, then collect whatever is decided."""
        while transaction.next_consensus_group_id < transaction.group_count:
            group = transaction.machine.group(transaction.next_consensus_group_id)
            if group.state is not GroupState.LOCAL_DONE:
                break
            if group.plan.group_id == 0:
                transaction.mark("g0_local_done")
            group.begin_consensus()
            self._consensus.begin(
                transaction_id=transaction.transaction_id,
                group_id=group.plan.group_id,
                local_success=True,
            )
            transaction.next_consensus_group_id += 1

        for group in transaction.machine.groups:
            if group.state is not GroupState.CONSENSUS_PENDING:
                continue
            verdict = self._consensus.poll(
                transaction_id=transaction.transaction_id,
                group_id=group.plan.group_id,
            )
            if verdict is None:
                break
            if not verdict:
                self.abort(
                    transaction,
                    reason=f"group {group.plan.group_id} failed on a peer rank",
                )
                return
            group.mark_global_ready()
            if group.plan.group_id == 0:
                transaction.mark("g0_global_ready")

    def _advance_h2d(self, transaction: LayerwiseTransaction) -> None:
        """Hand agreed groups to the H2D stream in strict plan order."""
        while transaction.next_h2d_group_id < transaction.group_count:
            group = transaction.machine.group(transaction.next_h2d_group_id)
            if group.state is not GroupState.GLOBAL_READY:
                break
            if transaction.machine.state is not TransactionState.STREAMING:
                break
            plan = transaction.plan.groups[group.plan.group_id]
            self._submit_h2d_range(transaction, plan.layer_start, plan.layer_end)
            group.mark_h2d_submitted()
            group.mark_device_ready()
            transaction.next_h2d_group_id += 1
        transaction.machine.retire_ready_groups()

    def _advance_transaction_state(self, transaction: LayerwiseTransaction) -> None:
        machine = transaction.machine
        if machine.state is TransactionState.READING_GROUP0:
            if self._group0_ready(transaction):
                machine.advance(TransactionState.ADMISSION_READY)
                transaction.admitted_at = time.monotonic()
                transaction.mark("admission_ready")
                self._report_admission(transaction)
            return
        if machine.state is TransactionState.STREAMING and machine.all_groups_retired:
            machine.advance(TransactionState.STORAGE_COMPLETE)

    def _report_admission(self, transaction: LayerwiseTransaction) -> None:
        """One line per transaction, at the moment admission is decided.

        Every mark is an offset from group 0's submission, so "did group 0 gate
        this, or did the whole read" is answered by comparing two numbers
        rather than by reading the state machine. ``advances`` counts the
        scheduler-thread visits it took to get here, which is what separates a
        slow read from a rendezvous that came late.
        """
        from sglang.srt.observability.req_time_stats import convert_time_to_realtime

        base = transaction.marks.get("g0_submitted")
        if base is None:
            return

        def offset(event: str) -> float:
            mark = transaction.marks.get(event)
            return -1.0 if mark is None else (mark - base) * 1e3

        logger.info(
            "layerwise_admit: txn=%s groups=%d advances=%d g0_submit=%.3f "
            "g0_local_done=+%.2f g0_global_ready=+%.2f rest_submitted=+%.2f "
            "admission_ready=+%.2f",
            transaction.transaction_id,
            transaction.group_count,
            transaction.marks.get("advances", 0),
            convert_time_to_realtime(base),
            offset("g0_local_done"),
            offset("g0_global_ready"),
            offset("rest_submitted"),
            offset("admission_ready"),
        )

    def _group0_ready(self, transaction: LayerwiseTransaction) -> bool:
        """Admission waits for agreement, not just for the local read."""
        return transaction.machine.group(0).state in (
            GroupState.GLOBAL_READY,
            GroupState.H2D_SUBMITTED,
            GroupState.DEVICE_READY,
            GroupState.RETIRED,
        )

    def admission_over_budget(self, transaction: LayerwiseTransaction) -> bool:
        """True when group 0 blew the latency budget the SLO was derived from."""
        if self._config.admission_budget_s <= 0.0:
            return False
        if transaction.admission_ready:
            return False
        return time.monotonic() - transaction.started_at > (
            self._config.admission_budget_s
        )


def _reject_unusable_backend(capabilities: LayerwiseBackendCapabilities) -> None:
    if not capabilities.supports_range_read:
        raise ValueError(
            "layerwise streaming needs a backend that can read a layer range; "
            "this one can only return whole pages"
        )
    if capabilities.cancel_level is CancelLevel.NEITHER:
        raise ValueError(
            "layerwise streaming needs a backend whose reads reach a terminal "
            "state; without it a failed transaction could never free its staging"
        )
