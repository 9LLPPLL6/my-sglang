from __future__ import annotations

import enum

from sglang.srt.mem_cache.layerwise_storage.types import (
    ExtentCompletionStatus,
    LayerGroupPlan,
    LayerwiseReadPlan,
    LayerwiseStorageCompletion,
)


class InvalidStateTransition(RuntimeError):
    pass


class InvalidStorageCompletion(RuntimeError):
    pass


class OwnershipResolutionError(RuntimeError):
    pass


class TransactionState(str, enum.Enum):
    NEW = "new"
    QUERYING_L3 = "querying_l3"
    HOST_PRIVATE_ALLOCATED = "host_private_allocated"
    READING_GROUP0 = "reading_group0"
    ADMISSION_READY = "admission_ready"
    DEVICE_PRIVATE_ALLOCATED = "device_private_allocated"
    STREAMING = "streaming"
    STORAGE_COMPLETE = "storage_complete"
    FORWARD_COMPLETE = "forward_complete"
    L2_COMMITTED = "l2_committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    DONE = "done"


class GroupState(str, enum.Enum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    LOCAL_DONE = "local_done"
    CONSENSUS_PENDING = "consensus_pending"
    GLOBAL_READY = "global_ready"
    H2D_SUBMITTED = "h2d_submitted"
    DEVICE_READY = "device_ready"
    RETIRED = "retired"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OwnershipState(str, enum.Enum):
    PRIVATE = "private"
    INSERTED = "inserted"
    FREED = "freed"
    QUARANTINED = "quarantined"

    @property
    def is_terminal(self) -> bool:
        return self is not OwnershipState.PRIVATE


_TRANSACTION_TRANSITIONS = {
    TransactionState.NEW: frozenset(
        {TransactionState.QUERYING_L3, TransactionState.ABORTING}
    ),
    TransactionState.QUERYING_L3: frozenset(
        {TransactionState.HOST_PRIVATE_ALLOCATED, TransactionState.ABORTING}
    ),
    TransactionState.HOST_PRIVATE_ALLOCATED: frozenset(
        {TransactionState.READING_GROUP0, TransactionState.ABORTING}
    ),
    TransactionState.READING_GROUP0: frozenset(
        {TransactionState.ADMISSION_READY, TransactionState.ABORTING}
    ),
    TransactionState.ADMISSION_READY: frozenset(
        {TransactionState.DEVICE_PRIVATE_ALLOCATED, TransactionState.ABORTING}
    ),
    TransactionState.DEVICE_PRIVATE_ALLOCATED: frozenset(
        {TransactionState.STREAMING, TransactionState.ABORTING}
    ),
    TransactionState.STREAMING: frozenset(
        {TransactionState.STORAGE_COMPLETE, TransactionState.ABORTING}
    ),
    TransactionState.STORAGE_COMPLETE: frozenset(
        {TransactionState.FORWARD_COMPLETE, TransactionState.ABORTING}
    ),
    TransactionState.FORWARD_COMPLETE: frozenset(
        {TransactionState.L2_COMMITTED, TransactionState.ABORTING}
    ),
    TransactionState.L2_COMMITTED: frozenset({TransactionState.DONE}),
    TransactionState.ABORTING: frozenset({TransactionState.ABORTED}),
    TransactionState.ABORTED: frozenset(),
    TransactionState.DONE: frozenset(),
}

_GROUP_TRANSITIONS = {
    GroupState.PLANNED: frozenset(
        {GroupState.SUBMITTED, GroupState.FAILED, GroupState.CANCELLED}
    ),
    GroupState.SUBMITTED: frozenset(
        {GroupState.LOCAL_DONE, GroupState.FAILED, GroupState.CANCELLED}
    ),
    GroupState.LOCAL_DONE: frozenset({GroupState.CONSENSUS_PENDING, GroupState.FAILED}),
    GroupState.CONSENSUS_PENDING: frozenset(
        {GroupState.GLOBAL_READY, GroupState.FAILED}
    ),
    GroupState.GLOBAL_READY: frozenset({GroupState.H2D_SUBMITTED, GroupState.FAILED}),
    GroupState.H2D_SUBMITTED: frozenset({GroupState.DEVICE_READY, GroupState.FAILED}),
    GroupState.DEVICE_READY: frozenset({GroupState.RETIRED, GroupState.FAILED}),
    GroupState.RETIRED: frozenset(),
    GroupState.FAILED: frozenset(),
    GroupState.CANCELLED: frozenset(),
}


class PrivateBufferOwnership:
    """Tracks the exactly-once disposition of one private allocation."""

    def __init__(self, *, resource_id: str):
        if not resource_id:
            raise ValueError("resource_id must not be empty")
        self.resource_id = resource_id
        self.state = OwnershipState.PRIVATE

    def resolve(
        self,
        *,
        target: OwnershipState,
        operation_terminal: bool,
    ) -> None:
        if self.state.is_terminal:
            raise OwnershipResolutionError(
                f"{self.resource_id} ownership was already resolved as "
                f"{self.state.value}"
            )
        if not target.is_terminal:
            raise OwnershipResolutionError("ownership must resolve to a terminal state")
        if target is not OwnershipState.QUARANTINED and not operation_terminal:
            raise OwnershipResolutionError(
                f"cannot resolve {self.resource_id} as {target.value} before the "
                "operation is terminal"
            )
        self.state = target


class LayerGroupStateMachine:
    """Pure lifecycle and completion accounting for one layer group."""

    def __init__(
        self,
        *,
        transaction_id: str,
        generation: int,
        plan: LayerGroupPlan,
    ):
        if not transaction_id:
            raise ValueError("transaction_id must not be empty")
        if generation < 0:
            raise ValueError("generation must be non-negative")
        self.transaction_id = transaction_id
        self.generation = generation
        self.plan = plan
        self.state = GroupState.PLANNED
        self.cancel_requested = False
        self._storage_terminal = False
        self._completions: dict[int, LayerwiseStorageCompletion] = {}
        self._extents_by_id = {extent.extent_id: extent for extent in self.plan.extents}

    @property
    def storage_terminal(self) -> bool:
        return self._storage_terminal

    @property
    def completion_count(self) -> int:
        return len(self._completions)

    def submit(self) -> None:
        self._transition(GroupState.SUBMITTED)

    def request_cancel(self) -> bool:
        """Records intent only; completion still controls storage termination."""

        if self.state is not GroupState.SUBMITTED or self._storage_terminal:
            return False
        if self.cancel_requested:
            return False
        self.cancel_requested = True
        return True

    def cancel_unsubmitted(self) -> None:
        if self.state is not GroupState.PLANNED:
            raise InvalidStateTransition(
                "only a planned group can be cancelled without terminal completions"
            )
        self._storage_terminal = True
        self._transition(GroupState.CANCELLED)

    def apply_completion(self, completion: LayerwiseStorageCompletion) -> None:
        if self.state is not GroupState.SUBMITTED:
            raise InvalidStorageCompletion(
                f"group {self.plan.group_id} cannot accept a completion in "
                f"state {self.state.value}"
            )
        self._validate_completion(completion)
        self._completions[completion.extent_id] = completion
        if len(self._completions) != len(self.plan.extents):
            return

        self._storage_terminal = True
        statuses = {result.status for result in self._completions.values()}
        if ExtentCompletionStatus.FAILED in statuses:
            self._transition(GroupState.FAILED)
        elif ExtentCompletionStatus.CANCELLED in statuses:
            self._transition(GroupState.CANCELLED)
        else:
            self._transition(GroupState.LOCAL_DONE)

    def begin_consensus(self) -> None:
        self._transition(GroupState.CONSENSUS_PENDING)

    def mark_global_ready(self) -> None:
        self._transition(GroupState.GLOBAL_READY)

    def mark_h2d_submitted(self) -> None:
        self._transition(GroupState.H2D_SUBMITTED)

    def mark_device_ready(self) -> None:
        self._transition(GroupState.DEVICE_READY)

    def mark_failed(self) -> None:
        if self.state is GroupState.SUBMITTED and not self._storage_terminal:
            raise InvalidStateTransition(
                "an in-flight group cannot become terminal before every extent does"
            )
        self._transition(GroupState.FAILED)

    def _retire(self) -> None:
        self._transition(GroupState.RETIRED)

    def _validate_completion(self, completion: LayerwiseStorageCompletion) -> None:
        if completion.transaction_id != self.transaction_id:
            raise InvalidStorageCompletion(
                f"completion transaction {completion.transaction_id!r} does not "
                f"match {self.transaction_id!r}"
            )
        if completion.generation != self.generation:
            raise InvalidStorageCompletion(
                f"completion generation {completion.generation} does not match "
                f"{self.generation}"
            )
        if completion.group_id != self.plan.group_id:
            raise InvalidStorageCompletion(
                f"completion group {completion.group_id} does not match "
                f"{self.plan.group_id}"
            )
        if completion.extent_id in self._completions:
            raise InvalidStorageCompletion(
                f"extent {completion.extent_id} completed more than once"
            )

        extent = self._extents_by_id.get(completion.extent_id)
        if extent is None:
            raise InvalidStorageCompletion(
                f"extent {completion.extent_id} is not part of group "
                f"{self.plan.group_id}"
            )
        if completion.bytes_transferred > extent.io_nbytes:
            raise InvalidStorageCompletion(
                f"extent {completion.extent_id} transferred more bytes than submitted"
            )
        if (
            completion.status is ExtentCompletionStatus.SUCCEEDED
            and completion.bytes_transferred != extent.io_nbytes
        ):
            raise InvalidStorageCompletion(
                f"successful extent {completion.extent_id} was a short read"
            )

    def _transition(self, target: GroupState) -> None:
        if target not in _GROUP_TRANSITIONS[self.state]:
            raise InvalidStateTransition(
                f"invalid group transition {self.state.value} -> {target.value}"
            )
        self.state = target


class LayerwiseTransactionStateMachine:
    """Transaction phase plus ordered group retirement for one L3 hit."""

    def __init__(
        self,
        *,
        transaction_id: str,
        generation: int,
        plan: LayerwiseReadPlan,
    ):
        if not transaction_id:
            raise ValueError("transaction_id must not be empty")
        if generation < 0:
            raise ValueError("generation must be non-negative")
        self.transaction_id = transaction_id
        self.generation = generation
        self.plan = plan
        self.state = TransactionState.NEW
        self._groups = tuple(
            LayerGroupStateMachine(
                transaction_id=transaction_id,
                generation=generation,
                plan=group,
            )
            for group in plan.groups
        )
        self._next_retire_group_id = 0

    @property
    def groups(self) -> tuple[LayerGroupStateMachine, ...]:
        return self._groups

    @property
    def next_retire_group_id(self) -> int:
        return self._next_retire_group_id

    @property
    def all_groups_retired(self) -> bool:
        return self._next_retire_group_id == len(self._groups)

    def group(self, group_id: int) -> LayerGroupStateMachine:
        if group_id < 0 or group_id >= len(self._groups):
            raise IndexError(f"unknown group_id {group_id}")
        return self._groups[group_id]

    def advance(self, target: TransactionState) -> None:
        if target not in _TRANSACTION_TRANSITIONS[self.state]:
            raise InvalidStateTransition(
                f"invalid transaction transition {self.state.value} -> {target.value}"
            )
        self.state = target

    def apply_completion(self, completion: LayerwiseStorageCompletion) -> None:
        if completion.transaction_id != self.transaction_id:
            raise InvalidStorageCompletion(
                f"completion transaction {completion.transaction_id!r} does not "
                f"match {self.transaction_id!r}"
            )
        if completion.generation != self.generation:
            raise InvalidStorageCompletion(
                f"completion generation {completion.generation} does not match "
                f"{self.generation}"
            )
        if completion.group_id < 0 or completion.group_id >= len(self._groups):
            raise InvalidStorageCompletion(
                f"completion references unknown group {completion.group_id}"
            )
        self.group(completion.group_id).apply_completion(completion)

    def retire_ready_groups(self) -> tuple[int, ...]:
        """Retires only the contiguous DEVICE_READY prefix of group order."""

        retired = []
        while self._next_retire_group_id < len(self._groups):
            group = self._groups[self._next_retire_group_id]
            if group.state is not GroupState.DEVICE_READY:
                break
            group._retire()
            retired.append(group.plan.group_id)
            self._next_retire_group_id += 1
        return tuple(retired)
