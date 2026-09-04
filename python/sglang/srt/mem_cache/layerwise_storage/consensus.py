"""Per-group cross-rank agreement for layerwise storage reads.

Every TP rank holds a different KV shard of the same page, so a group is usable
only when it succeeded on all of them.  Agreement must also be non-blocking:
the scheduler thread drives the pipeline, and a synchronous collective there
would serialize the very overlap the pipeline exists to create.

Groups are agreed in plan order.  Local completions may arrive out of order,
but a collective started out of order would deadlock the moment two ranks
disagree about which group comes next.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


class GroupConsensus(ABC):
    """Asynchronous agreement on whether one layer group succeeded everywhere."""

    @abstractmethod
    def begin(self, *, transaction_id: str, group_id: int, local_success: bool) -> None:
        pass

    @abstractmethod
    def poll(self, *, transaction_id: str, group_id: int) -> Optional[bool]:
        """``None`` while undecided, otherwise the agreed verdict."""

    @abstractmethod
    def release(self, *, transaction_id: str, group_id: int) -> None:
        pass


class SingleRankConsensus(GroupConsensus):
    """Agreement for TP=1 and for tests: the local verdict is the verdict."""

    def __init__(self):
        self._verdicts: dict[tuple[str, int], bool] = {}

    def begin(self, *, transaction_id: str, group_id: int, local_success: bool) -> None:
        self._verdicts[(transaction_id, group_id)] = local_success

    def poll(self, *, transaction_id: str, group_id: int) -> Optional[bool]:
        return self._verdicts.get((transaction_id, group_id))

    def release(self, *, transaction_id: str, group_id: int) -> None:
        self._verdicts.pop((transaction_id, group_id), None)


class TorchDistGroupConsensus(GroupConsensus):
    """Async ``all_reduce(MIN)`` over a one-element success flag per group.

    The work handle is polled rather than waited on, so a rank whose storage is
    slow delays only this transaction's next group, not the scheduler loop.
    """

    def __init__(self, *, group: torch.distributed.ProcessGroup, device: str):
        import torch

        self._torch = torch
        self._group = group
        self._device = device
        self._pending: dict[tuple[str, int], tuple] = {}

    def begin(self, *, transaction_id: str, group_id: int, local_success: bool) -> None:
        key = (transaction_id, group_id)
        if key in self._pending:
            raise RuntimeError(f"consensus for {key} was already started")
        flag = self._torch.tensor(
            [1 if local_success else 0], dtype=self._torch.int32, device=self._device
        )
        work = self._torch.distributed.all_reduce(
            flag,
            op=self._torch.distributed.ReduceOp.MIN,
            group=self._group,
            async_op=True,
        )
        self._pending[key] = (work, flag)

    def poll(self, *, transaction_id: str, group_id: int) -> Optional[bool]:
        entry = self._pending.get((transaction_id, group_id))
        if entry is None:
            return None
        work, flag = entry
        if not work.is_completed():
            return None
        return bool(flag.item())

    def release(self, *, transaction_id: str, group_id: int) -> None:
        entry = self._pending.pop((transaction_id, group_id), None)
        if entry is None:
            return
        work, _ = entry
        if not work.is_completed():
            # A collective must not be abandoned mid-flight: the peer ranks are
            # still in it, and dropping the handle leaks the communicator slot.
            work.wait()
