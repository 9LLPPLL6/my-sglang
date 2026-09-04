from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sglang.srt.mem_cache.layerwise_storage.types import (
    CancelRequestResult,
    HandleTerminalStatus,
    LayerGroupPlan,
    LayerwiseBackendCapabilities,
    LayerwiseGroupTicket,
    LayerwiseReadHandle,
    LayerwiseReadPlan,
    LayerwiseStorageCompletion,
)


class LayerwiseStorageBackend(ABC):
    """Optional asynchronous range-read interface for HiCache backends.

    Existing storage backends do not inherit from this class, so introducing
    the interface changes no default behavior. A layerwise pipeline must probe
    for this interface and its capabilities before using it.
    """

    @abstractmethod
    def capabilities(self) -> LayerwiseBackendCapabilities:
        pass

    @abstractmethod
    def begin_read(
        self,
        *,
        transaction_id: str,
        generation: int,
        plan: LayerwiseReadPlan,
        target: Any,
    ) -> LayerwiseReadHandle:
        pass

    @abstractmethod
    def submit_group(
        self,
        *,
        handle: LayerwiseReadHandle,
        group: LayerGroupPlan,
        priority: int,
        deadline_s: float | None,
    ) -> LayerwiseGroupTicket:
        pass

    @abstractmethod
    def poll(
        self,
        *,
        handle: LayerwiseReadHandle,
        max_completions: int | None = None,
    ) -> tuple[LayerwiseStorageCompletion, ...]:
        pass

    @abstractmethod
    def request_cancel(
        self,
        *,
        handle: LayerwiseReadHandle,
        group_ids: tuple[int, ...],
    ) -> tuple[CancelRequestResult, ...]:
        """Requests cancellation without promising that target access stopped."""
        pass

    @abstractmethod
    def poll_terminal(
        self,
        *,
        handle: LayerwiseReadHandle,
    ) -> HandleTerminalStatus:
        pass

    @abstractmethod
    def close(self, *, handle: LayerwiseReadHandle) -> None:
        """Releases backend metadata after ``poll_terminal`` is terminal."""
        pass
