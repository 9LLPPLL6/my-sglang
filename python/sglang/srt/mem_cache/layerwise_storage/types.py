from __future__ import annotations

import enum
from typing import Any

import msgspec


class CancelLevel(str, enum.Enum):
    CANCELLABLE = "cancellable"
    BOUNDED_TERMINAL = "bounded_terminal"
    NEITHER = "neither"


class KVPart(str, enum.Enum):
    KEY = "key"
    VALUE = "value"


class ExtentCompletionStatus(str, enum.Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CancelRequestDisposition(str, enum.Enum):
    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"
    ALREADY_TERMINAL = "already_terminal"


class HandleTerminalStatus(str, enum.Enum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self is not HandleTerminalStatus.ACTIVE


class LayerwiseBackendCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """Limits and guarantees exposed by a layerwise storage backend."""

    required_alignment: int
    supports_range_read: bool
    supports_direct_to_host: bool
    max_inflight_groups: int
    max_inflight_extents: int
    max_inflight_bytes: int
    max_iov: int
    cancel_level: CancelLevel

    def __post_init__(self) -> None:
        _require_positive("required_alignment", self.required_alignment)
        _require_positive("max_inflight_groups", self.max_inflight_groups)
        _require_positive("max_inflight_extents", self.max_inflight_extents)
        _require_positive("max_inflight_bytes", self.max_inflight_bytes)
        _require_positive("max_iov", self.max_iov)


class HostTargetBase(msgspec.Struct, frozen=True, kw_only=True):
    """Base host addresses that an extent's ``target_offset`` is relative to.

    The host staging for one transaction is not contiguous: its pages are
    scattered across the host KV pool.  Anchoring on the two pool buffers keeps
    every extent target expressible as one base plus an offset, which is what a
    backend needs to build an I/O descriptor without understanding the pool.
    """

    k_ptr: int
    v_ptr: int

    def base_for(self, kv_part: KVPart) -> int:
        return self.k_ptr if kv_part is KVPart.KEY else self.v_ptr

    def __post_init__(self) -> None:
        _require_positive("k_ptr", self.k_ptr)
        _require_positive("v_ptr", self.v_ptr)


class LayerwiseStorageExtent(msgspec.Struct, frozen=True, kw_only=True):
    """One submitted storage range and its valid payload within that range.

    ``io_offset`` and ``io_nbytes`` describe the aligned backend operation.
    ``payload_offset`` selects valid bytes from that operation, which permits
    an aligned covering read to feed an unaligned compact on-disk payload.
    ``target_offset`` is relative to the transaction-owned staging target.
    """

    extent_id: int
    storage_key: str
    kv_part: KVPart
    layer_start: int
    layer_end: int
    io_offset: int
    io_nbytes: int
    payload_offset: int
    payload_nbytes: int
    target_offset: int

    def __post_init__(self) -> None:
        _require_non_negative("extent_id", self.extent_id)
        if not self.storage_key:
            raise ValueError("storage_key must not be empty")
        _validate_layer_range(
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )
        _require_non_negative("io_offset", self.io_offset)
        _require_positive("io_nbytes", self.io_nbytes)
        _require_non_negative("payload_offset", self.payload_offset)
        _require_positive("payload_nbytes", self.payload_nbytes)
        _require_non_negative("target_offset", self.target_offset)
        if self.payload_offset + self.payload_nbytes > self.io_nbytes:
            raise ValueError("payload range must be contained in the I/O range")


class LayerGroupPlan(msgspec.Struct, frozen=True, kw_only=True):
    """The storage extents needed to make one contiguous layer group ready."""

    group_id: int
    layer_start: int
    layer_end: int
    extents: tuple[LayerwiseStorageExtent, ...]

    def __post_init__(self) -> None:
        _require_non_negative("group_id", self.group_id)
        _validate_layer_range(
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )
        if not self.extents:
            raise ValueError("a layer group must contain at least one extent")

        extent_ids = {extent.extent_id for extent in self.extents}
        if len(extent_ids) != len(self.extents):
            raise ValueError("extent_id values must be unique within a group")
        if any(
            extent.layer_start < self.layer_start or extent.layer_end > self.layer_end
            for extent in self.extents
        ):
            raise ValueError("every extent layer range must be inside its group")

    @property
    def total_io_nbytes(self) -> int:
        return sum(extent.io_nbytes for extent in self.extents)


class LayerwiseReadPlan(msgspec.Struct, frozen=True, kw_only=True):
    """An ordered, gap-free sequence of layer groups for one storage hit."""

    groups: tuple[LayerGroupPlan, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("a layerwise read plan must contain at least one group")

        expected_group_ids = tuple(range(len(self.groups)))
        group_ids = tuple(group.group_id for group in self.groups)
        if group_ids != expected_group_ids:
            raise ValueError("group_id values must be contiguous and ordered from zero")

        for previous, current in zip(self.groups, self.groups[1:]):
            if previous.layer_end != current.layer_start:
                raise ValueError("layer group ranges must be contiguous")

    @property
    def total_io_nbytes(self) -> int:
        return sum(group.total_io_nbytes for group in self.groups)


class LayerwiseReadHandle(msgspec.Struct, frozen=True, kw_only=True):
    """Stable read identity plus an opaque backend-owned token."""

    transaction_id: str
    generation: int
    backend_token: Any = None

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id must not be empty")
        _require_non_negative("generation", self.generation)


class LayerwiseGroupTicket(msgspec.Struct, frozen=True, kw_only=True):
    """Backend submission identity for one group of a read handle."""

    handle: LayerwiseReadHandle
    group_id: int
    backend_token: Any = None

    def __post_init__(self) -> None:
        _require_non_negative("group_id", self.group_id)


class LayerwiseStorageCompletion(msgspec.Struct, frozen=True, kw_only=True):
    """A terminal completion for one extent.

    There is deliberately no pending or cancel-requested status. A backend
    emits this record only after it will no longer access the extent target.
    """

    transaction_id: str
    generation: int
    group_id: int
    extent_id: int
    status: ExtentCompletionStatus
    bytes_transferred: int
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id must not be empty")
        _require_non_negative("generation", self.generation)
        _require_non_negative("group_id", self.group_id)
        _require_non_negative("extent_id", self.extent_id)
        _require_non_negative("bytes_transferred", self.bytes_transferred)
        if self.status is ExtentCompletionStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a successful completion must not carry an error")


class CancelRequestResult(msgspec.Struct, frozen=True, kw_only=True):
    """Backend acknowledgement of a cancel request, not I/O termination."""

    group_id: int
    disposition: CancelRequestDisposition

    def __post_init__(self) -> None:
        _require_non_negative("group_id", self.group_id)


def validate_group_against_capabilities(
    *,
    group: LayerGroupPlan,
    capabilities: LayerwiseBackendCapabilities,
) -> None:
    """Reject a group that cannot fit one backend submission window."""

    if len(group.extents) > capabilities.max_inflight_extents:
        raise ValueError(
            f"group {group.group_id} has {len(group.extents)} extents, exceeding "
            f"the backend limit {capabilities.max_inflight_extents}"
        )
    if len(group.extents) > capabilities.max_iov:
        raise ValueError(
            f"group {group.group_id} has {len(group.extents)} extents, exceeding "
            f"the backend iov limit {capabilities.max_iov}"
        )
    if group.total_io_nbytes > capabilities.max_inflight_bytes:
        raise ValueError(
            f"group {group.group_id} has {group.total_io_nbytes} I/O bytes, "
            f"exceeding the backend limit {capabilities.max_inflight_bytes}"
        )

    alignment = capabilities.required_alignment
    for extent in group.extents:
        if extent.io_offset % alignment or extent.io_nbytes % alignment:
            raise ValueError(
                f"extent {extent.extent_id} is not aligned to {alignment} bytes"
            )


def _validate_layer_range(*, layer_start: int, layer_end: int) -> None:
    _require_non_negative("layer_start", layer_start)
    if layer_end <= layer_start:
        raise ValueError("layer_end must be greater than layer_start")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
