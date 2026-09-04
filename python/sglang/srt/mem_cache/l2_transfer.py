from __future__ import annotations

import logging
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@cache
def _timing_events_supported() -> bool:
    try:
        device_module.Event(enable_timing=True)
        return True
    except (TypeError, NotImplementedError):
        logger.warning(
            "%s.Event does not support timing; L2 transfer timing is disabled",
            device_module.__name__,
        )
        return False


def make_timing_event_pair():
    timing_enabled = _timing_events_supported()
    kwargs = {"enable_timing": True} if timing_enabled else {}
    return device_module.Event(**kwargs), device_module.Event(**kwargs), timing_enabled


class L2Transfer(NamedTuple):
    host_pool: Any
    device_pool: Any
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    layer_mapper: Optional[Callable[[int], Optional[int]]] = None
    is_draft: bool = False


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool


class StreamingL2Transfer:
    """State for one ordered, group-at-a-time host-to-device transfer.

    The session deliberately does not own cache or allocator state.  It keeps
    the transfer descriptors alive until the final CUDA event is recorded and
    enforces monotonically increasing layer ranges so a storage completion
    dispatcher cannot accidentally expose a later layer before an earlier one.
    """

    def __init__(
        self,
        *,
        transfers: list[L2Transfer],
        layer_num: int,
        start_event: Any,
        finish_event: Any,
        timing_enabled: bool,
    ) -> None:
        self.transfers = tuple(transfers)
        self.layer_num = layer_num
        self.next_layer = 0
        self.start_event = start_event
        self.finish_event = finish_event
        self.timing_enabled = timing_enabled
        self.closed = False


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
        start_event = self._start_event(None)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.device_to_host_stream):
            start_event.wait(self.device_to_host_stream)
            ack_start.record()
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
            ack_finish.record()
            self._record_stream(transfers, self.device_to_host_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        layer_num: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        session = self.begin_host_to_device_streaming(
            transfers,
            layer_num=layer_num,
            start_event=start_event,
        )
        if layer_num > 0:
            self.submit_host_to_device_range(
                session,
                layer_start=0,
                layer_end=layer_num,
                on_layer_done=on_layer_done,
            )
        return self.finish_host_to_device_streaming(session)

    def begin_host_to_device_streaming(
        self,
        transfers: list[L2Transfer],
        *,
        layer_num: int,
        start_event=None,
    ) -> StreamingL2Transfer:
        """Start an ordered H2D session whose ranges can be submitted later."""
        if layer_num < 0:
            raise ValueError(f"layer_num must be non-negative, got {layer_num}")

        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.host_to_device_stream):
            start_event.wait(self.host_to_device_stream)
            ack_start.record()
        return StreamingL2Transfer(
            transfers=transfers,
            layer_num=layer_num,
            start_event=ack_start,
            finish_event=ack_finish,
            timing_enabled=timing_enabled,
        )

    def submit_host_to_device_range(
        self,
        session: StreamingL2Transfer,
        *,
        layer_start: int,
        layer_end: int,
        on_layer_done=None,
    ) -> None:
        """Submit one contiguous global-layer range to an active session."""
        self._validate_streaming_range(
            session=session,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        primary = session.transfers[0] if session.transfers else None
        with device_module.stream(self.host_to_device_stream):
            for layer_id in range(layer_start, layer_end):
                self._submit_host_to_device_layer(
                    transfers=session.transfers,
                    primary=primary,
                    layer_id=layer_id,
                )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
        session.next_layer = layer_end

    def finish_host_to_device_streaming(
        self, session: StreamingL2Transfer
    ) -> TransferCompletion:
        """Close a fully submitted session and return its aggregate completion."""
        if session.closed:
            raise RuntimeError("streaming H2D session is already closed")
        if session.next_layer != session.layer_num:
            raise RuntimeError(
                "cannot finish streaming H2D session before every layer is submitted: "
                f"next_layer={session.next_layer}, layer_num={session.layer_num}"
            )

        with device_module.stream(self.host_to_device_stream):
            session.finish_event.record()
            self._record_stream(session.transfers, self.host_to_device_stream)
        session.closed = True
        return TransferCompletion(
            session.start_event,
            session.finish_event,
            session.timing_enabled,
        )

    def _submit_host_to_device_layer(
        self,
        *,
        transfers: tuple[L2Transfer, ...],
        primary: Optional[L2Transfer],
        layer_id: int,
    ) -> None:
        for transfer in transfers:
            local_layer_id = (
                transfer.layer_mapper(layer_id)
                if transfer.layer_mapper is not None
                else layer_id
            )
            if local_layer_id is None or (
                transfer is not primary
                and transfer.layer_mapper is None
                and layer_id >= transfer.host_pool.layer_num
            ):
                continue
            transfer.host_pool.load_to_device_per_layer(
                transfer.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                self.io_backend,
                is_draft=transfer.is_draft,
            )

    @staticmethod
    def _validate_streaming_range(
        *,
        session: StreamingL2Transfer,
        layer_start: int,
        layer_end: int,
    ) -> None:
        if session.closed:
            raise RuntimeError("streaming H2D session is already closed")
        if layer_start != session.next_layer:
            raise ValueError(
                "streaming H2D ranges must be contiguous and ordered: "
                f"expected layer_start={session.next_layer}, got {layer_start}"
            )
        if not layer_start < layer_end <= session.layer_num:
            raise ValueError(
                "invalid streaming H2D layer range: "
                f"[{layer_start}, {layer_end}) for layer_num={session.layer_num}"
            )

    @staticmethod
    def _start_event(start_event):
        if start_event is None:
            start_event = device_module.Event()
            start_event.record()
        return start_event

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream) -> None:
        for transfer in transfers:
            for indices in (transfer.host_indices, transfer.device_indices):
                if indices.is_cuda:
                    indices.record_stream(stream)
