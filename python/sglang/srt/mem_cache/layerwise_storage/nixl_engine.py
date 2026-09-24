"""A NIXL-backed I/O engine for the layerwise storage read path.

The arbiter needs only six operations from an engine -- ``free_slots``,
``inflight``, ``submit_reads``, ``submit_writes``, ``poll`` and ``close`` --
so NIXL is introduced behind that seam. Sharding, priorities, bounce buffers,
fd caching and the read report all stay with ``LayerwiseFileBackend``; only the
transport differs, which is what makes an A/B against Linux AIO meaningful.

One engine owns one NIXL agent. Agents are not shared across shards, so no
lock is needed here: the backend already guarantees that a shard's context is
touched by one thread.

Unlike Linux AIO, a NIXL transfer is a batch: ``initialize_xfer`` pairs a list
of DRAM descriptors with a list of FILE descriptors element-wise and completes
as a unit. One submitted batch therefore retires all of its extents together,
and a failed batch fails all of them.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

from sglang.srt.mem_cache.layerwise_storage.aio_engine import AioCompletion

logger = logging.getLogger(__name__)

_EIO = 5


class NixlUnavailable(RuntimeError):
    pass


def _import_nixl():
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as error:
        raise NixlUnavailable(
            "the layerwise NIXL engine needs the `nixl` package; "
            "install nixl-cu12 or nixl-cu13 plus the `nixl` meta package"
        ) from error
    return nixl_agent, nixl_agent_config


class _Batch:
    """One submitted NIXL transfer and the extents it retires together."""

    __slots__ = ("xfer", "entries", "dram_reg")

    def __init__(self, *, xfer, entries, dram_reg):
        self.xfer = xfer
        # (user_data, nbytes) per extent, in descriptor order.
        self.entries = entries
        # Registration held only for targets outside the pinned regions.
        self.dram_reg = dram_reg


class NixlIoContext:
    """Moves layer-group extents through NIXL with the AIO engine's surface."""

    def __init__(
        self,
        *,
        queue_depth: int,
        plugin: str = "POSIX",
        agent_name: Optional[str] = None,
        pinned_regions: Sequence[tuple[int, int]] = (),
        backend_params: Optional[dict] = None,
    ):
        if queue_depth <= 0:
            raise ValueError(f"queue_depth must be positive, got {queue_depth}")
        nixl_agent, nixl_agent_config = _import_nixl()

        self.queue_depth = queue_depth
        self.plugin = plugin
        self._name = agent_name or f"sglang-layerwise-{os.getpid()}-{id(self):x}"
        self._agent = nixl_agent(self._name, nixl_agent_config(backends=[]))
        available = self._agent.agent.getAvailPlugins()
        if plugin not in available:
            raise NixlUnavailable(
                f"NIXL plugin {plugin!r} is not available; have {available}"
            )
        self._agent.create_backend(plugin, backend_params or {})

        # Registering the host KV pool once removes it from the submit path.
        # Everything else (bounce buffers) is registered per batch, which a
        # measured 2 us per call makes cheaper than tracking allocations.
        self._pinned_regs = []
        ranges = []
        for ptr, size in pinned_regions:
            descs = self._agent.get_reg_descs([(ptr, size, 0, "")], "DRAM")
            self._pinned_regs.append(self._agent.register_memory(descs))
            ranges.append((ptr, ptr + size))
        self._pinned_ranges = tuple(ranges)

        self._fd_regs: dict[int, object] = {}
        self._batches: list[_Batch] = []
        self._inflight = 0
        self._closed = False
        logger.info(
            "NixlIoContext: plugin=%s queue_depth=%d pinned_regions=%d",
            plugin,
            queue_depth,
            len(self._pinned_ranges),
        )

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def free_slots(self) -> int:
        return max(0, self.queue_depth - self._inflight)

    def submit_reads(self, requests) -> int:
        return self._submit(requests, "READ")

    def submit_writes(self, requests) -> int:
        return self._submit(requests, "WRITE")

    def poll(self, *, max_events: Optional[int] = None) -> tuple[AioCompletion, ...]:
        """Drain whole transfers that finished; never blocks."""
        if not self._batches:
            return ()
        completions: list[AioCompletion] = []
        still_running: list[_Batch] = []
        for batch in self._batches:
            if max_events is not None and len(completions) >= max_events:
                still_running.append(batch)
                continue
            state = self._agent.check_xfer_state(batch.xfer)
            if state == "DONE":
                completions.extend(
                    AioCompletion(user_data=ud, result=nbytes)
                    for ud, nbytes in batch.entries
                )
            elif state == "ERR":
                completions.extend(
                    AioCompletion(user_data=ud, result=-_EIO) for ud, _ in batch.entries
                )
            else:
                still_running.append(batch)
                continue
            self._retire(batch)
        self._batches = still_running
        return tuple(completions)

    def wait(
        self,
        *,
        min_events: int = 1,
        max_events: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[AioCompletion, ...]:
        """Poll-only: NIXL exposes no blocking completion wait."""
        return self.poll(max_events=max_events)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for batch in self._batches:
            self._retire(batch)
        self._batches = []
        for reg in self._fd_regs.values():
            self._deregister(reg)
        self._fd_regs = {}
        for reg in self._pinned_regs:
            self._deregister(reg)
        self._pinned_regs = []

    def _submit(self, requests, direction: str) -> int:
        if self._closed:
            raise RuntimeError("submit on a closed NixlIoContext")
        accepted = min(len(requests), self.free_slots)
        if accepted <= 0:
            return 0
        batch = list(requests[:accepted])

        for fd, _ptr, _nbytes, _offset, _ud in batch:
            self._ensure_fd_registered(fd)

        unpinned = [
            (ptr, nbytes)
            for _fd, ptr, nbytes, _offset, _ud in batch
            if not self._is_pinned(ptr, nbytes)
        ]
        dram_reg = None
        if unpinned:
            descs = self._agent.get_reg_descs(
                [(ptr, nbytes, 0, "") for ptr, nbytes in unpinned], "DRAM"
            )
            dram_reg = self._agent.register_memory(descs)

        dram_descs = self._agent.get_xfer_descs(
            [(ptr, nbytes, 0) for _fd, ptr, nbytes, _offset, _ud in batch], "DRAM"
        )
        file_descs = self._agent.get_xfer_descs(
            [(offset, nbytes, fd) for fd, _ptr, nbytes, offset, _ud in batch], "FILE"
        )
        try:
            xfer = self._agent.initialize_xfer(
                direction, dram_descs, file_descs, self._name
            )
            state = self._agent.transfer(xfer)
        except Exception as error:  # NIXL raises bare Exception on bad descs
            if dram_reg is not None:
                self._deregister(dram_reg)
            logger.error("NIXL %s submission failed: %s", direction, error)
            return 0
        if state == "ERR":
            self._agent.release_xfer_handle(xfer)
            if dram_reg is not None:
                self._deregister(dram_reg)
            logger.error("NIXL %s rejected the transfer", direction)
            return 0

        entries = [(ud, nbytes) for _fd, _ptr, nbytes, _offset, ud in batch]
        self._batches.append(_Batch(xfer=xfer, entries=entries, dram_reg=dram_reg))
        self._inflight += accepted
        return accepted

    def _retire(self, batch: _Batch) -> None:
        self._inflight -= len(batch.entries)
        try:
            self._agent.release_xfer_handle(batch.xfer)
        except Exception as error:
            logger.debug("release_xfer_handle failed: %s", error)
        if batch.dram_reg is not None:
            self._deregister(batch.dram_reg)

    def _is_pinned(self, ptr: int, nbytes: int) -> bool:
        end = ptr + nbytes
        return any(ptr >= low and end <= high for low, high in self._pinned_ranges)

    def _ensure_fd_registered(self, fd: int) -> None:
        if fd in self._fd_regs:
            return
        # The POSIX plugin keys its registration on the descriptor; the path is
        # metadata, and /proc is the only way back to it from a cached fd.
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            path = ""
        size = os.fstat(fd).st_size
        descs = self._agent.get_reg_descs([(0, size, fd, path)], "FILE")
        self._fd_regs[fd] = self._agent.register_memory(descs)

    def forget_fd(self, fd: int) -> None:
        """Deregister a descriptor the backend is about to close."""
        reg = self._fd_regs.pop(fd, None)
        if reg is not None:
            self._deregister(reg)

    def _deregister(self, reg) -> None:
        try:
            self._agent.deregister_memory(reg)
        except Exception as error:
            logger.debug("deregister_memory failed: %s", error)
