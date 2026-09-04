"""Linux Direct I/O + asynchronous submission primitives for HiCache L3.

The layerwise pipeline needs many small K/V range reads in flight at once and a
non-blocking way to observe their completions.  ``pread`` on a thread pool
cannot provide that: the pool size caps the queue depth, and every completion
costs a GIL round trip.  This module binds the Linux AIO syscalls directly, so
one submission carries a whole layer group and ``poll`` drains completions
without blocking the caller.

Two deliberate constraints:

* ``O_DIRECT`` is requested, verified, and never silently downgraded.  A
  buffered fallback would invalidate every bandwidth measurement taken with it.
* A completion is emitted only when the kernel is done with the target buffer,
  so a cancelled or failed read never leaves a write racing against reuse.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import platform
import threading
from typing import NamedTuple, Optional

DEFAULT_ALIGNMENT = 4096
_ALIGNMENT_CANDIDATES = (512, 4096, 16384, 65536)

IOCB_CMD_PREAD = 0
IOCB_CMD_PWRITE = 1

# io_setup / io_destroy / io_getevents / io_submit / io_cancel
_SYSCALL_NUMBERS = {
    "x86_64": (206, 207, 208, 209, 210),
    "aarch64": (0, 1, 4, 2, 3),
}


class DirectIOUnavailable(RuntimeError):
    """``O_DIRECT`` cannot be used, and the caller asked not to fall back."""


class AioSubmissionError(OSError):
    """``io_submit`` rejected a batch before any of its reads started."""


class _IOCB(ctypes.Structure):
    _fields_ = [
        ("aio_data", ctypes.c_uint64),
        ("aio_key", ctypes.c_uint32),
        ("aio_rw_flags", ctypes.c_int32),
        ("aio_lio_opcode", ctypes.c_uint16),
        ("aio_reqprio", ctypes.c_int16),
        ("aio_fildes", ctypes.c_uint32),
        ("aio_buf", ctypes.c_uint64),
        ("aio_nbytes", ctypes.c_uint64),
        ("aio_offset", ctypes.c_int64),
        ("aio_reserved2", ctypes.c_uint64),
        ("aio_flags", ctypes.c_uint32),
        ("aio_resfd", ctypes.c_uint32),
    ]


class _IOEvent(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_uint64),
        ("obj", ctypes.c_uint64),
        ("res", ctypes.c_int64),
        ("res2", ctypes.c_int64),
    ]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class AioCompletion(NamedTuple):
    """One terminal kernel completion; the target buffer is free afterwards."""

    user_data: int
    result: int

    @property
    def failed(self) -> bool:
        return self.result < 0

    @property
    def error(self) -> Optional[OSError]:
        if self.result >= 0:
            return None
        return OSError(-self.result, os.strerror(-self.result))


class AlignmentProfile(NamedTuple):
    """Direct I/O constraints of one filesystem, probed at startup."""

    memory_alignment: int
    offset_alignment: int
    length_alignment: int
    max_transfer_nbytes: int
    direct_io_available: bool

    @property
    def alignment(self) -> int:
        """The single alignment a page layout must satisfy on this filesystem."""
        return max(self.memory_alignment, self.offset_alignment, self.length_alignment)


class AlignedBuffer:
    """Page-aligned host memory usable as an ``O_DIRECT`` bounce buffer.

    Allocated once and reused: allocating on the read path would put a
    ``malloc`` and a page-fault storm inside the latency the pipeline exists to
    hide.
    """

    def __init__(self, nbytes: int, *, alignment: int = DEFAULT_ALIGNMENT):
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        _require_power_of_two(alignment)
        self.nbytes = nbytes
        self.alignment = alignment
        self._storage = bytearray(nbytes + alignment)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._storage))
        self._offset = (-base) % alignment
        self.ptr = base + self._offset

    @property
    def view(self) -> memoryview:
        return memoryview(self._storage)[self._offset : self._offset + self.nbytes]

    def copy_out(self, *, src_offset: int, nbytes: int, dst_ptr: int) -> None:
        """Copy valid bytes out of a covering read into a compact target."""
        if src_offset < 0 or nbytes < 0 or src_offset + nbytes > self.nbytes:
            raise ValueError(
                f"[{src_offset}, {src_offset + nbytes}) is outside a "
                f"{self.nbytes}-byte bounce buffer"
            )
        ctypes.memmove(dst_ptr, self.ptr + src_offset, nbytes)


class LinuxAioContext:
    """A bounded queue of in-flight reads over the raw Linux AIO syscalls."""

    def __init__(self, *, queue_depth: int):
        if queue_depth <= 0:
            raise ValueError(f"queue_depth must be positive, got {queue_depth}")
        machine = platform.machine()
        if machine not in _SYSCALL_NUMBERS:
            raise RuntimeError(
                f"Linux AIO syscall numbers are unknown for machine {machine!r}"
            )
        (
            self._nr_setup,
            self._nr_destroy,
            self._nr_getevents,
            self._nr_submit,
            self._nr_cancel,
        ) = _SYSCALL_NUMBERS[machine]

        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.syscall.restype = ctypes.c_long
        self.queue_depth = queue_depth
        self._lock = threading.Lock()
        self._inflight = 0
        self._closed = False
        # Keep submitted iocbs alive until their completion is drained; the
        # kernel dereferences them and Python would otherwise free them.
        self._pending: dict[int, _IOCB] = {}
        self._event_buffer = (_IOEvent * queue_depth)()

        context = ctypes.c_ulong(0)
        self._checked(
            self._libc.syscall(
                ctypes.c_long(self._nr_setup),
                ctypes.c_long(queue_depth),
                ctypes.byref(context),
            ),
            "io_setup",
        )
        self._context = context

    @staticmethod
    def _checked(result: int, operation: str) -> int:
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), operation)
        return result

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def free_slots(self) -> int:
        with self._lock:
            return self.queue_depth - self._inflight

    def submit_reads(self, requests) -> int:
        """Submit ``(fd, ptr, nbytes, offset, user_data)`` reads as one batch.

        Returns the number accepted.  A partial accept is normal back pressure,
        not an error: the caller resubmits the tail on a later turn.
        """
        return self._submit(requests, opcode=IOCB_CMD_PREAD)

    def submit_writes(self, requests) -> int:
        return self._submit(requests, opcode=IOCB_CMD_PWRITE)

    def poll(self, *, max_events: Optional[int] = None) -> tuple[AioCompletion, ...]:
        """Drain finished reads without blocking."""
        return self._get_events(min_events=0, max_events=max_events, timeout_s=0.0)

    def wait(
        self,
        *,
        min_events: int = 1,
        max_events: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[AioCompletion, ...]:
        return self._get_events(
            min_events=min_events, max_events=max_events, timeout_s=timeout_s
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._libc.syscall(
            ctypes.c_long(self._nr_destroy), ctypes.c_ulong(self._context.value)
        )
        self._pending.clear()

    def _submit(self, requests, *, opcode: int) -> int:
        requests = tuple(requests)
        if not requests:
            return 0
        with self._lock:
            if self._closed:
                raise RuntimeError("AIO context is closed")
            capacity = self.queue_depth - self._inflight
        if capacity <= 0:
            return 0

        batch = requests[:capacity]
        blocks = []
        for fd, ptr, nbytes, offset, user_data in batch:
            block = _IOCB()
            block.aio_lio_opcode = opcode
            block.aio_fildes = fd
            block.aio_buf = ptr
            block.aio_nbytes = nbytes
            block.aio_offset = offset
            block.aio_data = user_data
            blocks.append(block)

        pointers = (ctypes.POINTER(_IOCB) * len(blocks))(
            *(ctypes.pointer(block) for block in blocks)
        )
        accepted = self._libc.syscall(
            ctypes.c_long(self._nr_submit),
            ctypes.c_ulong(self._context.value),
            ctypes.c_long(len(blocks)),
            pointers,
        )
        if accepted < 0:
            raise AioSubmissionError(
                ctypes.get_errno(), os.strerror(ctypes.get_errno()), "io_submit"
            )
        with self._lock:
            for block in blocks[:accepted]:
                self._pending[block.aio_data] = block
            self._inflight += accepted
        return accepted

    def _get_events(
        self,
        *,
        min_events: int,
        max_events: Optional[int],
        timeout_s: Optional[float],
    ) -> tuple[AioCompletion, ...]:
        limit = (
            self.queue_depth
            if max_events is None
            else min(max_events, self.queue_depth)
        )
        if limit <= 0:
            return ()
        with self._lock:
            if self._closed:
                return ()
            if self._inflight == 0:
                return ()
        timeout = None
        if timeout_s is not None:
            timeout = _Timespec(
                int(timeout_s), int((timeout_s - int(timeout_s)) * 1_000_000_000)
            )
        count = self._libc.syscall(
            ctypes.c_long(self._nr_getevents),
            ctypes.c_ulong(self._context.value),
            ctypes.c_long(min_events),
            ctypes.c_long(limit),
            self._event_buffer,
            None if timeout is None else ctypes.byref(timeout),
        )
        if count < 0:
            code = ctypes.get_errno()
            if code == errno.EINTR:
                return ()
            raise OSError(code, os.strerror(code), "io_getevents")

        completions = []
        with self._lock:
            for index in range(count):
                event = self._event_buffer[index]
                self._pending.pop(event.data, None)
                completions.append(
                    AioCompletion(user_data=event.data, result=event.res)
                )
            self._inflight -= count
        return tuple(completions)


class DirectIOFileCache:
    """Bounded cache of ``O_DIRECT`` descriptors keyed by path.

    A parallel filesystem charges real latency for ``open``; the read path must
    not pay it per page.  The cache is bounded so a long-running server cannot
    exhaust its descriptor limit.
    """

    def __init__(self, *, capacity: int = 1024):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._descriptors: dict[str, int] = {}
        # Reference counts keep a descriptor open while reads are in flight,
        # so an eviction can never close a file the kernel is still writing to.
        self._refcounts: dict[str, int] = {}

    def acquire(self, path: str, *, direct: bool = True) -> int:
        with self._lock:
            fd = self._descriptors.get(path)
            if fd is None:
                fd = _open_direct(path, direct=direct)
                self._descriptors[path] = fd
                self._refcounts[path] = 0
            self._refcounts[path] += 1
            self._evict_locked()
            return fd

    def release(self, path: str) -> None:
        with self._lock:
            if path not in self._refcounts:
                return
            self._refcounts[path] -= 1
            self._evict_locked()

    def invalidate(self, path: str) -> None:
        with self._lock:
            if self._refcounts.get(path):
                return
            fd = self._descriptors.pop(path, None)
            self._refcounts.pop(path, None)
        if fd is not None:
            os.close(fd)

    def close(self) -> None:
        with self._lock:
            descriptors = tuple(self._descriptors.values())
            self._descriptors.clear()
            self._refcounts.clear()
        for fd in descriptors:
            os.close(fd)

    def _evict_locked(self) -> None:
        if len(self._descriptors) <= self.capacity:
            return
        for path in tuple(self._descriptors):
            if len(self._descriptors) <= self.capacity:
                return
            if self._refcounts.get(path):
                continue
            fd = self._descriptors.pop(path)
            self._refcounts.pop(path, None)
            os.close(fd)


def probe_alignment(directory: str, *, require_direct: bool = True) -> AlignmentProfile:
    """Measure the Direct I/O constraints of ``directory``.

    Filesystems disagree: ``tmpfs`` rejects ``O_DIRECT`` outright, ext4 accepts
    512-byte alignment, and several parallel filesystems demand 4 KiB or more.
    Hard-coding 4096 would either waste bandwidth or fail at runtime, so probe
    once at startup and let the page layout follow the answer.
    """
    os.makedirs(directory, exist_ok=True)
    probe_path = os.path.join(directory, f".hicache-direct-probe.{os.getpid()}")
    largest = _ALIGNMENT_CANDIDATES[-1]
    try:
        with open(probe_path, "wb") as handle:
            handle.write(b"\x00" * (2 * largest))
        for candidate in _ALIGNMENT_CANDIDATES:
            if _direct_read_works(probe_path, alignment=candidate):
                return AlignmentProfile(
                    memory_alignment=candidate,
                    offset_alignment=candidate,
                    length_alignment=candidate,
                    max_transfer_nbytes=_max_transfer_nbytes(),
                    direct_io_available=True,
                )
        if require_direct:
            raise DirectIOUnavailable(
                f"O_DIRECT is not usable under {directory!r}; refusing a silent "
                "buffered fallback because it would invalidate bandwidth results"
            )
        return AlignmentProfile(
            memory_alignment=DEFAULT_ALIGNMENT,
            offset_alignment=DEFAULT_ALIGNMENT,
            length_alignment=DEFAULT_ALIGNMENT,
            max_transfer_nbytes=_max_transfer_nbytes(),
            direct_io_available=False,
        )
    finally:
        try:
            os.unlink(probe_path)
        except OSError:
            pass


def _direct_read_works(path: str, *, alignment: int) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return False
    try:
        buffer = AlignedBuffer(alignment, alignment=alignment)
        read = _pread_into(fd, buffer.ptr, alignment, 0)
        return read == alignment
    except OSError:
        return False
    finally:
        os.close(fd)


def _pread_into(fd: int, ptr: int, nbytes: int, offset: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pread.restype = ctypes.c_ssize_t
    libc.pread.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
    ]
    result = libc.pread(fd, ctypes.c_void_p(ptr), nbytes, offset)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), "pread")
    return result


def _open_direct(path: str, *, direct: bool) -> int:
    flags = os.O_RDONLY
    if direct:
        flags |= os.O_DIRECT
    return os.open(path, flags)


def _max_transfer_nbytes() -> int:
    # Conservative default; the kernel splits larger requests anyway and every
    # supported backend advertises at least this much per operation.
    return 1 << 24


def _require_power_of_two(value: int) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"alignment must be a positive power of two, got {value}")
