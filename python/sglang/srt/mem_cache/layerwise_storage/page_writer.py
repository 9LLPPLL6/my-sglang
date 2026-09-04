"""Write-through publisher for layerwise HiCache page files.

A page becomes visible only through an atomic rename, so a reader can never
observe a half-written payload and a crash leaves a temp file rather than a
corrupt cache entry.  The K and V regions are padded independently to the
filesystem's Direct I/O alignment; padding exists only on disk and never
reaches the host KV buffer.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import uuid
from typing import Optional

from sglang.srt.mem_cache.layerwise_storage.aio_engine import (
    AlignedBuffer,
    AlignmentProfile,
    probe_alignment,
)
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    InvalidPageHeader,
    PageIdentity,
    PageLayout,
    build_page_layout,
    decode_header,
    encode_header,
    page_relative_path,
)
from sglang.srt.mem_cache.layerwise_storage.page_store_evictor import PageStoreEvictor

logger = logging.getLogger(__name__)


class _WriterLocal(threading.local):
    """Per-thread staging so concurrent write-through threads never share it."""

    def __init__(self):
        self.staging = None
        self.header = None


class PageFileWriter:
    """Publishes one page per call using a reusable aligned staging buffer."""

    def __init__(
        self,
        *,
        root: str,
        identity: PageIdentity,
        alignment_profile: Optional[AlignmentProfile] = None,
        require_direct_io: bool = True,
        evictor: Optional[PageStoreEvictor] = None,
    ):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.alignment_profile = alignment_profile or probe_alignment(
            self.root, require_direct=require_direct_io
        )
        self.direct_io = self.alignment_profile.direct_io_available
        if require_direct_io and not self.direct_io:
            raise RuntimeError(
                f"O_DIRECT is required but unavailable under {self.root!r}"
            )
        self.layout: PageLayout = build_page_layout(
            identity, alignment=self.alignment_profile.alignment
        )
        self._header = encode_header(self.layout)
        self._local = _WriterLocal()
        self.evictor = (
            evictor if evictor is not None else PageStoreEvictor(root=self.root)
        )
        self._region_logical = self.layout.identity.layer_num * self.layout.layer_stride

    def page_path(self, page_key: str) -> str:
        return os.path.join(self.root, self._relative_path(page_key))

    def exists(self, page_key: str) -> bool:
        path = self.page_path(page_key)
        if not os.path.exists(path):
            return False
        self.evictor.touch(path)
        return True

    def write_page(self, page_key: str, *, k_ptr: int, v_ptr: int) -> bool:
        """Publish one page from the host pool's K and V slices for that page."""
        staging = self._staging()
        base = staging.ptr
        ctypes.memmove(base, self._header_ptr(), self.layout.header_nbytes)
        ctypes.memmove(base + self.layout.k_offset, k_ptr, self._region_logical)
        ctypes.memmove(base + self.layout.v_offset, v_ptr, self._region_logical)
        self._zero_padding(base)
        return self._publish(page_key, staging)

    def write_page_bytes(
        self, page_key: str, *, k_bytes: bytes, v_bytes: bytes
    ) -> bool:
        if len(k_bytes) != self._region_logical or len(v_bytes) != self._region_logical:
            raise ValueError(
                f"each region must be {self._region_logical} bytes, got "
                f"{len(k_bytes)} and {len(v_bytes)}"
            )
        k_buffer = (ctypes.c_char * len(k_bytes)).from_buffer_copy(k_bytes)
        v_buffer = (ctypes.c_char * len(v_bytes)).from_buffer_copy(v_bytes)
        return self.write_page(
            page_key,
            k_ptr=ctypes.addressof(k_buffer),
            v_ptr=ctypes.addressof(v_buffer),
        )

    def read_layout(self, page_key: str) -> PageLayout:
        """Parse a published page's header, failing closed on any mismatch."""
        path = self.page_path(page_key)
        with open(path, "rb", buffering=0) as handle:
            raw = handle.read(self.layout.header_nbytes)
        layout = decode_header(raw)
        if layout.identity != self.layout.identity:
            raise InvalidPageHeader(
                f"page {page_key!r} was written by a different model or shard"
            )
        return layout

    def _relative_path(self, page_key: str) -> str:
        return page_relative_path(
            fingerprint=self.layout.identity.fingerprint,
            tp_size=self.layout.identity.tp_size,
            tp_rank=self.layout.identity.tp_rank,
            page_key=page_key,
        )

    def _staging(self) -> AlignedBuffer:
        buffer = self._local.staging
        if buffer is None:
            buffer = AlignedBuffer(
                self.layout.file_nbytes,
                alignment=self.alignment_profile.memory_alignment,
            )
            self._local.staging = buffer
        return buffer

    def _header_ptr(self) -> int:
        header = self._local.header
        if header is None:
            header = (ctypes.c_char * len(self._header)).from_buffer_copy(self._header)
            self._local.header = header
        return ctypes.addressof(header)

    def _zero_padding(self, base: int) -> None:
        k_padding = self.layout.v_offset - (self.layout.k_offset + self._region_logical)
        if k_padding:
            ctypes.memset(
                base + self.layout.k_offset + self._region_logical, 0, k_padding
            )
        v_end = self.layout.v_offset + self._region_logical
        v_padding = self.layout.file_nbytes - v_end
        if v_padding:
            ctypes.memset(base + v_end, 0, v_padding)

    def _publish(self, page_key: str, staging: AlignedBuffer) -> bool:
        path = self.page_path(page_key)
        if not self.evictor.reserve(path, self.layout.file_nbytes):
            logger.debug("Declined to publish HiCache page %s: store is full", page_key)
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if self.direct_io:
            flags |= os.O_DIRECT
        fd = None
        try:
            fd = os.open(tmp_path, flags, 0o644)
            written = _pwrite(fd, staging.ptr, self.layout.file_nbytes, 0)
            if written != self.layout.file_nbytes:
                raise OSError(
                    f"short write for {page_key!r}: {written} of "
                    f"{self.layout.file_nbytes} bytes"
                )
            os.close(fd)
            fd = None
            os.replace(tmp_path, path)
            self.evictor.commit(path)
            return True
        except OSError as error:
            logger.error("Failed to publish HiCache page %s: %s", page_key, error)
            self.evictor.abort(path)
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return False


def _pwrite(fd: int, ptr: int, nbytes: int, offset: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pwrite.restype = ctypes.c_ssize_t
    libc.pwrite.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
    ]
    written = 0
    while written < nbytes:
        result = libc.pwrite(
            fd, ctypes.c_void_p(ptr + written), nbytes - written, offset + written
        )
        if result <= 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), "pwrite")
        written += result
    return written
