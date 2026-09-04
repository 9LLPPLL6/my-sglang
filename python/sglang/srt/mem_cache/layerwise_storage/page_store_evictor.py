"""Disk bounding for the layerwise HiCache page store.

An L3 tier whose whole point is "as big as your storage" will consume all of it
unless something says stop, and a full filesystem does not degrade gracefully:
the write fails, the scheduler process dies, and the server goes with it.  This
evictor keeps the page tree inside a byte cap and above a free-space watermark,
dropping least-recently-used pages first.

The watermark defaults to a non-zero value, unlike the older flat-file backend:
losing a cached page costs a recompute, while filling the device costs the
server.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.utils.common import human_readable_int

logger = logging.getLogger(__name__)

PAGE_SUFFIX = ".kv"


class PageStoreEvictor:
    """LRU bound over a nested page-file tree.

    Recency is tracked in memory and seeded from file mtimes at startup, so a
    restart inherits a reasonable order instead of evicting arbitrarily.
    """

    def __init__(
        self,
        *,
        root: str,
        max_bytes: Optional[int] = None,
        min_free_bytes: Optional[int] = None,
        eviction_ratio: float = 0.9,
    ):
        self.root = root
        self.max_bytes = (
            _parse_size(envs.SGLANG_HICACHE_LAYERWISE_MAX_SIZE.get())
            if max_bytes is None
            else max_bytes
        )
        self.min_free_bytes = (
            _parse_size(envs.SGLANG_HICACHE_LAYERWISE_MIN_FREE_SPACE.get())
            if min_free_bytes is None
            else min_free_bytes
        )
        if not 0.0 < eviction_ratio <= 1.0:
            raise ValueError(f"eviction_ratio must be in (0, 1], got {eviction_ratio}")
        self.eviction_ratio = eviction_ratio

        self._lock = threading.Lock()
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._used_bytes = 0
        self._reserved: dict[str, int] = {}
        if self.enabled:
            self._scan()

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0 or self.min_free_bytes > 0

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def reserve(self, path: str, nbytes: int) -> bool:
        """Admit one page, evicting first.  ``False`` means do not write it."""
        if not self.enabled:
            return True
        with self._lock:
            if self.max_bytes and nbytes > self.max_bytes:
                logger.warning(
                    "A %d-byte page cannot fit the %d-byte HiCache page store cap",
                    nbytes,
                    self.max_bytes,
                )
                return False
            self._evict_for_locked(nbytes)
            if self.max_bytes and self._used_bytes + nbytes > self.max_bytes:
                return False
            if not self._has_free_space_locked(nbytes):
                return False
            self._reserved[path] = nbytes
            self._used_bytes += nbytes
            return True

    def commit(self, path: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            nbytes = self._reserved.pop(path, None)
            if nbytes is None:
                return
            self._entries.pop(path, None)
            self._entries[path] = nbytes

    def abort(self, path: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            nbytes = self._reserved.pop(path, None)
            if nbytes is not None:
                self._used_bytes -= nbytes

    def touch(self, path: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if path in self._entries:
                self._entries.move_to_end(path)

    def _scan(self) -> None:
        for directory, _, filenames in os.walk(self.root):
            for filename in filenames:
                if not filename.endswith(PAGE_SUFFIX):
                    continue
                path = os.path.join(directory, filename)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                self._entries[path] = stat.st_size
                self._used_bytes += stat.st_size
        self._entries = OrderedDict(
            sorted(self._entries.items(), key=lambda item: _mtime(item[0]))
        )
        if self._entries:
            logger.info(
                "HiCache page store at %s holds %d pages, %.2f GiB",
                self.root,
                len(self._entries),
                self._used_bytes / (1 << 30),
            )

    def _evict_for_locked(self, nbytes: int) -> None:
        target = (
            int(self.max_bytes * self.eviction_ratio) - nbytes
            if self.max_bytes
            else None
        )
        while self._entries:
            over_cap = target is not None and self._used_bytes > target
            if not over_cap and self._has_free_space_locked(nbytes):
                return
            if not self._evict_one_locked():
                return

    def _evict_one_locked(self) -> bool:
        path, nbytes = self._entries.popitem(last=False)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning("Failed to evict HiCache page %s: %s", path, error)
            return False
        self._used_bytes -= nbytes
        return True

    def _has_free_space_locked(self, nbytes: int) -> bool:
        if not self.min_free_bytes:
            return True
        try:
            stats = os.statvfs(self.root)
        except OSError:
            return True
        free_bytes = stats.f_bavail * stats.f_frsize
        return free_bytes - nbytes >= self.min_free_bytes


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _parse_size(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value).strip()
    if not text or text == "0":
        return 0
    try:
        return max(0, human_readable_int(text))
    except (argparse.ArgumentTypeError, ValueError):
        logger.warning(
            "Invalid HiCache page store size %r; treating as unlimited", value
        )
        return 0
