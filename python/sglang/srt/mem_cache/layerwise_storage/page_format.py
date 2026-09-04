"""On-disk page format for the layerwise (L2/L3 fused) HiCache storage tier.

One logical KV page maps to one storage object whose payload is byte-identical
to the ``page_first_direct`` flat data page produced by the host pool:

```text
[aligned header]
K[layer 0 .. layer N-1]      each layer: page_size x local_kv_heads x head_dim
[padding to alignment]
V[layer 0 .. layer N-1]
[padding to alignment]
```

Keeping the payload identical to ``get_data_page(..., flat=True)`` means the
existing write-through path and the layerwise range-read path agree without a
second copy of the KV layout.  The header is a fixed-size aligned prefix so the
first K layer starts on a Direct I/O boundary; identity fields in the header let
a reader fail closed on a stale or foreign file instead of loading wrong KV.
"""

from __future__ import annotations

import binascii
import hashlib
import struct
from typing import NamedTuple

FORMAT_VERSION = 1
FORMAT_DIR = f"format-v{FORMAT_VERSION}"
MAGIC = b"SGLKVPG\x01"
DEFAULT_ALIGNMENT = 4096
# Number of hash buckets between the rank directory and the page files. A flat
# rank directory degrades badly on most parallel filesystems once it holds
# hundreds of thousands of entries.
BUCKET_COUNT = 256

# magic, header_nbytes, format_version, fingerprint, tp_size, tp_rank,
# dtype_name, layer_num, page_size, local_kv_heads, head_dim, element_size,
# k_offset, v_offset, layer_stride, logical_nbytes, physical_nbytes,
# alignment, flags, payload_crc32, header_crc32
_HEADER_STRUCT = struct.Struct("<8sII32sII16sIIIIIQQQQQIIII")
FLAG_COMPLETE = 1 << 0
FLAG_PAYLOAD_CRC = 1 << 1


class InvalidPageHeader(ValueError):
    """The bytes read from storage are not a usable page header."""


class PageIdentity(NamedTuple):
    """Everything a reader must agree on before trusting a page payload."""

    fingerprint: bytes
    tp_size: int
    tp_rank: int
    dtype_name: str
    layer_num: int
    page_size: int
    local_kv_heads: int
    head_dim: int
    element_size: int


class PageLayout(NamedTuple):
    """Byte geometry of one page object, derived from a :class:`PageIdentity`."""

    identity: PageIdentity
    alignment: int
    header_nbytes: int
    k_offset: int
    v_offset: int
    layer_stride: int
    logical_nbytes: int
    physical_nbytes: int

    @property
    def file_nbytes(self) -> int:
        return self.header_nbytes + self.physical_nbytes

    @property
    def layer_stride_aligned(self) -> bool:
        return self.layer_stride % self.alignment == 0

    def region_offset(self, kv_part: str) -> int:
        """File offset of layer 0 for ``"K"`` or ``"V"``."""
        if kv_part == "K":
            return self.k_offset
        if kv_part == "V":
            return self.v_offset
        raise ValueError(f"kv_part must be 'K' or 'V', got {kv_part!r}")

    def layer_range_offset(self, kv_part: str, layer_start: int) -> int:
        if not 0 <= layer_start < self.identity.layer_num:
            raise ValueError(
                f"layer_start must be in [0, {self.identity.layer_num}), "
                f"got {layer_start}"
            )
        return self.region_offset(kv_part) + layer_start * self.layer_stride

    def layer_range_nbytes(self, layer_start: int, layer_end: int) -> int:
        if not 0 <= layer_start < layer_end <= self.identity.layer_num:
            raise ValueError(
                "layer range must satisfy "
                f"0 <= layer_start < layer_end <= {self.identity.layer_num}, "
                f"got [{layer_start}, {layer_end})"
            )
        return (layer_end - layer_start) * self.layer_stride


def build_page_layout(
    identity: PageIdentity,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> PageLayout:
    """Derive the on-disk geometry for one page.

    The K and V regions are padded independently so that a V-region range read
    starts on an alignment boundary even when the compact K region does not end
    on one.  Padding is never exposed as host capacity; it only exists on disk.
    """
    _validate_identity(identity)
    _require_alignment(alignment)

    layer_stride = (
        identity.page_size
        * identity.local_kv_heads
        * identity.head_dim
        * identity.element_size
    )
    region_logical = identity.layer_num * layer_stride
    region_physical = _align_up(region_logical, alignment)
    header_nbytes = _align_up(_HEADER_STRUCT.size, alignment)
    return PageLayout(
        identity=identity,
        alignment=alignment,
        header_nbytes=header_nbytes,
        k_offset=header_nbytes,
        v_offset=header_nbytes + region_physical,
        layer_stride=layer_stride,
        logical_nbytes=2 * region_logical,
        physical_nbytes=2 * region_physical,
    )


def encode_header(
    layout: PageLayout,
    *,
    complete: bool = True,
    payload_crc32: int = 0,
) -> bytes:
    """Serialize a page header padded to ``layout.header_nbytes``."""
    identity = layout.identity
    flags = FLAG_COMPLETE if complete else 0
    if payload_crc32:
        flags |= FLAG_PAYLOAD_CRC
    fields = (
        MAGIC,
        layout.header_nbytes,
        FORMAT_VERSION,
        identity.fingerprint,
        identity.tp_size,
        identity.tp_rank,
        identity.dtype_name.encode("ascii").ljust(16, b"\x00"),
        identity.layer_num,
        identity.page_size,
        identity.local_kv_heads,
        identity.head_dim,
        identity.element_size,
        layout.k_offset,
        layout.v_offset,
        layout.layer_stride,
        layout.logical_nbytes,
        layout.physical_nbytes,
        layout.alignment,
        flags,
        payload_crc32,
    )
    body = _HEADER_STRUCT.pack(*fields, 0)[: _HEADER_STRUCT.size - 4]
    header = body + struct.pack("<I", binascii.crc32(body) & 0xFFFFFFFF)
    return header.ljust(layout.header_nbytes, b"\x00")


def decode_header(raw: bytes) -> PageLayout:
    """Parse and validate a page header prefix.

    Raises :class:`InvalidPageHeader` rather than returning a partially trusted
    layout: a caller that cannot fully identify a page must fall back to
    recompute instead of loading unknown bytes into the KV cache.
    """
    if len(raw) < _HEADER_STRUCT.size:
        raise InvalidPageHeader(
            f"header needs {_HEADER_STRUCT.size} bytes, got {len(raw)}"
        )
    body = raw[: _HEADER_STRUCT.size - 4]
    (stored_crc,) = struct.unpack_from("<I", raw, _HEADER_STRUCT.size - 4)
    if stored_crc != binascii.crc32(body) & 0xFFFFFFFF:
        raise InvalidPageHeader("header checksum mismatch")

    (
        magic,
        header_nbytes,
        version,
        fingerprint,
        tp_size,
        tp_rank,
        dtype_raw,
        layer_num,
        page_size,
        local_kv_heads,
        head_dim,
        element_size,
        k_offset,
        v_offset,
        layer_stride,
        logical_nbytes,
        physical_nbytes,
        alignment,
        flags,
        payload_crc32,
        _,
    ) = _HEADER_STRUCT.unpack(raw[: _HEADER_STRUCT.size])
    if magic != MAGIC:
        raise InvalidPageHeader(f"bad magic {magic!r}")
    if version != FORMAT_VERSION:
        raise InvalidPageHeader(
            f"unsupported page format version {version}, expected {FORMAT_VERSION}"
        )
    if not flags & FLAG_COMPLETE:
        raise InvalidPageHeader("page is not marked complete")

    identity = PageIdentity(
        fingerprint=fingerprint,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype_name=dtype_raw.rstrip(b"\x00").decode("ascii"),
        layer_num=layer_num,
        page_size=page_size,
        local_kv_heads=local_kv_heads,
        head_dim=head_dim,
        element_size=element_size,
    )
    layout = build_page_layout(identity, alignment=alignment)
    if (
        layout.header_nbytes != header_nbytes
        or layout.k_offset != k_offset
        or layout.v_offset != v_offset
        or layout.layer_stride != layer_stride
        or layout.logical_nbytes != logical_nbytes
        or layout.physical_nbytes != physical_nbytes
    ):
        raise InvalidPageHeader("header geometry disagrees with its own identity")
    return layout


def header_payload_crc32(raw: bytes) -> int:
    """Payload CRC recorded in a header, or 0 when the writer skipped it."""
    flags, payload_crc32 = struct.unpack_from("<II", raw, _HEADER_STRUCT.size - 12)
    return payload_crc32 if flags & FLAG_PAYLOAD_CRC else 0


def model_fingerprint(
    *,
    model_name: str,
    dtype_name: str,
    layer_num: int,
    page_size: int,
    local_kv_heads: int,
    head_dim: int,
    tp_size: int,
) -> bytes:
    """A 32-byte digest that changes whenever cached bytes stop being valid.

    ``tp_rank`` is deliberately excluded: every rank of one deployment shares a
    fingerprint and separates its shard through the directory tree, so a cache
    stays usable when ranks restart in a different order.
    """
    material = "\x1f".join(
        (
            model_name,
            dtype_name,
            str(layer_num),
            str(page_size),
            str(local_kv_heads),
            str(head_dim),
            str(tp_size),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).digest()


def page_relative_path(
    *,
    fingerprint: bytes,
    tp_size: int,
    tp_rank: int,
    page_key: str,
) -> str:
    """Storage-relative path of one page object.

    The path carries only identity that the directory tree must partition on.
    Everything else lives in the header, so a format change never requires
    renaming files.
    """
    if not page_key:
        raise ValueError("page_key must not be empty")
    safe_key = _sanitize_page_key(page_key)
    bucket = binascii.crc32(safe_key.encode("ascii")) % BUCKET_COUNT
    return (
        f"{FORMAT_DIR}/{fingerprint[:8].hex()}/tp-{tp_size}/rank-{tp_rank}/"
        f"{bucket:02x}/{safe_key}.kv"
    )


def _sanitize_page_key(page_key: str) -> str:
    """Map a HiCache page key onto a filesystem-safe, collision-free name."""
    if all(character.isalnum() or character in "-_" for character in page_key):
        return page_key
    return hashlib.sha256(page_key.encode("utf-8")).hexdigest()


def _validate_identity(identity: PageIdentity) -> None:
    if len(identity.fingerprint) != 32:
        raise ValueError("fingerprint must be exactly 32 bytes")
    if len(identity.dtype_name.encode("ascii")) > 16:
        raise ValueError("dtype_name must fit in 16 ASCII bytes")
    positive = (
        ("tp_size", identity.tp_size),
        ("layer_num", identity.layer_num),
        ("page_size", identity.page_size),
        ("local_kv_heads", identity.local_kv_heads),
        ("head_dim", identity.head_dim),
        ("element_size", identity.element_size),
    )
    for name, value in positive:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if not 0 <= identity.tp_rank < identity.tp_size:
        raise ValueError(
            f"tp_rank must be in [0, {identity.tp_size}), got {identity.tp_rank}"
        )


def _require_alignment(alignment: int) -> None:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a positive power of two, got {alignment}")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
