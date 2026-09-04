"""Turns one storage-backed prefix hit into an ordered layer-group read plan.

Two geometries meet here.  The host side comes from the page-major host pool:
where this rank's compact K/V slice for a page and layer range lives.  The file
side comes from :mod:`page_format`: where those same bytes sit inside a padded,
alignment-friendly page object.  Keeping both in one place means the pipeline
never recomputes an offset, and an alignment change lands in exactly one file.

Group 0 is split out with its own layer count: it is the only group whose
latency no computation can hide, so it is deliberately made small even when the
steady-state groups are large.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Sequence

from sglang.srt.mem_cache.layerwise_storage.page_format import PageLayout
from sglang.srt.mem_cache.layerwise_storage.types import (
    HostTargetBase,
    KVPart,
    LayerGroupPlan,
    LayerwiseReadPlan,
    LayerwiseStorageExtent,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost


class LayerGroupSpec(NamedTuple):
    group_id: int
    layer_start: int
    layer_end: int


def split_layer_groups(
    *,
    layer_num: int,
    first_group_layers: int,
    group_size: int,
) -> tuple[LayerGroupSpec, ...]:
    """Split ``[0, layer_num)`` into a small first group and steady groups."""
    if layer_num <= 0:
        raise ValueError(f"layer_num must be positive, got {layer_num}")
    if first_group_layers <= 0 or group_size <= 0:
        raise ValueError("first_group_layers and group_size must be positive")

    groups = []
    layer_start = 0
    while layer_start < layer_num:
        span = first_group_layers if not groups else group_size
        layer_end = min(layer_start + span, layer_num)
        groups.append(
            LayerGroupSpec(
                group_id=len(groups), layer_start=layer_start, layer_end=layer_end
            )
        )
        layer_start = layer_end
    return tuple(groups)


def build_read_plan(
    *,
    host_pool: MHATokenToKVPoolHost,
    host_indices: torch.Tensor,
    page_keys: Sequence[str],
    layout: PageLayout,
    first_group_layers: int,
    group_size: int,
) -> tuple[LayerwiseReadPlan, HostTargetBase]:
    """Plan every extent needed to stage ``page_keys`` into ``host_indices``."""
    page_size = host_pool.page_size
    if len(host_indices) != len(page_keys) * page_size:
        raise ValueError(
            f"host_indices covers {len(host_indices)} tokens but {len(page_keys)} "
            f"pages of {page_size} tokens were requested"
        )
    if host_pool.layer_num != layout.identity.layer_num:
        raise ValueError(
            f"host pool has {host_pool.layer_num} layers, page layout has "
            f"{layout.identity.layer_num}"
        )

    specs = split_layer_groups(
        layer_num=layout.identity.layer_num,
        first_group_layers=first_group_layers,
        group_size=group_size,
    )
    target = HostTargetBase(
        k_ptr=host_pool.k_buffer.data_ptr(), v_ptr=host_pool.v_buffer.data_ptr()
    )

    groups = []
    for spec in specs:
        host_plan = host_pool.get_layer_group_buffer_meta(
            host_indices,
            spec.layer_start,
            spec.layer_end,
            file_payload_offset=layout.k_offset,
            alignment=layout.alignment,
        )
        groups.append(
            LayerGroupPlan(
                group_id=spec.group_id,
                layer_start=spec.layer_start,
                layer_end=spec.layer_end,
                extents=_build_extents(
                    host_plan=host_plan,
                    page_keys=page_keys,
                    layout=layout,
                    spec=spec,
                ),
            )
        )
    return LayerwiseReadPlan(groups=tuple(groups)), target


def _build_extents(
    *,
    host_plan,
    page_keys: Sequence[str],
    layout: PageLayout,
    spec: LayerGroupSpec,
) -> tuple[LayerwiseStorageExtent, ...]:
    """Pair each host slice with the aligned file range that covers it.

    The file offset comes from the page layout rather than the host plan: only
    the layout knows about inter-region padding, and letting two modules derive
    it independently is how a silent off-by-one region shift would happen.
    """
    logical_nbytes = layout.layer_range_nbytes(spec.layer_start, spec.layer_end)
    alignment = layout.alignment
    extents = []
    for extent_id, host_extent in enumerate(host_plan.extents):
        # get_layer_group_buffer_meta emits pages in host_indices order, two
        # extents (K then V) per page.
        page_key = page_keys[extent_id // 2]
        kv_part = KVPart.KEY if host_extent.kv == "K" else KVPart.VALUE
        logical_offset = layout.layer_range_offset(host_extent.kv, spec.layer_start)
        io_offset = logical_offset // alignment * alignment
        payload_offset = logical_offset - io_offset
        io_nbytes = _align_up(payload_offset + logical_nbytes, alignment)
        extents.append(
            LayerwiseStorageExtent(
                extent_id=extent_id,
                storage_key=page_key,
                kv_part=kv_part,
                layer_start=spec.layer_start,
                layer_end=spec.layer_end,
                io_offset=io_offset,
                io_nbytes=io_nbytes,
                payload_offset=payload_offset,
                payload_nbytes=logical_nbytes,
                target_offset=host_extent.buffer_offset,
            )
        )
    return tuple(extents)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
