"""Scheduler-facing entry point for the layerwise (L2/L3 fused) storage tier.

Everything the rest of HiCache needs to know about layer-group streaming lives
behind this object: page identity, the storage backend, the read plan, the
transaction pipeline, and the layer-range H2D session.  A cache implementation
holds one of these only when ``--hicache-storage-load-mode=layerwise`` is on,
so the default path keeps its existing code exactly.

The staging a transaction reads into stays private until every group is agreed
and the forward pass has consumed it.  Publishing earlier would let a second
request match a prefix whose later layers are still in flight.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.layerwise_storage.consensus import (
    GroupConsensus,
    SingleRankConsensus,
    TorchDistGroupConsensus,
)
from sglang.srt.mem_cache.layerwise_storage.file_backend import LayerwiseFileBackend
from sglang.srt.mem_cache.layerwise_storage.page_format import (
    PageIdentity,
    model_fingerprint,
)
from sglang.srt.mem_cache.layerwise_storage.page_writer import PageFileWriter
from sglang.srt.mem_cache.layerwise_storage.pipeline import (
    LayerwiseStoragePipeline,
    LayerwiseTransaction,
    PipelineConfig,
)
from sglang.srt.mem_cache.layerwise_storage.plan_builder import build_read_plan
from sglang.srt.mem_cache.layerwise_storage.state_machine import OwnershipState

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import (
        HiCacheController,
        StreamingLoadHandle,
    )
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class LayerwiseControllerConfig(NamedTuple):
    root: str
    first_group_layers: int
    group_size: int
    read_ahead_groups: int
    group_timeout_s: float
    admission_budget_s: float
    max_inflight_bytes: int
    slow_fallback: str
    queue_depth: int = 128


class _ActiveTransaction:
    """One streaming request: its storage transaction plus its H2D session."""

    def __init__(self, *, transaction: LayerwiseTransaction, page_keys, host_indices):
        self.transaction = transaction
        self.page_keys = tuple(page_keys)
        self.host_indices = host_indices
        self.load_handle: Optional[StreamingLoadHandle] = None
        self.submitted_layers = 0


class LayerwiseStorageController:
    def __init__(
        self,
        *,
        host_pool: MHATokenToKVPoolHost,
        cache_controller: HiCacheController,
        config: LayerwiseControllerConfig,
        identity: PageIdentity,
        consensus: GroupConsensus,
        backend: Optional[LayerwiseFileBackend] = None,
        writer: Optional[PageFileWriter] = None,
    ):
        self.host_pool = host_pool
        self.cache_controller = cache_controller
        self.config = config
        self.writer = writer or PageFileWriter(root=config.root, identity=identity)
        self.layout = self.writer.layout
        self.backend = backend or LayerwiseFileBackend(
            root=config.root,
            identity=identity,
            queue_depth=config.queue_depth,
            max_inflight_bytes=config.max_inflight_bytes,
            alignment_profile=self.writer.alignment_profile,
        )
        self.pipeline = LayerwiseStoragePipeline(
            backend=self.backend,
            consensus=consensus,
            config=PipelineConfig(
                read_ahead_groups=config.read_ahead_groups,
                group_timeout_s=config.group_timeout_s,
                admission_budget_s=config.admission_budget_s,
            ),
            submit_h2d_range=self._submit_h2d_range,
        )
        self._active: dict[str, _ActiveTransaction] = {}
        self._generation = 0

    @classmethod
    def from_server_args(
        cls,
        *,
        server_args: ServerArgs,
        host_pool: MHATokenToKVPoolHost,
        cache_controller: HiCacheController,
        tp_rank: int,
        tp_size: int,
        tp_group=None,
        model_name: str,
    ) -> LayerwiseStorageController:
        dtype_name = str(host_pool.dtype).removeprefix("torch.")
        identity = PageIdentity(
            fingerprint=model_fingerprint(
                model_name=model_name,
                dtype_name=dtype_name,
                layer_num=host_pool.layer_num,
                page_size=host_pool.page_size,
                local_kv_heads=host_pool.head_num,
                head_dim=host_pool.head_dim,
                tp_size=tp_size,
            ),
            tp_size=tp_size,
            tp_rank=tp_rank,
            dtype_name=dtype_name,
            layer_num=host_pool.layer_num,
            page_size=host_pool.page_size,
            local_kv_heads=host_pool.head_num,
            head_dim=host_pool.head_dim,
            element_size=host_pool.dtype.itemsize,
        )
        config = LayerwiseControllerConfig(
            root=_storage_root(server_args),
            first_group_layers=server_args.hicache_storage_first_group_layers,
            group_size=server_args.hicache_storage_group_size,
            read_ahead_groups=server_args.hicache_storage_read_ahead_groups,
            group_timeout_s=server_args.hicache_storage_group_timeout_ms / 1000.0,
            admission_budget_s=(
                server_args.hicache_storage_admission_budget_ms / 1000.0
            ),
            max_inflight_bytes=server_args.hicache_storage_max_inflight_bytes,
            slow_fallback=server_args.hicache_storage_slow_fallback,
        )
        consensus: GroupConsensus
        if tp_size > 1 and tp_group is not None:
            consensus = TorchDistGroupConsensus(
                group=tp_group, device=server_args.device
            )
        else:
            consensus = SingleRankConsensus()
        return cls(
            host_pool=host_pool,
            cache_controller=cache_controller,
            config=config,
            identity=identity,
            consensus=consensus,
        )

    def query_hit_pages(self, page_keys: Sequence[str]) -> int:
        """Length of the leading run of pages this rank has in storage."""
        for position, page_key in enumerate(page_keys):
            if not self.writer.exists(page_key):
                return position
        return len(page_keys)

    def begin(
        self,
        *,
        request_id: str,
        page_keys: Sequence[str],
        host_indices: torch.Tensor,
    ) -> LayerwiseTransaction:
        """Start streaming a storage-backed prefix into private host staging."""
        if request_id in self._active:
            raise RuntimeError(f"request {request_id!r} already has a transaction")
        plan, target = build_read_plan(
            host_pool=self.host_pool,
            host_indices=host_indices,
            page_keys=page_keys,
            layout=self.layout,
            first_group_layers=self.config.first_group_layers,
            group_size=self.config.group_size,
        )
        self._generation += 1
        transaction = self.pipeline.begin(
            transaction_id=request_id,
            generation=self._generation,
            plan=plan,
            target=target,
            host_resource_id=f"{request_id}:host-staging",
        )
        self._active[request_id] = _ActiveTransaction(
            transaction=transaction,
            page_keys=page_keys,
            host_indices=host_indices,
        )
        return transaction

    def poll(self) -> None:
        """Advance every active transaction; called once per scheduler step."""
        for entry in tuple(self._active.values()):
            self.pipeline.advance(entry.transaction)
            self._finish_h2d_if_complete(entry)

    def admission_ready(self, request_id: str) -> bool:
        entry = self._active.get(request_id)
        if entry is None:
            return False
        return entry.transaction.admission_ready

    def admission_over_budget(self, request_id: str) -> bool:
        entry = self._active.get(request_id)
        if entry is None:
            return False
        return self.pipeline.admission_over_budget(entry.transaction)

    def attach_device(
        self,
        *,
        request_id: str,
        device_indices: torch.Tensor,
        node_ids: Sequence[int],
    ) -> None:
        """Bind admitted device slots and open the layer-range H2D session."""
        entry = self._active[request_id]
        entry.load_handle = self.cache_controller.start_streaming_load(
            host_indices=entry.host_indices,
            device_indices=device_indices,
            node_ids=list(node_ids),
        )
        self.pipeline.note_device_allocated(
            entry.transaction, device_resource_id=f"{request_id}:device-slots"
        )

    def consumer_index(self, request_id: str) -> int:
        """Layer-gate slot the model must wait on for this request."""
        entry = self._active.get(request_id)
        if entry is None or entry.load_handle is None:
            return -1
        return entry.load_handle.producer_id

    def note_forward_complete(self, request_id: str) -> None:
        entry = self._active[request_id]
        self.pipeline.note_forward_complete(entry.transaction)

    def abort(self, request_id: str, *, reason: str) -> None:
        entry = self._active.get(request_id)
        if entry is None:
            return
        self.pipeline.abort(entry.transaction, reason=reason)

    def try_release(self, request_id: str, *, committed: bool) -> bool:
        """Resolve ownership once storage can no longer touch the staging.

        Returns ``False`` while an operation is still live; the caller retries
        on a later step rather than freeing memory the kernel may write to.
        """
        entry = self._active.get(request_id)
        if entry is None:
            return True
        if not self.pipeline.is_release_safe(entry.transaction):
            return False
        self.pipeline.resolve_ownership(
            entry.transaction,
            host_state=(OwnershipState.INSERTED if committed else OwnershipState.FREED),
            device_state=(
                OwnershipState.INSERTED if committed else OwnershipState.FREED
            ),
        )
        if committed:
            self.pipeline.commit(entry.transaction)
        del self._active[request_id]
        return True

    def quarantine(self, request_id: str) -> None:
        """Give up on reclaiming staging that an unterminated read still owns."""
        entry = self._active.pop(request_id, None)
        if entry is None:
            return
        self.pipeline.resolve_ownership(
            entry.transaction, host_state=OwnershipState.QUARANTINED
        )
        logger.error(
            "Quarantined host staging for request %s; storage never reported a "
            "terminal completion",
            request_id,
        )

    def write_through(
        self, *, page_keys: Sequence[str], host_indices: torch.Tensor
    ) -> list[bool]:
        """Publish freshly computed pages so a later hit can stream them."""
        plan = self.host_pool.get_layer_group_buffer_meta(
            host_indices,
            0,
            self.host_pool.layer_num,
            file_payload_offset=self.layout.k_offset,
            alignment=self.layout.alignment,
        )
        results = []
        for position, page_key in enumerate(page_keys):
            k_extent = plan.extents[2 * position]
            v_extent = plan.extents[2 * position + 1]
            results.append(
                self.writer.write_page(
                    page_key,
                    k_ptr=k_extent.logical_ptr,
                    v_ptr=v_extent.logical_ptr,
                )
            )
        return results

    def shutdown(self) -> None:
        self.backend.shutdown()

    def _submit_h2d_range(
        self, transaction: LayerwiseTransaction, layer_start: int, layer_end: int
    ) -> None:
        entry = self._active[transaction.transaction_id]
        if entry.load_handle is None:
            raise RuntimeError(
                f"request {transaction.transaction_id!r} reached H2D before its "
                "device slots were attached"
            )
        self.cache_controller.submit_streaming_load_range(
            entry.load_handle, layer_start=layer_start, layer_end=layer_end
        )
        entry.submitted_layers = layer_end

    def _finish_h2d_if_complete(self, entry: _ActiveTransaction) -> None:
        if entry.load_handle is None:
            return
        if entry.submitted_layers != self.host_pool.layer_num:
            return
        self.cache_controller.finish_streaming_load(entry.load_handle)
        entry.load_handle = None


def _storage_root(server_args: ServerArgs) -> str:
    extra = server_args.hicache_storage_backend_extra_config
    if isinstance(extra, dict) and extra.get("layerwise_root"):
        return str(extra["layerwise_root"])
    return envs.SGLANG_HICACHE_LAYERWISE_ROOT.get()
