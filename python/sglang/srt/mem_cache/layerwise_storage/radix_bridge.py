"""Connects the layerwise storage pipeline to ``UnifiedRadixCache``.

The default L3 path reads a whole prefix into host memory before the request is
admitted, so the read sits in front of TTFT with nothing to hide behind. This
bridge admits a request as soon as the *first layer group* has landed, and lets
the remaining groups stream in behind the forward pass: while the GPU computes
layer L, storage is fetching layer L+1.

Where the pieces live:

* the existing prefetch machinery still does the page-hash query, the
  cross-rank hit-length agreement and the host staging allocation, all in the
  background before admission;
* this bridge takes over from there, driving the read on the scheduler thread
  through :class:`LayerwiseStoragePipeline` instead of handing the operation to
  the blocking I/O thread;
* the host staging stays private until the transaction is complete, so no other
  request can match a prefix whose later layers are still in flight.

The layer gate is what makes early admission safe: the model blocks on layer L
until its H2D is submitted, and the wait itself pumps this pipeline (see
``LayerLoadingEvent.wait``).
"""

from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InitLoadBackResult,
    InsertParams,
    LoadBackOwnership,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.layerwise_storage.controller import (
        LayerwiseStorageController,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class LayerwiseStreamError(RuntimeError):
    """A streamed transaction failed after its request was already admitted.

    V1 fails loudly here rather than continuing: the device slots are published
    in the radix tree and partly filled, so silently proceeding would serve KV
    that was never read. Multi-request poison/replay is the follow-up.
    """


class _Staged:
    """One request's private staging between admission and completion."""

    def __init__(
        self,
        *,
        req_id: str,
        transaction,
        host_indices: torch.Tensor,
        page_keys: list[str],
        fetched_key,
        key_tokens: list[int],
        extra_key,
        matched_len: int,
        num_tokens: int,
        operation_id: int,
        anchor_node_id,
        anchor_lock_params,
    ):
        self.req_id = req_id
        self.transaction = transaction
        self.host_indices = host_indices
        self.page_keys = page_keys
        self.fetched_key = fetched_key
        self.key_tokens = key_tokens
        self.extra_key = extra_key
        self.matched_len = matched_len
        self.num_tokens = num_tokens
        self.operation_id = operation_id
        self.anchor_node_id = anchor_node_id
        self.anchor_lock_params = anchor_lock_params
        self.admitted = False


class LayerwiseRadixBridge:
    """Scheduler-side glue; owns no cache state of its own."""

    def __init__(
        self,
        *,
        cache: UnifiedRadixCache,
        controller: LayerwiseStorageController,
    ):
        self._cache = cache
        self.controller = controller
        self._staged: dict[str, _Staged] = {}
        # At most one transaction may be streaming at a time in V1: a second
        # would need its own producer slot in the layer gate, and the batch
        # carries only one consumer index.
        self._streaming_req_id: Optional[str] = None
        self._streaming_producer_index = -1
        # Device prefix matched when the fetch was enqueued; needed to rebuild
        # the full-span tree key at admission.
        self._prefix_ctx: dict[str, list[int]] = {}

    @property
    def page_size(self) -> int:
        return self._cache.page_size

    def has_staged(self, req_id: str) -> bool:
        return req_id in self._staged

    def busy(self) -> bool:
        return self._streaming_req_id is not None

    def set_prefix_ctx(self, req_id: str, matched_prefix_tokens) -> None:
        self._prefix_ctx[req_id] = list(matched_prefix_tokens or [])

    def start(self, operation) -> bool:
        """Take over a storage hit whose host staging is already allocated.

        Returns ``False`` to let the caller fall back to the ordinary blocking
        read; the request then behaves exactly as it does today.
        """
        req_id = operation.request_id
        info = self._cache.ongoing_prefetch.get(req_id)
        matched_prefix_tokens = self._prefix_ctx.get(req_id)
        if info is None or operation.host_indices is None:
            return False
        if matched_prefix_tokens is None:
            return False
        if self.busy() or self._staged:
            # One streaming transaction at a time; the rest use the old path.
            return False

        page_keys = list(operation.hash_value)
        num_tokens = len(page_keys) * self.page_size
        if num_tokens == 0 or len(operation.host_indices) < num_tokens:
            return False
        host_indices = operation.host_indices[:num_tokens]

        try:
            transaction = self.controller.begin(
                request_id=req_id,
                page_keys=page_keys,
                host_indices=host_indices,
            )
        except Exception as error:
            logger.warning("Layerwise streaming declined for req=%s: %s", req_id, error)
            return False

        self._staged[req_id] = _Staged(
            req_id=req_id,
            transaction=transaction,
            host_indices=host_indices,
            page_keys=page_keys,
            fetched_key=info.prefetch_key[:num_tokens],
            key_tokens=list(matched_prefix_tokens or [])
            + list(info.prefetch_key[:num_tokens].token_ids),
            extra_key=info.prefetch_key.extra_key,
            matched_len=len(matched_prefix_tokens or []),
            num_tokens=num_tokens,
            operation_id=operation.id,
            anchor_node_id=info.anchor_node_id,
            anchor_lock_params=info.anchor_lock_params,
        )
        return True

    def check_progress(self, req_id: str) -> bool:
        """Drive the pipeline; True once admission can be decided.

        Unlike the blocking path this returns as soon as the first layer group
        is agreed across ranks — the rest of the prefix is still in flight and
        will stream in behind the forward.
        """
        staged = self._staged.get(req_id)
        if staged is None:
            return True
        self.controller.poll()
        transaction = staged.transaction
        if transaction.aborted:
            self._discard(staged, reason=transaction.error or "storage read failed")
            return True
        return transaction.admission_ready

    def staged_tokens(self, req_id: str) -> int:
        """Tokens the admission step will splice in, surfaced as host_hit_length."""
        staged = self._staged.get(req_id)
        if staged is None or not staged.transaction.admission_ready:
            return 0
        return staged.num_tokens

    def init_load_back(self, params: InitLoadBackParams) -> InitLoadBackResult:
        """Consume the staged transaction at admission.

        Mirrors the buffer-mode consumption: validate that the span still
        splices, allocate device slots, publish them, and open the H2D session.
        The difference is that only the first layer group is in host memory
        here; the rest arrives through the pump while the model runs.
        """
        cache = self._cache
        req = params.req
        assert req is not None
        unchanged = InitLoadBackResult(
            device_indices=cache.tree_core.empty_match_result.device_indices,
            last_node=req.last_node,
            ownership=LoadBackOwnership.TREE,
        )
        staged = self._staged.get(req.rid)
        if staged is None:
            return unchanged
        if not staged.transaction.admission_ready:
            return unchanged

        if len(req.prefix_indices) != staged.matched_len:
            self._discard(staged, reason="device prefix moved before admission")
            return unchanged

        key = RadixKey(
            array("q", staged.key_tokens),
            extra_key=staged.extra_key,
            is_bigram=cache.tree_core.is_eagle,
        ).page_aligned(self.page_size)
        span_end = staged.matched_len + staged.num_tokens

        live = cache.match_prefix(MatchPrefixParams(key=key))
        if (
            len(live.device_indices) != staged.matched_len
            or live.full_kv_hit_length != staged.matched_len
        ):
            self._discard(staged, reason="another request published this prefix")
            return unchanged

        device_indices = self._alloc_device(staged.num_tokens)
        if device_indices is None:
            self._discard(staged, reason="no device slots for the streamed prefix")
            return unchanged

        load_back_id = -(staged.operation_id) - 1
        self.controller.attach_device(
            request_id=staged.req_id,
            device_indices=device_indices,
            node_ids=[load_back_id],
        )
        self._streaming_req_id = staged.req_id
        self._streaming_producer_index = self.controller.consumer_index(staged.req_id)
        cache.cache_controller.layer_done_counter.stream_pump = self._pump
        staged.admitted = True

        cache.insert(
            InsertParams(
                key=key,
                value=torch.cat([req.prefix_indices, device_indices]),
                prev_prefix_len=staged.matched_len,
            )
        )
        match = cache.match_prefix(MatchPrefixParams(key=key))
        canonical = match.device_indices[staged.matched_len : span_end]
        if len(match.device_indices) < span_end or not torch.equal(
            canonical, device_indices
        ):
            raise LayerwiseStreamError(
                f"layerwise load-back ownership violation req={staged.req_id}: "
                "the insert freed or replaced slots the in-flight streaming H2D "
                "still targets"
            )
        return InitLoadBackResult(
            device_indices=canonical,
            last_node=match.last_device_node,
            ownership=LoadBackOwnership.TREE,
        )

    def streaming_consumer_index(self) -> int:
        """Producer slot the admitting batch must gate on, or -1."""
        return self._streaming_producer_index

    def try_finish_load_back(self, ack_id: int) -> bool:
        """Claim the streaming ack and release the private staging."""
        staged = next(
            (
                candidate
                for candidate in self._staged.values()
                if candidate.admitted and -(candidate.operation_id) - 1 == ack_id
            ),
            None,
        )
        if staged is None:
            return False
        self.controller.note_forward_complete(staged.req_id)
        self._release(staged, insert_host=True)
        return True

    def revoke(self, req_id: str) -> None:
        staged = self._staged.get(req_id)
        if staged is None:
            return
        self._discard(staged, reason="request aborted")

    def _pump(self) -> None:
        """Advance the streaming transaction from inside the model's layer wait."""
        req_id = self._streaming_req_id
        if req_id is None:
            return
        staged = self._staged.get(req_id)
        if staged is None:
            return
        self.controller.poll()
        if staged.transaction.aborted:
            # The gate is blocking the forward; a silent return would hang it.
            raise LayerwiseStreamError(
                f"layerwise storage read failed mid-forward for req={req_id}: "
                f"{staged.transaction.error}"
            )

    def _alloc_device(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Evict before allocating, the way an ordinary load-back does."""
        cache = self._cache
        if cache._component_available_size(ComponentType.FULL) < num_tokens:
            needed = num_tokens - cache._component_available_size(ComponentType.FULL)
            cache.evict_for_alloc(EvictParams(num_tokens=needed))
            if cache._component_available_size(ComponentType.FULL) < num_tokens:
                return None
        return cache.token_to_kv_pool_allocator.alloc(num_tokens)

    def _discard(self, staged: _Staged, *, reason: str) -> None:
        """Give up on a transaction that was never admitted."""
        if staged.admitted:
            raise LayerwiseStreamError(
                f"cannot discard an admitted transaction req={staged.req_id}: {reason}"
            )
        logger.info(
            "Layerwise streaming dropped req=%s tokens=%d reason=%s",
            staged.req_id,
            staged.num_tokens,
            reason,
        )
        self.controller.abort(staged.req_id, reason=reason)
        self._release(staged, insert_host=False)

    def _release(self, staged: _Staged, *, insert_host: bool) -> None:
        """Return every private allocation exactly once.

        Host staging is only freed once the backend confirms no read can still
        write to it; until then the pages are quarantined rather than handed
        back to the pool.
        """
        cache = self._cache
        released = self.controller.try_release(staged.req_id, committed=insert_host)
        if not released:
            self.controller.quarantine(staged.req_id)
            insert_host = False

        if insert_host and released:
            self._publish_host(staged)
        else:
            cache.cache_controller.append_host_mem_release(staged.host_indices)

        if staged.anchor_lock_params is not None:
            cache.dec_host_lock_ref(staged.anchor_node_id, staged.anchor_lock_params)
        cache.ongoing_prefetch.pop(staged.req_id, None)
        cache.cache_controller.prefetch_tokens_occupied -= staged.num_tokens
        cache.prefetch_loaded_tokens_by_reqid[staged.req_id] = (
            staged.num_tokens if insert_host else 0
        )
        self._staged.pop(staged.req_id, None)
        self._prefix_ctx.pop(staged.req_id, None)
        if self._streaming_req_id == staged.req_id:
            self._streaming_req_id = None
            self._streaming_producer_index = -1
            cache.cache_controller.layer_done_counter.stream_pump = None

    def _publish_host(self, staged: _Staged) -> None:
        """Make the completed staging an ordinary L2 entry.

        Only reached once every group is agreed and the forward consumed it, so
        a later request matching this prefix gets complete KV.
        """
        cache = self._cache
        insert_result = cache.tree_core.insert_host(
            staged.anchor_node_id,
            staged.fetched_key,
            staged.host_indices,
            staged.page_keys,
        )
        cache._apply_cache_actions(insert_result.cache_actions)
        if insert_result.host_insert_dropped:
            cache.cache_controller.append_host_mem_release(staged.host_indices)
            return
        cache.cache_controller.mem_pool_host.free(
            staged.host_indices[: insert_result.prefix_len]
        )
