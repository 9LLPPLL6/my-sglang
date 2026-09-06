"""Release-path tests for the layerwise streaming bridge.

Both cases guard failures that leave no trace at the point they happen:

* a transaction that ends without clearing the layer gate's pump leaves a stale
  callback on the shared counter, and the *next* batch hangs inside a wait that
  drives a transaction which no longer exists;
* a transaction dropped before admission must hand its private host staging
  back exactly once, or the host pool leaks a whole prefix per dropped fetch.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.layerwise_storage.radix_bridge import LayerwiseRadixBridge
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_PAGES = 3


class _FakeTransaction:
    def __init__(self):
        self.admission_ready = False
        self.aborted = False
        self.error = None


class _FakeController:
    def __init__(self):
        self.transaction = _FakeTransaction()
        self.aborted_with = None
        self.released = []
        self.release_ok = True
        self.quarantined = []
        self.forward_complete = []

    def begin(self, *, request_id, page_keys, host_indices):
        return self.transaction

    def poll(self):
        pass

    def abort(self, req_id, *, reason):
        self.aborted_with = reason
        self.transaction.aborted = True

    def try_release(self, req_id, *, committed):
        self.released.append((req_id, committed))
        return self.release_ok

    def quarantine(self, req_id):
        self.quarantined.append(req_id)

    def note_forward_complete(self, req_id):
        self.forward_complete.append(req_id)


def _fake_cache():
    counter = SimpleNamespace(stream_pump="stale-callback-from-a-previous-batch")
    cache_controller = SimpleNamespace(
        layer_done_counter=counter,
        append_host_mem_release=mock.Mock(),
        prefetch_tokens_occupied=1000,
        mem_pool_host=SimpleNamespace(free=mock.Mock()),
    )
    return SimpleNamespace(
        page_size=_PAGE_SIZE,
        ongoing_prefetch={},
        cache_controller=cache_controller,
        prefetch_loaded_tokens_by_reqid={},
        dec_host_lock_ref=mock.Mock(),
        tree_core=SimpleNamespace(insert_host=mock.Mock()),
        _apply_cache_actions=mock.Mock(),
    )


def _stage(bridge, cache, controller, *, req_id="req-0"):
    num_tokens = _PAGES * _PAGE_SIZE
    prefetch_key = mock.MagicMock()
    prefetch_key.__getitem__.return_value = SimpleNamespace(
        token_ids=list(range(num_tokens))
    )
    prefetch_key.extra_key = None
    cache.ongoing_prefetch[req_id] = SimpleNamespace(
        prefetch_key=prefetch_key,
        anchor_node_id=7,
        anchor_lock_params=SimpleNamespace(),
    )
    operation = SimpleNamespace(
        request_id=req_id,
        host_indices=torch.arange(num_tokens, dtype=torch.int64),
        hash_value=[f"page{index}" for index in range(_PAGES)],
        id=11,
    )
    bridge.set_prefix_ctx(req_id, [1, 2, 3])
    assert bridge.start(operation)
    return req_id, num_tokens


class TestLayerwiseRadixBridge(CustomTestCase):
    def setUp(self):
        self.cache = _fake_cache()
        self.controller = _FakeController()
        self.bridge = LayerwiseRadixBridge(cache=self.cache, controller=self.controller)

    def test_drop_before_admission_returns_the_staging_once(self):
        req_id, num_tokens = _stage(self.bridge, self.cache, self.controller)
        self.controller.transaction.aborted = True
        self.controller.transaction.error = "storage read failed"

        self.assertTrue(self.bridge.check_progress(req_id))

        release = self.cache.cache_controller.append_host_mem_release
        self.assertEqual(release.call_count, 1)
        self.assertEqual(len(release.call_args[0][0]), num_tokens)
        self.cache.tree_core.insert_host.assert_not_called()
        self.assertEqual(
            self.cache.cache_controller.prefetch_tokens_occupied, 1000 - num_tokens
        )
        self.assertNotIn(req_id, self.cache.ongoing_prefetch)
        self.assertFalse(self.bridge.has_staged(req_id))
        self.assertEqual(self.cache.prefetch_loaded_tokens_by_reqid[req_id], 0)

        # A second progress check must not double-release.
        self.assertTrue(self.bridge.check_progress(req_id))
        self.assertEqual(release.call_count, 1)

    def test_finishing_a_stream_clears_the_shared_layer_gate_pump(self):
        req_id, _ = _stage(self.bridge, self.cache, self.controller)
        self.controller.transaction.admission_ready = True
        staged = self.bridge._staged[req_id]
        staged.admitted = True
        self.bridge._streaming_req_id = req_id
        self.bridge._streaming_producer_index = 2
        self.cache.cache_controller.layer_done_counter.stream_pump = self.bridge._pump
        self.assertEqual(self.bridge.streaming_consumer_index(), 2)

        self.assertTrue(self.bridge.try_finish_load_back(-(staged.operation_id) - 1))

        self.assertIsNone(self.cache.cache_controller.layer_done_counter.stream_pump)
        self.assertEqual(self.bridge.streaming_consumer_index(), -1)
        self.assertFalse(self.bridge.busy())
        self.assertEqual(self.controller.forward_complete, [req_id])
        self.cache.tree_core.insert_host.assert_called_once()

    def test_staging_a_backend_still_holds_is_quarantined_not_freed(self):
        req_id, _ = _stage(self.bridge, self.cache, self.controller)
        staged = self.bridge._staged[req_id]
        staged.admitted = True
        self.bridge._streaming_req_id = req_id
        self.controller.release_ok = False

        self.assertTrue(self.bridge.try_finish_load_back(-(staged.operation_id) - 1))

        self.assertEqual(self.controller.quarantined, [req_id])
        self.cache.tree_core.insert_host.assert_not_called()

    def test_only_one_transaction_streams_at_a_time(self):
        _stage(self.bridge, self.cache, self.controller)
        second = SimpleNamespace(
            request_id="req-1",
            host_indices=torch.arange(_PAGES * _PAGE_SIZE, dtype=torch.int64),
            hash_value=[f"page{index}" for index in range(_PAGES)],
            id=12,
        )
        self.cache.ongoing_prefetch["req-1"] = self.cache.ongoing_prefetch["req-0"]
        self.bridge.set_prefix_ctx("req-1", [1, 2, 3])

        self.assertFalse(self.bridge.start(second))


if __name__ == "__main__":
    unittest.main()
