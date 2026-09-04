"""Ordering and failure-handling tests for the layerwise storage pipeline.

The pipeline's value is entirely in what it refuses to do early: publish a
group before every rank agreed on it, hand layer ``n+1`` to the H2D stream
before layer ``n``, read ahead before the admission group landed, or free
staging that an in-flight read can still write to.
"""

import unittest

from sglang.srt.mem_cache.layerwise_storage.consensus import (
    GroupConsensus,
    SingleRankConsensus,
)
from sglang.srt.mem_cache.layerwise_storage.pipeline import (
    LayerwiseStoragePipeline,
    PipelineConfig,
)
from sglang.srt.mem_cache.layerwise_storage.state_machine import (
    GroupState,
    OwnershipResolutionError,
    OwnershipState,
    TransactionState,
)
from sglang.srt.mem_cache.layerwise_storage.types import (
    CancelLevel,
    CancelRequestDisposition,
    CancelRequestResult,
    ExtentCompletionStatus,
    HandleTerminalStatus,
    LayerGroupPlan,
    LayerwiseBackendCapabilities,
    LayerwiseGroupTicket,
    LayerwiseReadHandle,
    LayerwiseReadPlan,
    LayerwiseStorageCompletion,
    LayerwiseStorageExtent,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EXTENT_NBYTES = 4096


def _make_plan(*, group_count: int, extents_per_group: int = 2) -> LayerwiseReadPlan:
    groups = []
    for group_id in range(group_count):
        extents = tuple(
            LayerwiseStorageExtent(
                extent_id=extent_id,
                storage_key=f"page{extent_id}",
                kv_part="key" if extent_id % 2 == 0 else "value",
                layer_start=group_id,
                layer_end=group_id + 1,
                io_offset=_EXTENT_NBYTES * (group_id + 1),
                io_nbytes=_EXTENT_NBYTES,
                payload_offset=0,
                payload_nbytes=_EXTENT_NBYTES,
                target_offset=_EXTENT_NBYTES * extent_id,
            )
            for extent_id in range(extents_per_group)
        )
        groups.append(
            LayerGroupPlan(
                group_id=group_id,
                layer_start=group_id,
                layer_end=group_id + 1,
                extents=extents,
            )
        )
    return LayerwiseReadPlan(groups=tuple(groups))


class _FakeBackend:
    """A backend whose completions are produced only when a test asks for them."""

    def __init__(self, *, cancel_level=CancelLevel.BOUNDED_TERMINAL):
        self.capabilities_value = LayerwiseBackendCapabilities(
            required_alignment=4096,
            supports_range_read=True,
            supports_direct_to_host=True,
            max_inflight_groups=8,
            max_inflight_extents=64,
            max_inflight_bytes=1 << 30,
            max_iov=64,
            cancel_level=cancel_level,
        )
        self.submitted: list[int] = []
        self.submitted_priorities: dict[int, int] = {}
        self.cancelled: list[int] = []
        self.closed = False
        self._plan = None
        self._handle = None
        self._pending: list[LayerwiseStorageCompletion] = []
        self._inflight_extents = 0

    def capabilities(self):
        return self.capabilities_value

    def begin_read(self, *, transaction_id, generation, plan, target):
        self._plan = plan
        self._handle = LayerwiseReadHandle(
            transaction_id=transaction_id, generation=generation
        )
        return self._handle

    def submit_group(self, *, handle, group, priority, deadline_s):
        self.submitted.append(group.group_id)
        self.submitted_priorities[group.group_id] = priority
        self._inflight_extents += len(group.extents)
        return LayerwiseGroupTicket(handle=handle, group_id=group.group_id)

    def poll(self, *, handle, max_completions=None):
        drained = tuple(self._pending)
        self._pending = []
        return drained

    def request_cancel(self, *, handle, group_ids):
        self.cancelled.extend(group_ids)
        return tuple(
            CancelRequestResult(
                group_id=group_id, disposition=CancelRequestDisposition.ACCEPTED
            )
            for group_id in group_ids
        )

    def poll_terminal(self, *, handle):
        if self._inflight_extents > 0:
            return HandleTerminalStatus.ACTIVE
        return HandleTerminalStatus.SUCCEEDED

    def close(self, *, handle):
        self.closed = True

    def complete_group(
        self, group_id: int, *, status=ExtentCompletionStatus.SUCCEEDED
    ) -> None:
        group = self._plan.groups[group_id]
        for extent in group.extents:
            self._pending.append(
                LayerwiseStorageCompletion(
                    transaction_id=self._handle.transaction_id,
                    generation=self._handle.generation,
                    group_id=group_id,
                    extent_id=extent.extent_id,
                    status=status,
                    bytes_transferred=(
                        extent.io_nbytes
                        if status is ExtentCompletionStatus.SUCCEEDED
                        else 0
                    ),
                    error=(
                        None
                        if status is ExtentCompletionStatus.SUCCEEDED
                        else "injected failure"
                    ),
                )
            )
            self._inflight_extents -= 1


class _DeferredConsensus(GroupConsensus):
    """Agreement that only resolves when a test decides it does."""

    def __init__(self):
        self.started: list[int] = []
        self.verdicts: dict[int, bool] = {}
        self.released: list[int] = []

    def begin(self, *, transaction_id, group_id, local_success):
        self.started.append(group_id)

    def poll(self, *, transaction_id, group_id):
        return self.verdicts.get(group_id)

    def release(self, *, transaction_id, group_id):
        self.released.append(group_id)


class _PipelineFixture:
    def __init__(self, *, group_count=4, read_ahead_groups=2, consensus=None):
        self.backend = _FakeBackend()
        self.consensus = consensus or SingleRankConsensus()
        self.h2d_ranges: list[tuple[int, int]] = []
        self.plan = _make_plan(group_count=group_count)
        self.pipeline = LayerwiseStoragePipeline(
            backend=self.backend,
            consensus=self.consensus,
            config=PipelineConfig(
                read_ahead_groups=read_ahead_groups,
                group_timeout_s=60.0,
                admission_budget_s=0.0,
            ),
            submit_h2d_range=lambda transaction, start, end: self.h2d_ranges.append(
                (start, end)
            ),
        )
        self.transaction = self.pipeline.begin(
            transaction_id="txn",
            generation=0,
            plan=self.plan,
            target=object(),
            host_resource_id="host-staging",
        )

    def admit(self):
        self.pipeline.note_device_allocated(
            self.transaction, device_resource_id="device-slots"
        )


class TestLayerwiseStoragePipeline(CustomTestCase):
    def test_only_group_zero_is_submitted_before_it_lands(self):
        fixture = _PipelineFixture()

        self.assertEqual(fixture.backend.submitted, [0])
        self.assertEqual(fixture.backend.submitted_priorities[0], 0)

        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(
            fixture.backend.submitted, [0], "read-ahead must wait for group 0"
        )

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(fixture.backend.submitted[:3], [0, 1, 2])
        self.assertGreater(fixture.backend.submitted_priorities[1], 0)

    def test_admission_waits_for_cross_rank_agreement_on_group_zero(self):
        consensus = _DeferredConsensus()
        fixture = _PipelineFixture(consensus=consensus)

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(consensus.started, [0])
        self.assertFalse(fixture.transaction.admission_ready)

        consensus.verdicts[0] = True
        fixture.pipeline.advance(fixture.transaction)
        self.assertTrue(fixture.transaction.admission_ready)
        self.assertIs(
            fixture.transaction.machine.state, TransactionState.ADMISSION_READY
        )

    def test_read_ahead_window_is_bounded_by_retirement(self):
        read_ahead_groups = 2
        fixture = _PipelineFixture(group_count=6, read_ahead_groups=read_ahead_groups)
        limit = read_ahead_groups + 1

        def unretired() -> int:
            return (
                len(fixture.backend.submitted)
                - fixture.transaction.machine.next_retire_group_id
            )

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(sorted(fixture.backend.submitted), [0, 1, 2])
        self.assertLessEqual(unretired(), limit)

        fixture.admit()
        for group_id in range(1, 5):
            fixture.pipeline.advance(fixture.transaction)
            self.assertLessEqual(
                unretired(),
                limit,
                "an unretired group still holds staging and must count "
                "against the window",
            )
            fixture.backend.complete_group(group_id)
        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(sorted(fixture.backend.submitted), [0, 1, 2, 3, 4, 5])

    def test_h2d_follows_plan_order_even_when_storage_does_not(self):
        fixture = _PipelineFixture(group_count=4, read_ahead_groups=4)

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        fixture.admit()
        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(fixture.h2d_ranges, [(0, 1)])

        fixture.backend.complete_group(2)
        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(
            fixture.h2d_ranges,
            [(0, 1)],
            "group 2 must not reach the device before group 1",
        )

        fixture.backend.complete_group(1)
        fixture.pipeline.advance(fixture.transaction)
        self.assertEqual(fixture.h2d_ranges, [(0, 1), (1, 2), (2, 3)])

    def test_a_peer_rank_failure_aborts_before_any_h2d(self):
        consensus = _DeferredConsensus()
        fixture = _PipelineFixture(consensus=consensus)

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        consensus.verdicts[0] = False
        fixture.pipeline.advance(fixture.transaction)

        self.assertTrue(fixture.transaction.aborted)
        self.assertEqual(fixture.h2d_ranges, [])
        self.assertIn("peer rank", fixture.transaction.error)

    def test_local_read_failure_aborts_and_cancels_only_submitted_groups(self):
        fixture = _PipelineFixture(group_count=4, read_ahead_groups=3)

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        submitted_before = set(fixture.backend.submitted)

        fixture.backend.complete_group(1, status=ExtentCompletionStatus.FAILED)
        fixture.pipeline.advance(fixture.transaction)

        self.assertTrue(fixture.transaction.aborted)
        self.assertTrue(set(fixture.backend.cancelled) <= submitted_before)
        never_submitted = [
            group
            for group in fixture.transaction.machine.groups
            if group.plan.group_id not in submitted_before
        ]
        for group in never_submitted:
            self.assertIs(group.state, GroupState.CANCELLED)

    def test_staging_is_not_freed_while_a_read_can_still_write_to_it(self):
        fixture = _PipelineFixture(group_count=2, read_ahead_groups=2)

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        fixture.pipeline.abort(fixture.transaction, reason="test abort")

        self.assertFalse(fixture.pipeline.is_release_safe(fixture.transaction))
        with self.assertRaises(OwnershipResolutionError):
            fixture.pipeline.resolve_ownership(
                fixture.transaction, host_state=OwnershipState.FREED
            )

        fixture.pipeline.resolve_ownership(
            fixture.transaction, host_state=OwnershipState.QUARANTINED
        )
        self.assertIs(
            fixture.transaction.host_ownership.state, OwnershipState.QUARANTINED
        )
        self.assertFalse(fixture.backend.closed)

    def test_successful_transaction_resolves_each_allocation_exactly_once(self):
        fixture = _PipelineFixture(group_count=2, read_ahead_groups=2)

        fixture.backend.complete_group(0)
        fixture.pipeline.advance(fixture.transaction)
        fixture.admit()
        fixture.backend.complete_group(1)
        fixture.pipeline.advance(fixture.transaction)

        self.assertEqual(fixture.h2d_ranges, [(0, 1), (1, 2)])
        self.assertIs(
            fixture.transaction.machine.state, TransactionState.STORAGE_COMPLETE
        )

        fixture.pipeline.note_forward_complete(fixture.transaction)
        fixture.pipeline.resolve_ownership(
            fixture.transaction,
            host_state=OwnershipState.INSERTED,
            device_state=OwnershipState.INSERTED,
        )
        fixture.pipeline.commit(fixture.transaction)

        self.assertIs(fixture.transaction.machine.state, TransactionState.DONE)
        self.assertTrue(fixture.backend.closed)
        with self.assertRaises(OwnershipResolutionError):
            fixture.transaction.host_ownership.resolve(
                target=OwnershipState.FREED, operation_terminal=True
            )

    def test_backend_without_terminal_cancellation_is_rejected(self):
        backend = _FakeBackend(cancel_level=CancelLevel.NEITHER)
        with self.assertRaisesRegex(ValueError, "terminal state"):
            LayerwiseStoragePipeline(
                backend=backend,
                consensus=SingleRankConsensus(),
                config=PipelineConfig(
                    read_ahead_groups=1,
                    group_timeout_s=1.0,
                    admission_budget_s=0.0,
                ),
                submit_h2d_range=lambda transaction, start, end: None,
            )

    def test_group_timeout_aborts_the_transaction(self):
        backend = _FakeBackend()
        pipeline = LayerwiseStoragePipeline(
            backend=backend,
            consensus=SingleRankConsensus(),
            config=PipelineConfig(
                read_ahead_groups=1,
                group_timeout_s=-1.0,
                admission_budget_s=0.0,
            ),
            submit_h2d_range=lambda transaction, start, end: None,
        )
        transaction = pipeline.begin(
            transaction_id="txn-timeout",
            generation=0,
            plan=_make_plan(group_count=2),
            target=object(),
            host_resource_id="host-staging",
        )

        pipeline.advance(transaction)
        self.assertTrue(transaction.aborted)
        self.assertIn("timeout", transaction.error)


if __name__ == "__main__":
    unittest.main()
