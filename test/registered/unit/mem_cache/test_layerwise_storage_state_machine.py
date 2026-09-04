import unittest

from sglang.srt.mem_cache.layerwise_storage import (
    CancelLevel,
    ExtentCompletionStatus,
    GroupState,
    InvalidStateTransition,
    InvalidStorageCompletion,
    KVPart,
    LayerGroupPlan,
    LayerwiseBackendCapabilities,
    LayerwiseReadPlan,
    LayerwiseStorageCompletion,
    LayerwiseStorageExtent,
    LayerwiseTransactionStateMachine,
    OwnershipResolutionError,
    OwnershipState,
    PrivateBufferOwnership,
    TransactionState,
    validate_group_against_capabilities,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _extent(*, extent_id: int, layer: int, io_offset: int = 0):
    return LayerwiseStorageExtent(
        extent_id=extent_id,
        storage_key=f"page-{extent_id}",
        kv_part=KVPart.KEY if extent_id % 2 == 0 else KVPart.VALUE,
        layer_start=layer,
        layer_end=layer + 1,
        io_offset=io_offset,
        io_nbytes=4096,
        payload_offset=0,
        payload_nbytes=4096,
        target_offset=extent_id * 4096,
    )


def _plan(*, group_count: int, extents_per_group: int = 1):
    groups = []
    next_extent_id = 0
    for group_id in range(group_count):
        extents = []
        for _ in range(extents_per_group):
            extents.append(_extent(extent_id=next_extent_id, layer=group_id))
            next_extent_id += 1
        groups.append(
            LayerGroupPlan(
                group_id=group_id,
                layer_start=group_id,
                layer_end=group_id + 1,
                extents=tuple(extents),
            )
        )
    return LayerwiseReadPlan(groups=tuple(groups))


def _completion(
    *,
    group_id: int,
    extent_id: int,
    status: ExtentCompletionStatus = ExtentCompletionStatus.SUCCEEDED,
    generation: int = 7,
    bytes_transferred: int = 4096,
):
    return LayerwiseStorageCompletion(
        transaction_id="tx",
        generation=generation,
        group_id=group_id,
        extent_id=extent_id,
        status=status,
        bytes_transferred=bytes_transferred,
    )


def _advance_to_device_ready(
    transaction: LayerwiseTransactionStateMachine,
    *,
    group_id: int,
    extent_id: int,
) -> None:
    group = transaction.group(group_id)
    group.submit()
    transaction.apply_completion(_completion(group_id=group_id, extent_id=extent_id))
    group.begin_consensus()
    group.mark_global_ready()
    group.mark_h2d_submitted()
    group.mark_device_ready()


class TestLayerwiseStorageStateMachine(CustomTestCase):
    def test_out_of_order_ready_groups_retire_in_plan_order(self):
        """A faster later group must not become consumable ahead of group 0.

        This guards the reorder boundary shared by TP collectives, H2D, and
        forward consumption: local storage completions may be unordered, but
        retirement must expose only a contiguous prefix of group order.
        """

        transaction = LayerwiseTransactionStateMachine(
            transaction_id="tx",
            generation=7,
            plan=_plan(group_count=2),
        )

        _advance_to_device_ready(transaction, group_id=1, extent_id=1)
        self.assertEqual(transaction.retire_ready_groups(), ())
        self.assertEqual(transaction.group(1).state, GroupState.DEVICE_READY)

        _advance_to_device_ready(transaction, group_id=0, extent_id=0)
        self.assertEqual(transaction.retire_ready_groups(), (0, 1))
        self.assertTrue(transaction.all_groups_retired)
        self.assertEqual(transaction.group(0).state, GroupState.RETIRED)
        self.assertEqual(transaction.group(1).state, GroupState.RETIRED)

    def test_cancel_request_does_not_release_inflight_target(self):
        """A cancel acknowledgement is not a terminal completion.

        An accepted-but-still-running backend operation may continue writing
        the target. The owner therefore cannot free or insert the target until
        every submitted extent reports a terminal completion; quarantine is
        the only safe early ownership transfer.
        """

        transaction = LayerwiseTransactionStateMachine(
            transaction_id="tx",
            generation=7,
            plan=_plan(group_count=1, extents_per_group=2),
        )
        group = transaction.group(0)
        group.submit()
        target = PrivateBufferOwnership(resource_id="host-staging")

        self.assertTrue(group.request_cancel())
        self.assertEqual(group.state, GroupState.SUBMITTED)
        self.assertFalse(group.storage_terminal)
        with self.assertRaises(OwnershipResolutionError):
            target.resolve(
                target=OwnershipState.FREED,
                operation_terminal=group.storage_terminal,
            )

        transaction.apply_completion(
            _completion(
                group_id=0,
                extent_id=0,
                status=ExtentCompletionStatus.CANCELLED,
                bytes_transferred=0,
            )
        )
        self.assertEqual(group.state, GroupState.SUBMITTED)
        self.assertFalse(group.storage_terminal)

        transaction.apply_completion(
            _completion(
                group_id=0,
                extent_id=1,
                status=ExtentCompletionStatus.CANCELLED,
                bytes_transferred=0,
            )
        )
        self.assertEqual(group.state, GroupState.CANCELLED)
        self.assertTrue(group.storage_terminal)
        target.resolve(
            target=OwnershipState.FREED,
            operation_terminal=group.storage_terminal,
        )
        self.assertEqual(target.state, OwnershipState.FREED)
        with self.assertRaises(OwnershipResolutionError):
            target.resolve(
                target=OwnershipState.INSERTED,
                operation_terminal=True,
            )

        quarantined = PrivateBufferOwnership(resource_id="other-staging")
        quarantined.resolve(
            target=OwnershipState.QUARANTINED,
            operation_terminal=False,
        )
        self.assertEqual(quarantined.state, OwnershipState.QUARANTINED)

    def test_stale_and_short_success_completions_are_rejected(self):
        """A prior-generation event or a short read cannot make a group ready.

        Both can otherwise publish partially overwritten host storage as a
        valid L2 hit while satisfying a superficial successful-I/O flag.
        """

        transaction = LayerwiseTransactionStateMachine(
            transaction_id="tx",
            generation=7,
            plan=_plan(group_count=1),
        )
        group = transaction.group(0)
        group.submit()

        with self.assertRaises(InvalidStorageCompletion):
            transaction.apply_completion(
                _completion(group_id=0, extent_id=0, generation=6)
            )
        with self.assertRaises(InvalidStorageCompletion):
            transaction.apply_completion(
                _completion(group_id=0, extent_id=0, bytes_transferred=2048)
            )

        self.assertEqual(group.state, GroupState.SUBMITTED)
        self.assertEqual(group.completion_count, 0)

    def test_lifecycle_rejects_h2d_before_consensus(self):
        """H2D must not observe local-only storage success before TP consensus."""

        transaction = LayerwiseTransactionStateMachine(
            transaction_id="tx",
            generation=7,
            plan=_plan(group_count=1),
        )
        group = transaction.group(0)
        group.submit()
        transaction.apply_completion(_completion(group_id=0, extent_id=0))

        with self.assertRaises(InvalidStateTransition):
            group.mark_h2d_submitted()
        self.assertEqual(group.state, GroupState.LOCAL_DONE)

    def test_transaction_cannot_publish_l2_before_forward_completion(self):
        """L2 publication cannot skip storage, admission, or forward phases."""

        transaction = LayerwiseTransactionStateMachine(
            transaction_id="tx",
            generation=7,
            plan=_plan(group_count=1),
        )

        with self.assertRaises(InvalidStateTransition):
            transaction.advance(TransactionState.L2_COMMITTED)
        self.assertEqual(transaction.state, TransactionState.NEW)

        transaction.advance(TransactionState.QUERYING_L3)
        transaction.advance(TransactionState.ABORTING)
        transaction.advance(TransactionState.ABORTED)
        with self.assertRaises(InvalidStateTransition):
            transaction.advance(TransactionState.DONE)

    def test_group_limits_apply_to_aligned_covering_io(self):
        """Backend limits apply to submitted covering ranges, not payload bytes.

        Direct I/O may transfer padding around compact payloads. Accounting the
        smaller payload would silently exceed the queue byte budget.
        """

        group = _plan(group_count=1, extents_per_group=2).groups[0]
        capabilities = LayerwiseBackendCapabilities(
            required_alignment=4096,
            supports_range_read=True,
            supports_direct_to_host=True,
            max_inflight_groups=2,
            max_inflight_extents=2,
            max_inflight_bytes=8192,
            max_iov=2,
            cancel_level=CancelLevel.BOUNDED_TERMINAL,
        )
        validate_group_against_capabilities(
            group=group,
            capabilities=capabilities,
        )

        too_small = LayerwiseBackendCapabilities(
            required_alignment=4096,
            supports_range_read=True,
            supports_direct_to_host=True,
            max_inflight_groups=2,
            max_inflight_extents=2,
            max_inflight_bytes=8191,
            max_iov=2,
            cancel_level=CancelLevel.BOUNDED_TERMINAL,
        )
        with self.assertRaises(ValueError):
            validate_group_against_capabilities(
                group=group,
                capabilities=too_small,
            )


if __name__ == "__main__":
    unittest.main()
