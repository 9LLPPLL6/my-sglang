"""Unit tests for page-first-direct layer-group I/O metadata."""

import unittest

import torch

from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_host(
    *,
    page_num: int = 3,
    layer_num: int = 4,
    page_size: int = 2,
    head_num: int = 3,
    head_dim: int = 4,
    storage_offset: int = 0,
) -> MHATokenToKVPoolHost:
    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host.layout = "page_first_direct"
    host.page_num = page_num
    host.layer_num = layer_num
    host.page_size = page_size
    host.head_num = head_num
    host.head_dim = head_dim
    host.size = page_num * page_size
    host.dtype = torch.float32

    shape = (2, page_num, layer_num, page_size, head_num, head_dim)
    numel = 1
    for dim in shape:
        numel *= dim
    base_alignment = 64
    alignment_slack = base_alignment // host.dtype.itemsize
    raw = torch.empty(numel + storage_offset + alignment_slack, dtype=host.dtype)
    aligned_offset = ((-raw.data_ptr()) % base_alignment) // host.dtype.itemsize
    view_offset = aligned_offset + storage_offset
    host.kv_buffer = raw[view_offset : view_offset + numel].view(shape)
    # Keep the allocation alive explicitly; kv_buffer is a view but this also
    # makes the intentional storage offset visible in the fixture.
    host._test_raw_buffer = raw
    return host


class TestMHAHostLayerGroupMeta(CustomTestCase):
    def test_aligned_extents_use_tp_local_head_count(self):
        for local_head_num in (1, 3):
            with self.subTest(local_head_num=local_head_num):
                host = _make_host(head_num=local_head_num)
                host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
                plan = host.get_layer_group_buffer_meta(
                    host_indices,
                    1,
                    3,
                    file_payload_offset=32,
                    alignment=16,
                )

                layer_stride = (
                    host.page_size
                    * local_head_num
                    * host.head_dim
                    * host.dtype.itemsize
                )
                page_stride = host.layer_num * layer_stride
                logical_size = 2 * layer_stride

                self.assertEqual(
                    [extent.kv for extent in plan.extents], ["K", "V", "K", "V"]
                )
                self.assertEqual(
                    [extent.page_index for extent in plan.extents], [0, 0, 2, 2]
                )
                self.assertTrue(plan.direct_io_eligible)
                self.assertFalse(plan.bounce_required)
                self.assertEqual(plan.total_logical_size, 4 * logical_size)

                k0, v0, k2, v2 = plan.extents
                self.assertEqual(
                    k0.logical_ptr, host.k_buffer.data_ptr() + layer_stride
                )
                self.assertEqual(
                    v0.logical_ptr, host.v_buffer.data_ptr() + layer_stride
                )
                self.assertEqual(
                    k2.logical_ptr,
                    host.k_buffer.data_ptr() + 2 * page_stride + layer_stride,
                )
                self.assertEqual(
                    v2.logical_ptr,
                    host.v_buffer.data_ptr() + 2 * page_stride + layer_stride,
                )
                self.assertEqual(k0.logical_file_offset, 32 + layer_stride)
                self.assertEqual(
                    v0.logical_file_offset,
                    32 + page_stride + layer_stride,
                )
                for extent in plan.extents:
                    self.assertEqual(extent.logical_size, logical_size)
                    self.assertEqual(extent.layer_stride, layer_stride)
                    self.assertEqual(extent.page_stride, page_stride)
                    self.assertTrue(extent.base_aligned)
                    self.assertTrue(extent.buffer_offset_aligned)
                    self.assertTrue(extent.host_address_aligned)
                    self.assertTrue(extent.file_offset_aligned)
                    self.assertTrue(extent.length_aligned)
                    self.assertTrue(extent.direct_io_eligible)

    def test_unaligned_extents_describe_bounce_without_mutation(self):
        host = _make_host(
            layer_num=3,
            head_num=1,
            head_dim=3,
            storage_offset=1,
        )
        host.kv_buffer.copy_(
            torch.arange(host.kv_buffer.numel(), dtype=host.dtype).view_as(
                host.kv_buffer
            )
        )
        before = host.kv_buffer.clone()

        plan = host.get_layer_group_buffer_meta(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            1,
            2,
            file_payload_offset=16,
            alignment=64,
        )

        self.assertFalse(plan.direct_io_eligible)
        self.assertTrue(plan.bounce_required)
        self.assertTrue(torch.equal(host.kv_buffer, before))
        self.assertEqual(plan.extents[0].logical_file_offset, 40)
        self.assertEqual(plan.extents[0].covering_file_offset, 0)
        self.assertEqual(plan.extents[0].covering_size, 64)
        self.assertEqual(plan.extents[0].bounce_data_offset, 40)
        self.assertEqual(plan.extents[1].logical_file_offset, 112)
        self.assertEqual(plan.extents[1].covering_file_offset, 64)
        self.assertEqual(plan.extents[1].covering_size, 128)
        self.assertEqual(plan.extents[1].bounce_data_offset, 48)

        for extent in plan.extents:
            self.assertEqual(extent.logical_size, 24)
            self.assertTrue(extent.bounce_required)
            self.assertEqual(extent.covering_file_offset % 64, 0)
            self.assertEqual(extent.covering_size % 64, 0)
            self.assertLessEqual(
                extent.bounce_data_offset + extent.logical_size,
                extent.covering_size,
            )

    def test_rejects_invalid_layer_ranges_and_page_indices(self):
        host = _make_host()
        valid = torch.tensor([0, 1], dtype=torch.int64)

        invalid_ranges = [(-1, 1), (1, 1), (2, 1), (0, host.layer_num + 1)]
        for layer_start, layer_end in invalid_ranges:
            with self.subTest(layer_start=layer_start, layer_end=layer_end):
                with self.assertRaisesRegex(ValueError, "layer range"):
                    host.get_layer_group_buffer_meta(
                        valid, layer_start, layer_end, alignment=16
                    )

        invalid_indices = [
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
            torch.tensor([6, 7], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
        ]
        for indices in invalid_indices:
            with self.subTest(indices=indices.tolist()):
                with self.assertRaises((TypeError, ValueError)):
                    host.get_layer_group_buffer_meta(indices, 0, 1, alignment=16)

    def test_rejects_incompatible_layout_and_alignment_inputs(self):
        host = _make_host()
        indices = torch.tensor([0, 1], dtype=torch.int64)

        host.layout = "page_first"
        with self.assertRaisesRegex(ValueError, "page_first_direct"):
            host.get_layer_group_buffer_meta(indices, 0, 1)

        host.layout = "page_first_direct"
        with self.assertRaisesRegex(ValueError, "alignment"):
            host.get_layer_group_buffer_meta(indices, 0, 1, alignment=0)
        with self.assertRaisesRegex(ValueError, "file_payload_offset"):
            host.get_layer_group_buffer_meta(indices, 0, 1, file_payload_offset=-1)


if __name__ == "__main__":
    unittest.main()
