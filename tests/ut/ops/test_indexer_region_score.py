# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


def load_wrapper():
    # Exercise the real wrapper without importing unrelated model backends.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/indexer_region_score.py"
    spec = importlib.util.spec_from_file_location("indexer_region_score_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIndexerRegionScore(unittest.TestCase):
    def setUp(self):
        self.wrapper = load_wrapper()
        self.call = self.wrapper.indexer_region_score
        self.inputs = (
            torch.empty(2, 1, 4, 256),
            torch.empty(2, 1, 4),
            torch.empty(3, 2, 16, 1, 256)[:, 1],
            torch.tensor([32], dtype=torch.int32),
            torch.tensor([[0, 2]], dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int32),
        )
        self.operator = MagicMock()

    def test_defaults_preserve_paged_view_and_return_ids(self):
        ids = torch.empty(1, 2, 512, dtype=torch.int32)
        self.operator.default.return_value = ids
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            self.assertIs(self.call(*self.inputs, max_region_len=32), ids)
        self.operator.default.assert_called_once_with(*self.inputs, 32, 512, 8, 1.0, 1, 3, 1)
        self.assertIs(self.operator.default.call_args.args[2], self.inputs[2])
        self.assertFalse(self.inputs[2].is_contiguous())
        self.operator.out.assert_not_called()
        self.assertFalse(hasattr(self.wrapper, "indexer_region_score_topk"))

    def test_explicit_none_score_uses_the_default_ids_overload(self):
        ids = torch.empty(1, 2, 3, dtype=torch.int32)
        self.operator.default.return_value = ids
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            self.assertIs(self.call(*self.inputs, max_region_len=29, top_k=3, out=None), ids)
        self.operator.default.assert_called_once_with(*self.inputs, 29, 3, 8, 1.0, 1, 3, 1)

    def test_preallocated_outputs_and_nondefault_attributes(self):
        scores = torch.empty(1, 2, 48)
        ids = torch.empty(1, 2, 3, dtype=torch.int32)
        self.operator.out.return_value = None
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            result = self.call(
                *self.inputs,
                max_region_len=29,
                top_k=3,
                region_size=4,
                head_scale=0.5,
                out=scores,
                selected_region_ids=ids,
            )
        self.operator.out.assert_called_once_with(
            *self.inputs, 29, 3, 4, 0.5, 1, 3, 1, out=scores, selected_region_ids=ids
        )
        self.assertIs(result[0], scores)
        self.assertIs(result[1], ids)

    def test_score_output_allocates_missing_ids_and_returns_both_buffers(self):
        scores = torch.empty(1, 2, 48)
        self.operator.out.return_value = None
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            result = self.call(*self.inputs, max_region_len=29, top_k=3, out=scores)
        self.assertIs(result[0], scores)
        self.assertEqual(result[1].shape, (1, 2, 3))
        self.assertEqual(result[1].dtype, torch.int32)
        self.assertIs(self.operator.out.call_args.kwargs["selected_region_ids"], result[1])

    def test_grouped_heads_preserve_inputs_and_output_aliases(self):
        for groups, heads in ((4, 12), (8, 128)):
            with self.subTest(groups=groups, heads=heads):
                inputs = (torch.empty(2, groups, heads, 256), torch.empty(2, groups, heads), *self.inputs[2:])
                scores = torch.empty(groups, 2, 32)
                ids = torch.empty(groups, 2, 3, dtype=torch.int32)
                with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
                    result = self.call(*inputs, max_region_len=32, top_k=3, out=scores, selected_region_ids=ids)
                self.operator.out.assert_called_with(
                    *inputs, 32, 3, 8, 1.0, 1, 3, 1, out=scores, selected_region_ids=ids
                )
                for expected, forwarded in zip(inputs, self.operator.out.call_args.args[:6]):
                    self.assertIs(expected, forwarded)
                self.assertIs(result[0], scores)
                self.assertIs(result[1], ids)

    def test_key_summary_keyword_and_explicit_modes(self):
        query, weights, key, lengths, table, cu = self.inputs
        ids = torch.empty(1, 2, 3, dtype=torch.int32)
        self.operator.default.return_value = ids
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            self.assertIs(
                self.call(
                    query=query,
                    weights=weights,
                    key_summary=key,
                    actual_kv_lengths=lengths,
                    block_table=table,
                    cu_seq_lens=cu,
                    max_region_len=32,
                    top_k=3,
                    cache_mode=1,
                    mask_mode=3,
                    sort_mode=1,
                ),
                ids,
            )
        self.operator.default.assert_called_once_with(*self.inputs, 32, 3, 8, 1.0, 1, 3, 1)

    def test_noncontiguous_full_score_views_are_forwarded_without_materializing(self):
        query = torch.empty(2, 1, 4, 512)[..., ::2]
        weights = torch.empty(2, 1, 8)[..., ::2]
        scores = torch.empty(1, 2, 64)[..., ::2]
        ids = torch.empty(1, 2, 6, dtype=torch.int32)[..., ::2]
        inputs = (query, weights, *self.inputs[2:])
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            actual = self.call(*inputs, max_region_len=32, top_k=3, out=scores, selected_region_ids=ids)
        for expected, forwarded in zip(inputs, self.operator.out.call_args.args[:6]):
            self.assertIs(expected, forwarded)
        self.assertIs(actual[0], scores)
        self.assertIs(actual[1], ids)
        self.assertTrue(all(not tensor.is_contiguous() for tensor in (query, weights, scores, ids)))

    def test_ids_only_reuses_strided_caller_buffer_after_void_dispatch(self):
        ids = torch.empty(1, 2, 6, dtype=torch.int32)[..., ::2]
        self.operator.ids_out.return_value = None
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            actual = self.call(*self.inputs, max_region_len=29, top_k=3, selected_region_ids=ids)
        self.operator.ids_out.assert_called_once_with(*self.inputs, 29, 3, 8, 1.0, 1, 3, 1, selected_region_ids=ids)
        self.operator.default.assert_not_called()
        self.assertIs(actual, ids)

    def test_compiled_score_copies_only_logical_prefix_into_caller_views(self):
        score_storage = torch.full((1, 2, 96), -313.0)
        scores = score_storage[..., 1::2]
        ids_storage = torch.full((1, 2, 6), -313, dtype=torch.int32)
        ids = ids_storage[..., ::2]
        computed_score = torch.full((1, 2, 32), 0.5)
        computed_ids = torch.arange(6, dtype=torch.int32).view(1, 2, 3)
        self.operator.score.return_value = (computed_score, computed_ids)
        with (
            patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True),
            patch.object(torch.compiler, "is_compiling", return_value=True),
        ):
            result = self.call(*self.inputs, max_region_len=32, top_k=3, out=scores, selected_region_ids=ids)
        self.operator.score.assert_called_once_with(*self.inputs, 32, 3, 8, 1.0, 1, 3, 1)
        self.operator.out.assert_not_called()
        self.assertIs(result[0], scores)
        self.assertIs(result[1], ids)
        torch.testing.assert_close(scores[..., :32], computed_score)
        torch.testing.assert_close(ids, computed_ids)
        self.assertTrue(torch.all(scores[..., 32:] == -313))
        self.assertTrue(torch.all(score_storage[..., ::2] == -313))
        self.assertTrue(torch.all(ids_storage[..., 1::2] == -313))

    def test_invalid_output_metadata_is_rejected_before_dispatch(self):
        for outputs in (
            {"out": torch.empty(1, 2, 31)},
            {"out": torch.empty(1, 2, 32, dtype=torch.bfloat16)},
            {"out": torch.empty(1, 2, 32, device="meta")},
            {"selected_region_ids": torch.empty(1, 2, 3)},
            {"selected_region_ids": torch.empty(1, 2, 4, dtype=torch.int32)},
            {"selected_region_ids": torch.empty(1, 2, 3, dtype=torch.int32, device="meta")},
        ):
            with (
                self.subTest(outputs=outputs),
                patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True),
                self.assertRaises(RuntimeError),
            ):
                self.call(*self.inputs, max_region_len=32, top_k=3, **outputs)
        self.operator.default.assert_not_called()
        self.operator.ids_out.assert_not_called()
        self.operator.out.assert_not_called()
        self.operator.score.assert_not_called()

    def test_unsupported_modes_rejected_before_dispatch(self):
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", self.operator, create=True):
            for parameter, values in (
                ("cache_mode", (-1, 0, 2)),
                ("mask_mode", (-1, 0, 1, 2)),
                ("sort_mode", (-1, 0, 2)),
            ):
                for value in values:
                    with self.subTest(parameter=parameter, value=value), self.assertRaisesRegex(ValueError, parameter):
                        self.call(*self.inputs, max_region_len=32, **{parameter: value})
            self.operator.assert_not_called()
            self.operator.default.assert_not_called()
            self.operator.ids_out.assert_not_called()
            self.operator.out.assert_not_called()


if __name__ == "__main__":
    unittest.main()
