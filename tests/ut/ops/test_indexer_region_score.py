# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_wrapper():
    # Exercise the real wrapper without importing unrelated model backends.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/indexer_region_score.py"
    spec = importlib.util.spec_from_file_location("indexer_region_score_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.indexer_region_score


class TestIndexerRegionScore(unittest.TestCase):
    def setUp(self):
        self.call = load_wrapper()
        self.inputs = (
            torch.empty(2, 1, 4, 256),
            torch.empty(2, 1, 4),
            torch.empty(3, 2, 16, 1, 256)[:, 1],
            torch.tensor([32], dtype=torch.int32),
            torch.tensor([[0, 2]], dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int32),
        )

    def test_defaults_preserve_paged_view(self):
        result = (torch.empty(1, 2, 32), torch.empty(1, 2, 512, dtype=torch.int32))
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", create=True, return_value=result) as op:
            self.assertIs(self.call(*self.inputs, max_region_len=32), result)
        op.assert_called_once_with(*self.inputs, 32, 512, 8, 1.0, 1, 3, 1, None, None)
        self.assertIs(op.call_args.args[2], self.inputs[2])
        self.assertFalse(self.inputs[2].is_contiguous())

    def test_preallocated_outputs_and_nondefault_attributes(self):
        scores = torch.empty(1, 2, 48)
        ids = torch.empty(1, 2, 3, dtype=torch.int32)
        with patch.object(
            torch.ops._C_ascend, "npu_indexer_region_score", create=True, return_value=(scores, ids)
        ) as op:
            result = self.call(
                *self.inputs,
                max_region_len=29,
                top_k=3,
                region_size=4,
                head_scale=0.5,
                out=scores,
                selected_region_ids=ids,
            )
        op.assert_called_once_with(*self.inputs, 29, 3, 4, 0.5, 1, 3, 1, scores, ids)
        self.assertIs(result[0], scores)
        self.assertIs(result[1], ids)

    def test_grouped_heads_preserve_inputs_and_output_aliases(self):
        for groups, heads in ((4, 12), (8, 128)):
            with self.subTest(groups=groups, heads=heads):
                inputs = (torch.empty(2, groups, heads, 256), torch.empty(2, groups, heads), *self.inputs[2:])
                scores = torch.empty(groups, 2, 32)
                ids = torch.empty(groups, 2, 3, dtype=torch.int32)
                with patch.object(
                    torch.ops._C_ascend, "npu_indexer_region_score", create=True, return_value=(scores, ids)
                ) as op:
                    result = self.call(*inputs, max_region_len=32, top_k=3, out=scores, selected_region_ids=ids)
                op.assert_called_once_with(*inputs, 32, 3, 8, 1.0, 1, 3, 1, scores, ids)
                self.assertIs(op.call_args.args[0], inputs[0])
                self.assertIs(op.call_args.args[1], inputs[1])
                self.assertIs(op.call_args.args[2], self.inputs[2])
                self.assertIs(result[0], scores)
                self.assertIs(result[1], ids)

    def test_key_summary_keyword_and_explicit_modes(self):
        query, weights, key, lengths, table, cu = self.inputs
        result = (torch.empty(1, 2, 32), torch.empty(1, 2, 3, dtype=torch.int32))
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", create=True, return_value=result) as op:
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
                result,
            )
        op.assert_called_once_with(*self.inputs, 32, 3, 8, 1.0, 1, 3, 1, None, None)

    def test_noncontiguous_views_are_forwarded_without_materializing(self):
        query = torch.empty(2, 1, 4, 512)[..., ::2]
        weights = torch.empty(2, 1, 8)[..., ::2]
        scores = torch.empty(1, 2, 64)[..., ::2]
        ids = torch.empty(1, 2, 6, dtype=torch.int32)[..., ::2]
        inputs = (query, weights, *self.inputs[2:])
        with patch.object(
            torch.ops._C_ascend, "npu_indexer_region_score", create=True, return_value=(scores, ids)
        ) as op:
            actual = self.call(*inputs, max_region_len=32, top_k=3, out=scores, selected_region_ids=ids)
        for expected, forwarded in zip(inputs, op.call_args.args[:6]):
            self.assertIs(expected, forwarded)
        self.assertIs(actual[0], scores)
        self.assertIs(actual[1], ids)
        self.assertFalse(query.is_contiguous())
        self.assertFalse(weights.is_contiguous())
        self.assertFalse(scores.is_contiguous())
        self.assertFalse(ids.is_contiguous())

    def test_unsupported_modes_rejected_before_dispatch(self):
        with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", create=True) as op:
            for parameter, values in (
                ("cache_mode", (-1, 0, 2)),
                ("mask_mode", (-1, 0, 1, 2)),
                ("sort_mode", (-1, 0, 2)),
            ):
                for value in values:
                    with (
                        self.subTest(parameter=parameter, value=value),
                        self.assertRaisesRegex(ValueError, "cache_mode|mask_mode|sort_mode"),
                    ):
                        self.call(*self.inputs, max_region_len=32, **{parameter: value})
            op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
