# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


class TestIndexerRegionScore(unittest.TestCase):
    def test_model_ids_out_contract(self):
        # Load the real model helper without unrelated model/backend imports.
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/step5/ops/indexer_region_score.py"
        spec = importlib.util.spec_from_file_location("step5_indexer_region_score", path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                query = torch.ones(2, 4, 12, 256, dtype=dtype)
                weights = torch.empty(2, 4, 12)
                key = torch.empty(3, 2, 16, 1, 256, dtype=torch.float8_e4m3fn)[:, 1]
                lengths = torch.tensor([32], dtype=torch.int32)
                table = torch.tensor([[0, 2]], dtype=torch.int32)
                cu = torch.tensor([0, 2], dtype=torch.int32)
                operator = MagicMock()

                def fill_ids(*args, selected_region_ids, **kwargs):
                    selected_region_ids.copy_(torch.arange(24, dtype=torch.int32).reshape(4, 2, 3))

                operator.ids_out.side_effect = fill_ids
                with patch.object(torch.ops._C_ascend, "npu_indexer_region_score", operator, create=True):
                    result = model_module.indexer_score_topk(
                        query,
                        weights,
                        key,
                        table,
                        cu,
                        lengths,
                        max_region_len=32,
                        topk=3,
                        region_size=4,
                        head_scale=0.5,
                    )
                operator.assert_not_called()
                operator.ids_out.assert_called_once()
                call = operator.ids_out.call_args
                forwarded_query, *forwarded = call.args
                self.assertEqual(forwarded_query.dtype, torch.float8_e4m3fn)
                torch.testing.assert_close(forwarded_query.float(), query.float())
                for expected, actual in zip((weights, key, lengths, table, cu), forwarded):
                    self.assertIs(actual, expected)
                ids = call.kwargs["selected_region_ids"]
                self.assertEqual(ids.shape, (4, 2, 3))
                self.assertEqual(ids.dtype, torch.int32)
                self.assertEqual(result.shape, (8, 3))
                self.assertEqual(result.data_ptr(), ids.data_ptr())
                torch.testing.assert_close(result, torch.arange(24, dtype=torch.int32).reshape(8, 3))


if __name__ == "__main__":
    unittest.main()
