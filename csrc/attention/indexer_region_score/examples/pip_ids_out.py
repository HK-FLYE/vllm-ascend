# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Call the installed standalone package on an explicitly selected NPU."""

import argparse

import torch
import torch_npu  # noqa: F401
import vllm_ascend_indexer as irs

TOKENS = 1016
GROUPS = 4
HEADS = 4
DIMENSION = 256
REGIONS = 639
TOP_K = 512
REGION_SIZE = 8
KV_LENGTH = 8192


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="available logical NPU index")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    query = torch.randn((TOKENS, GROUPS, HEADS, DIMENSION), dtype=torch.bfloat16, device=device)
    weights = torch.randn((TOKENS, GROUPS, HEADS), dtype=torch.bfloat16, device=device)
    key = torch.randn((REGIONS, 1, 1, DIMENSION), dtype=torch.bfloat16, device=device)
    lengths = torch.tensor([KV_LENGTH], dtype=torch.int32, device=device)
    table = torch.arange(REGIONS, dtype=torch.int32, device=device).unsqueeze(0)
    cu = torch.tensor([0, TOKENS], dtype=torch.int32, device=device)
    ids = torch.empty((GROUPS, TOKENS, TOP_K), dtype=torch.int32, device=device)
    with torch.inference_mode():
        irs.ids_out(
            query,
            weights,
            key,
            lengths,
            table,
            cu,
            max_region_len=REGIONS,
            top_k=TOP_K,
            region_size=REGION_SIZE,
            selected_region_ids=ids,
        )
    torch.npu.synchronize()
    print(f"IndexerRegionScore completed: shape={tuple(ids.shape)}, dtype={ids.dtype}, device={ids.device}")


if __name__ == "__main__":
    main()
