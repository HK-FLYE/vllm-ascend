# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""End-to-end CPU tests for the Step5 KV cache materialization path:

plan (step5 cache_config) -> allocate (attn_utils._allocate_kv_cache) ->
reshape (attn_utils._reshape_kv_cache_v2) -> bind (patch_bind_kv_cache).
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.v1.worker.utils as worker_utils
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_ascend.worker.v2.attn_utils as attn_utils_v2
from vllm_ascend.models.step5.cache_config import (
    get_step5_kv_cache_config_from_groups,
    get_step5_kv_cache_groups,
    get_step5_pool_bytes_per_block,
)
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache_to_layers

BLOCK = 128
MAIN_PAGE = 98304
SUMMARY_PAGE = 4096
RING_PAGE = 16384

L0 = "model.layers.0.self_attn"
L1 = "model.layers.1.self_attn"
L2 = "model.layers.2.self_attn"
RING = f"{L0}.csa_tail_ring"
SUMMARY0 = f"{L0}.sparse_summary_cache"
SUMMARY2 = f"{L2}.sparse_summary_cache"


def _specs():
    main = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=4, head_size=48, head_size_v=48, dtype=torch.bfloat16
    )
    sliding = SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=48,
        head_size_v=48,
        dtype=torch.bfloat16,
        sliding_window=4096,
    )
    summary = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=16, dtype=torch.float8_e4m3fn
    )
    ring = CircularBufferSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=64,
        head_size_v=0,
        dtype=torch.bfloat16,
        block_stride_alignment=1024,
    )
    return {
        L0: main,
        L2: main,
        L1: sliding,
        SUMMARY0: summary,
        SUMMARY2: summary,
        RING: ring,
    }


def _orig_groups(vllm_config, kv_cache_spec):
    full = [name for name, spec in kv_cache_spec.items() if type(spec) is FullAttentionSpec]
    sliding = [name for name, spec in kv_cache_spec.items() if isinstance(spec, SlidingWindowSpec)]
    groups = [KVCacheGroupSpec(full, UniformTypeKVCacheSpecs.from_specs({n: kv_cache_spec[n] for n in full}))]
    groups.append(KVCacheGroupSpec(sliding, UniformTypeKVCacheSpecs.from_specs({n: kv_cache_spec[n] for n in sliding})))
    return groups


def _plan(num_blocks: int = 3):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="step5"),
            hf_text_config=SimpleNamespace(model_type="step5"),
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=8192,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_blocks,
            prefix_cache_retention_interval=None,
            cache_dtype="auto",
        ),
        additional_config={},
        kv_transfer_config=None,
    )
    groups = get_step5_kv_cache_groups(vllm_config, _specs(), _orig_groups)
    config = get_step5_kv_cache_config_from_groups(
        SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=num_blocks,
                prefix_cache_retention_interval=None,
            )
        ),
        groups,
        10**9,
    )
    return vllm_config, groups, config


def _allocate(config, vllm_config, monkeypatch):
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: vllm_config)
    backend = SimpleNamespace(full_cls_name=lambda: "step5-cache-test")
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: backend, num_heads=1)
        for group in config.kv_cache_groups
        for name in group.layer_names
    }
    monkeypatch.setattr(attn_utils_v2, "get_layers_from_vllm_config", lambda *_: layers)
    return attn_utils_v2._allocate_kv_cache(config, {}, torch.device("cpu"))


def _reshape(config, raw, monkeypatch):
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: SimpleNamespace())
    monkeypatch.setattr(attn_utils_v2, "get_layers_from_vllm_config", lambda *_: {})
    backend = SimpleNamespace(
        get_kv_cache_shape=lambda num_blocks, block_size, heads, head_size, *a, **k: (
            2,
            num_blocks,
            block_size,
            heads,
            head_size,
        )
    )
    groups = [
        SimpleNamespace(
            kv_cache_group_id=index,
            kv_cache_spec=group.kv_cache_spec,
            layer_names=group.layer_names,
            backend=backend,
        )
        for index, group in enumerate(config.kv_cache_groups)
    ]
    return attn_utils_v2._reshape_kv_cache_v2(
        groups,
        raw,
        "auto",
        [BLOCK] * len(groups),
        {},
        config,
    )


def test_step5_allocation_uses_one_envelope_sized_backing(monkeypatch) -> None:
    vllm_config, groups, config = _plan(num_blocks=3)
    raw = _allocate(config, vllm_config, monkeypatch)

    # Every planned layer draws its view from the single envelope backing:
    # one distinct storage of num_blocks x envelope bytes.
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in raw.values()}
    assert set(storages.values()) == {
        config.num_blocks * get_step5_pool_bytes_per_block(groups)
    }


def test_step5_reshape_builds_typed_aux_views(monkeypatch) -> None:
    vllm_config, groups, config = _plan(num_blocks=3)
    raw = _allocate(config, vllm_config, monkeypatch)
    caches = _reshape(config, raw, monkeypatch)

    layer_specs = {
        name: spec for group in groups for name, spec in _layer_specs(group)
    }

    # Auxiliary layers get the typed page view: one page per block id.
    for aux, page_bytes, dtype in (
        (RING, RING_PAGE, torch.bfloat16),
        (SUMMARY0, SUMMARY_PAGE, torch.float8_e4m3fn),
        (SUMMARY2, SUMMARY_PAGE, torch.float8_e4m3fn),
    ):
        view = caches[aux]
        assert isinstance(view, torch.Tensor), aux
        assert view.dtype is dtype, aux
        assert view.element_size() * view.numel() == config.num_blocks * page_bytes, aux
        assert view.stride(0) * view.element_size() == page_bytes, aux
        assert view.shape[0] == config.num_blocks, aux

    # Main attention layers keep the Ascend per-layer (k, v) tuple layout
    # over one 96 KiB page per block id.
    for main in (L0, L2, L1):
        entry = caches[main]
        tensors = entry if isinstance(entry, tuple) else (entry,)
        assert all(isinstance(tensor, torch.Tensor) for tensor in tensors), main
        assert all(tensor.dtype is torch.bfloat16 for tensor in tensors), main
        assert sum(tensor.element_size() * tensor.numel() for tensor in tensors) == (
            config.num_blocks * MAIN_PAGE
        ), main


def _layer_specs(group):
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return list(spec.kv_cache_specs.items())
    return [(name, spec) for name in group.layer_names]


def test_bind_kv_cache_keeps_deepseek_v41_unpacking(monkeypatch) -> None:
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer

    monkeypatch.setattr(worker_utils, "share_replayssm_ring_trackers", lambda *args, **kwargs: None)

    class FakeV41Layer(DeepseekV41CacheLayer):
        def __init__(self):
            self.bound = None

        def bind_kv_cache(self, kv_cache):
            self.bound = kv_cache

    layer = FakeV41Layer()
    kv_cache = object()
    bind_kv_cache_to_layers({"model.layers.0.self_attn": kv_cache}, {"model.layers.0.self_attn": layer})
    assert layer.bound is kv_cache
