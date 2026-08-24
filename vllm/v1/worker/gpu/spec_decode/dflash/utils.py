# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    if __import__("os").environ.get("DF2_DRAFT_BF16", "0") == "1":
        # get_model() passes draft_model_config to the LOADER, but the draft
        # model's __init__ reads get_current_vllm_config().model_config, which is
        # still the TARGET's -- so fc, mask_embedding and the DFlash2 candidate
        # selector would be built at the target's dtype and tear against bf16
        # weights. Make the draft's config the current one.
        draft_vllm_config = replace(draft_vllm_config, model_config=draft_model_config)
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
