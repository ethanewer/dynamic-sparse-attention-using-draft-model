from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from .llama_util import update_llama_model_for_ssa
from .qwen2_util import update_qwen2_model_for_ssa
from .qwen2_vl_util import update_qwen2_vl_model_for_ssa


def speculative_sparse_attention_without_lookahead_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    max_capacity_prompt: int,
    window_size: int = 32,
    prefill_window_size: int = 2048,
    num_vertical: int = 2048,
    query_aggregation: Literal["mean", "max"] = "max",
    pooling: Literal["mean", "max"] = "max",
    kernel_size: int = 5,
    generation_kwargs: dict[str, Any] = {},
    multimodal_inputs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_ssa(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_ssa(model)
    elif isinstance(model, Qwen2_5_VLForConditionalGeneration):
        update_qwen2_vl_model_for_ssa(model)
    else:
        raise NotImplementedError()

    model.config.window_size = window_size
    model.config.max_capacity_prompt = max_capacity_prompt
    model.config.prefill_window_size = prefill_window_size
    model.config.num_vertical = num_vertical
    model.config.query_aggregation = query_aggregation
    model.config.pooling = pooling
    model.config.kernel_size = kernel_size

    return model.generate(  # type: ignore
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        **generation_kwargs,
        **multimodal_inputs,
    )


def speculative_sparse_attention_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    lookahead_ids: Tensor,
    max_capacity_prompt: int,
    window_size: int = 32,
    prefill_window_size: int = 2048,
    num_vertical: int = 2048,
    query_aggregation: Literal["mean", "max"] = "max",
    pooling: Literal["mean", "max"] = "max",
    kernel_size: int = 5,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    assert max_capacity_prompt <= prefill_window_size + num_vertical
    assert prefill_window_size % 64 == 0 and kernel_size % 2 == 1

    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_ssa(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_ssa(model)
    else:
        raise NotImplementedError()

    if input_ids.shape[1] <= max_capacity_prompt:
        return model.generate(  # type: ignore
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **generation_kwargs,
        )

    lookahead_size = lookahead_ids.shape[1] - input_ids.shape[1]

    model.config.window_size = window_size + lookahead_size + 1
    model.config.max_capacity_prompt = max_capacity_prompt + lookahead_size + 1
    model.config.prefill_window_size = prefill_window_size
    model.config.num_vertical = num_vertical
    model.config.query_aggregation = query_aggregation
    model.config.pooling = pooling
    model.config.kernel_size = kernel_size

    extended_attention_mask = F.pad(attention_mask, (0, lookahead_size), value=1)

    past_key_values = DynamicCache()

    with torch.no_grad():
        model.model(
            input_ids=lookahead_ids,
            attention_mask=extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    past_key_values.crop(max_capacity_prompt)
    del extended_attention_mask

    generated_ids = model.generate(
        input_ids=input_ids[:, -max_capacity_prompt - 1 :],
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_key_values,
        **generation_kwargs,
    )

    return torch.cat((input_ids[:, : -max_capacity_prompt - 1], generated_ids), dim=1)  # type: ignore


def greedy_vl_ssa_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    lookahead_ids: Tensor,
    max_capacity_prompt: int,
    window_size: int = 32,
    prefill_window_size: int = 2048,
    num_vertical: int = 2048,
    query_aggregation: Literal["mean", "max"] = "max",
    pooling: Literal["mean", "max"] = "max",
    kernel_size: int = 5,
    generation_kwargs: dict[str, Any] = {},
    multimodal_inputs: dict[str, Any] = {},
) -> Tensor:
    assert max_capacity_prompt <= prefill_window_size + num_vertical
    assert prefill_window_size % 64 == 0 and kernel_size % 2 == 1

    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        update_qwen2_vl_model_for_ssa(model)
    else:
        raise NotImplementedError()

    if input_ids.shape[1] <= max_capacity_prompt:
        return model.generate(  # type: ignore
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **generation_kwargs,
            **multimodal_inputs,
        )

    lookahead_size = lookahead_ids.shape[1] - input_ids.shape[1]

    model.config.window_size = window_size + lookahead_size + 1
    model.config.max_capacity_prompt = max_capacity_prompt + lookahead_size + 1
    model.config.prefill_window_size = prefill_window_size
    model.config.num_vertical = num_vertical
    model.config.query_aggregation = query_aggregation
    model.config.pooling = pooling
    model.config.kernel_size = kernel_size

    extended_attention_mask = F.pad(attention_mask, (0, lookahead_size), value=1)

    lookahead_position_ids, rope_deltas = model.get_rope_index(
        lookahead_ids,  # type: ignore
        multimodal_inputs["image_grid_thw"],
        None,
        None,
        extended_attention_mask,
    )

    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=lookahead_ids,
            attention_mask=extended_attention_mask,
            position_ids=lookahead_position_ids,
            rope_deltas=rope_deltas,
            past_key_values=past_key_values,
            use_cache=True,
            **multimodal_inputs,
        )

    past_key_values.crop(max_capacity_prompt)
    del extended_attention_mask

    sequences = input_ids
    position_ids = model.get_rope_index(
        input_ids,  # type: ignore
        multimodal_inputs["image_grid_thw"],
        None,
        None,
        attention_mask,
    )[0][..., -1:]
    input_ids = input_ids[:, -1:]

    for _ in range(generation_kwargs.get("max_new_tokens", 64)):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rope_deltas=rope_deltas,
                past_key_values=past_key_values,
                use_cache=True,
            )

        input_ids = outputs.logits[:, -1:].argmax(dim=-1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        position_ids += 1
        sequences = torch.cat((sequences, input_ids), dim=1)

        eos_token_id: int | list[int] = model.generation_config.eos_token_id  # type: ignore
        if isinstance(eos_token_id, int) and input_ids[0, -1].item() == eos_token_id:
            break
        elif input_ids[0, -1].item() in eos_token_id:  # type: ignore
            break

    return sequences  # type: ignore
