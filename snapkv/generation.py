from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .llama_util import update_llama_model_for_snapkv
from .qwen2_util import update_qwen2_model_for_snapkv


def snapkv_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    window_size: int,
    max_capacity_prompt: int,
    kernel_size: int = 5,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_snapkv(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_snapkv(model)
    else:
        raise NotImplementedError()

    model.config.max_capacity_prompt = max_capacity_prompt
    model.config.window_size = window_size
    model.config.kernel_size = kernel_size

    return model.generate(  # type: ignore
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        **generation_kwargs,
    )


def lookahead_snapkv_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    lookahead_ids: Tensor,
    window_size: int,
    max_capacity_prompt: int,
    kernel_size: int = 5,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_snapkv(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_snapkv(model)
    else:
        raise NotImplementedError()

    lookahead_size = lookahead_ids.shape[1] - input_ids.shape[1]

    model.config.max_capacity_prompt = max_capacity_prompt + lookahead_size + 1
    model.config.window_size = window_size + lookahead_size + 1
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
