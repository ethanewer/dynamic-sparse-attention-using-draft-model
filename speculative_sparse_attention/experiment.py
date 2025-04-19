import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .llama_util import update_llama_model_for_ssa
from .qwen2_util import update_qwen2_model_for_ssa


def speculative_sparse_attention_without_lookahead_experiment(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    generated_ids: Tensor,
    window_size: int,
    max_capacity_prompt: int,
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_ssa(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_ssa(model)
    else:
        raise NotImplementedError()

    model.config.max_capacity_prompt = max_capacity_prompt
    model.config.window_size = window_size

    input_len = input_ids.shape[1]

    position_ids = None
    past_key_values = DynamicCache()

    logits = []
    for i in range(input_len, generated_ids.shape[1]):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        input_ids = generated_ids[:, i : i + 1]
        position_ids = torch.tensor([[i]], device=input_ids.device)

        if i > input_len:
            logits.append(outputs.logits[0, -1:])

    return torch.cat(logits).float().cpu()


def speculative_sparse_attention_experiment(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    lookahead_ids: Tensor,
    generated_ids: Tensor,
    window_size: int,
    max_capacity_prompt: int,
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_ssa(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_ssa(model)
    else:
        raise NotImplementedError()

    input_len = input_ids.shape[1]
    lookahead_size = lookahead_ids.shape[1] - input_ids.shape[1]

    model.config.max_capacity_prompt = max_capacity_prompt + lookahead_size + 1
    model.config.window_size = window_size + lookahead_size + 1

    past_key_values = DynamicCache()

    assert (input_ids == generated_ids[:, :input_len]).all()

    with torch.no_grad():
        _ = model(
            input_ids=lookahead_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    past_key_values.crop(max_capacity_prompt)

    logits = []
    for i in range(input_len, generated_ids.shape[1]):
        input_ids = generated_ids[:, i - 1 : i]
        position_ids = torch.tensor([[i - 1]], device=input_ids.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        if i > input_len:
            logits.append(outputs.logits[0, -1:])

    return torch.cat(logits).float().cpu()
