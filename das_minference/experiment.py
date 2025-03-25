import torch
from torch import Tensor
from transformers import (  # type: ignore
    DynamicCache,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

from .llama_util import update_llama_model_for_das_minference
from .qwen2_util import update_qwen2_model_for_das_minference


def das_minference_experiment(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    generated_ids: Tensor,
    reduced_attentions: Tensor,
    window_size: int,
    max_capacity_prompt: int,
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_das_minference(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_das_minference(model)
    else:
        raise NotImplementedError()

    input_len = input_ids.shape[1]

    assert window_size < max_capacity_prompt and window_size % 64 == 0
    k = min(max_capacity_prompt, input_len) - window_size

    past_key_values = DynamicCache()

    if k <= 0:
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
    else:
        v_idx = reduced_attentions[..., :input_len].topk(k, dim=-1).indices.int()
        s_idx = torch.arange(
            window_size,
            -1,
            -64,
            dtype=v_idx.dtype,
            device=v_idx.device,
        )[None, None, None].expand(*v_idx.shape[:-1], -1)

        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                v_idx=v_idx,
                s_idx=s_idx,
                window_size=window_size,
            )

    assert past_key_values.get_seq_length() <= max_capacity_prompt

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        input_ids = generated_ids[:, i - 1 : i]
        position_ids = torch.tensor([[i - 1]], device=input_ids.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )

        logits.append(outputs.logits)

    return torch.cat(logits, dim=1).float().cpu()
