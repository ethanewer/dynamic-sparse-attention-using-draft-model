import torch
from torch import Tensor
from transformers import (  # type: ignore
    DynamicCache,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

from .dsa import dsa_step
from .llama_util import update_llama_model_for_dsa
from .qwen2_util import update_qwen2_model_for_dsa


def dsa_experiment(
    draft_model: LlamaForCausalLM | Qwen2ForCausalLM,
    full_model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    generated_ids: Tensor,
    k: int,
) -> Tensor:
    if isinstance(full_model, LlamaForCausalLM):
        update_llama_model_for_dsa(full_model)
    elif isinstance(full_model, Qwen2ForCausalLM):
        update_qwen2_model_for_dsa(full_model)
    else:
        raise NotImplementedError()

    assert input_ids.shape[0] == 1
    input_len = input_ids.shape[1]
    draft_past_key_values = DynamicCache()
    full_past_key_values = DynamicCache()

    with torch.no_grad():
        _ = draft_model(
            input_ids=input_ids,
            past_key_values=draft_past_key_values,
            use_cache=True,
        )
        _ = full_model(
            input_ids=input_ids,
            past_key_values=full_past_key_values,
            use_cache=True,
        )

    logits = []
    for i in range(input_len + 1, generated_ids.shape[1]):
        input_ids = generated_ids[:, i - 1 : i]
        attention_mask = torch.ones(1, i, device=input_ids.device)
        position_ids = torch.tensor([[i - 1]], device=input_ids.device)

        logits.append(
            dsa_step(
                draft_model=draft_model,
                full_model=full_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=None,
                draft_past_key_values=draft_past_key_values,
                full_past_key_values=full_past_key_values,
                k=k,
            )[0, -1:].cpu()
        )

    return torch.cat(logits).float().cpu()
