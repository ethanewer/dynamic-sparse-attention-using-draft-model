import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (  # type: ignore
    DynamicCache,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

from .dsa import dsa_step
from .llama_util import update_llama_model_for_dsa
from .qwen2_util import update_qwen2_model_for_dsa


def dsa_generate_greedy(
    draft_model: LlamaForCausalLM | Qwen2ForCausalLM,
    full_model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    k: int,
    max_new_tokens: int,
) -> Tensor:
    if isinstance(full_model, LlamaForCausalLM):
        update_llama_model_for_dsa(full_model)
    elif isinstance(full_model, Qwen2ForCausalLM):
        update_qwen2_model_for_dsa(full_model)
    else:
        raise NotImplementedError()

    attention_mask = torch.ones_like(input_ids)
    position_ids = attention_mask.cumsum(dim=1) - 1
    sequences = input_ids.clone()

    draft_past_key_values = DynamicCache()
    full_past_key_values = DynamicCache()

    with torch.no_grad():
        _ = draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=draft_past_key_values,
            use_cache=True,
        )
        full_outputs = full_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=full_past_key_values,
            use_cache=True,
        )

    input_ids = full_outputs.logits[:, -1:].argmax(dim=-1)
    attention_mask = F.pad(attention_mask, (0, 1), value=1)
    position_ids = position_ids[:, -1:] + 1
    sequences = torch.cat((sequences, input_ids), dim=1)

    for _ in range(max_new_tokens):
        full_logits = dsa_step(
            draft_model=draft_model,
            full_model=full_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=None,
            draft_past_key_values=draft_past_key_values,
            full_past_key_values=full_past_key_values,
            k=k,
        )

        input_ids = full_logits.argmax(dim=-1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        position_ids = position_ids[:, -1:] + 1
        sequences = torch.cat((sequences, input_ids), dim=1)

        eos_token_id: int | list[int] = full_model.generation_config.eos_token_id  # type: ignore
        if isinstance(eos_token_id, int) and input_ids[0, -1].item() == eos_token_id:
            break
        elif input_ids[0, -1].item() in eos_token_id:  # type: ignore
            break

    return sequences
