from typing import Optional

import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


def dsa_step(
    draft_model: LlamaForCausalLM | Qwen2ForCausalLM,
    full_model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    cache_position: Optional[Tensor],
    draft_past_key_values: Cache,
    full_past_key_values: Cache,
    k: int,
) -> Tensor:
    assert input_ids.shape[1] == 1
    assert attention_mask.ndim == 2
    assert position_ids.shape[1] == 1
    assert cache_position is None or cache_position.shape == (1,)
    k = min(k, attention_mask.shape[1])

    with torch.no_grad():
        draft_outputs = draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
            cache_position=cache_position,
            past_key_values=draft_past_key_values,
            use_cache=True,
        )

    attentions = torch.cat([a[0, :, 0, :] for a in draft_outputs.attentions])
    reduced_attentions = attentions.square().mean(dim=0).sqrt()

    topk_attention_mask = torch.zeros_like(attention_mask)
    indices = reduced_attentions.topk(k).indices
    topk_attention_mask[:, indices] = attention_mask[:, indices]

    with torch.no_grad():
        full_outputs = full_model(
            input_ids=input_ids,
            attention_mask=topk_attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=full_past_key_values,
            use_cache=True,
            dsa_k=k,
        )

    return full_outputs.logits[:, -1:]


# def paged_dsa_step(
#     draft_model: LlamaForCausalLM | Qwen2ForCausalLM,
#     full_model: LlamaForCausalLM | Qwen2ForCausalLM,
#     input_ids: Tensor,
#     attention_mask: Tensor,
#     position_ids: Tensor,
#     cache_position: Optional[Tensor],
#     draft_past_key_values: Cache,
#     full_past_key_values: Cache,
#     k: int,
#     page_size: int,
# ) -> Tensor:
#     assert input_ids.shape[1] == 1
#     assert attention_mask.ndim == 2
#     assert position_ids.shape[1] == 1
#     assert cache_position is None or cache_position.shape == (1,)
#     assert k % page_size == 0
#     k = min(k, attention_mask.shape[1])

#     with torch.no_grad():
#         draft_outputs = draft_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=True,
#             cache_position=cache_position,
#             past_key_values=draft_past_key_values,
#             use_cache=True,
#         )

#     attentions = torch.cat([a[0, :, 0, :] for a in draft_outputs.attentions])
#     reduced_attentions = attentions.square().mean(dim=0)
#     paged_attentions = F.avg_pool1d(
#         F.pad(
#             reduced_attentions[None],
#             (0, -len(reduced_attentions) % 3),
#             value=reduced_attentions[-(len(reduced_attentions) % 3) :].mean(),  # type: ignore
#         ),
#         kernel_size=page_size,
#         stride=page_size,
#     )[0]
#     page_indices = paged_attentions.topk(k // page_size).indices
