from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import Cache, DynamicCache  # type: ignore


def dsa_step(
    draft_model: Any,
    full_model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    cache_position: Tensor,
    draft_past_key_values: Cache,
    full_past_key_values: Cache,
    k: int,
) -> Tensor:
    assert input_ids.shape[1] == 1
    assert attention_mask.ndim == 2
    assert position_ids.shape[1] == 1
    assert cache_position.shape == (1,)
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
    reduced_attentions = (attentions.square().sum(dim=0) / len(attentions)).sqrt()

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


def adaptive_dsa_prefill(
    draft_model: Any,
    full_model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    window_sizes: tuple[int, ...],
    kl_div_thresh: float,
) -> tuple[Tensor, Tensor, Tensor, DynamicCache, DynamicCache, DynamicCache]:
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    assert attention_mask.ndim == 2 and attention_mask.shape[0] == 1
    assert position_ids.ndim == 2 and position_ids.shape[0] == 1

    device = input_ids.device

    cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device=device)
    true_draft_past_key_values = DynamicCache()
    draft_past_key_values = DynamicCache()
    full_past_key_values = DynamicCache()

    with torch.no_grad():
        true_draft_outputs = draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
            cache_position=cache_position,
            past_key_values=true_draft_past_key_values,
            use_cache=True,
        )

    attentions = torch.cat([a[0, :, :, :] for a in true_draft_outputs.attentions])
    reduced_attentions = (attentions.square().sum(dim=0) / len(attentions)).sqrt()
    true_draft_logits: Tensor = true_draft_outputs.logits[0]

    window_mask = torch.zeros(
        len(window_sizes),
        attentions.shape[1],
        attentions.shape[2],
        dtype=torch.int64,
        device=device,
    )
    for i, window_size in enumerate(window_sizes):
        for j in range(len(reduced_attentions)):
            if j <= window_size:
                window_mask[i, j, : j + 1] = attention_mask[0, : j + 1]
            else:
                indices = reduced_attentions[j, : j + 1].topk(window_size).indices
                window_mask[i, j, indices] = attention_mask[0, indices]

    window_mask_4d = -3.4028e38 * (1 - window_mask[:, None, :, :]).to(draft_model.dtype)  # type: ignore

    with torch.no_grad():
        draft_outputs = draft_model(
            input_ids=input_ids.repeat(len(window_sizes), 1),
            attention_mask=window_mask_4d,
            position_ids=position_ids.repeat(len(window_sizes), 1),
            cache_position=cache_position,
            past_key_values=draft_past_key_values,
            use_cache=True,
        )

    draft_logits: Tensor = draft_outputs.logits

    best_mask_idx = None
    for i in range(len(window_sizes)):
        kl_div = F.kl_div(
            draft_logits[i].log_softmax(dim=-1),
            true_draft_logits.log_softmax(dim=-1),
            reduction="batchmean",
            log_target=True,
        ).item()

        if kl_div <= kl_div_thresh or i == len(window_sizes) - 1:
            best_mask_idx = i
            draft_past_key_values.reorder_cache(torch.tensor([i], device=device))  # type: ignore
            break

    assert best_mask_idx is not None

    with torch.no_grad():
        full_outputs = full_model(
            input_ids=input_ids,
            attention_mask=window_mask[[best_mask_idx]],
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=full_past_key_values,
            use_cache=True,
        )

    return (
        full_outputs.logits,
        window_mask[best_mask_idx],
        cache_position,
        true_draft_past_key_values,
        draft_past_key_values,
        full_past_key_values,
    )


def adaptive_dsa_step(
    draft_model: Any,
    full_model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    cache_position: Tensor,
    true_draft_past_key_values: DynamicCache,
    draft_past_key_values: DynamicCache,
    full_past_key_values: DynamicCache,
    window_sizes: tuple[int, ...],
    kl_div_thresh: float,
    greedy: bool,
) -> tuple[Tensor, Tensor]:
    assert input_ids.shape == (1, 1)
    assert attention_mask.ndim == 2 and attention_mask.shape[0] == 1
    assert position_ids.shape == (1, 1)
    assert cache_position.shape == (1,)

    device = input_ids.device

    with torch.no_grad():
        true_draft_outputs = draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
            cache_position=cache_position,
            past_key_values=true_draft_past_key_values,
            use_cache=True,
        )

    attentions = torch.cat([a[0, :, 0, :] for a in true_draft_outputs.attentions])
    reduced_attentions = (attentions.square().sum(dim=0) / len(attentions)).sqrt()
    true_draft_logits: Tensor = true_draft_outputs.logits[0, -1:]

    window_mask = torch.zeros(
        len(window_sizes),
        attention_mask.shape[1],
        dtype=torch.int64,
        device=device,
    )
    for i, window_size in enumerate(window_sizes):
        if attention_mask.shape[1] <= window_size:
            window_mask[i] = attention_mask[0]
        else:
            indices = reduced_attentions.topk(window_size).indices
            window_mask[i, indices] = attention_mask[0, indices]

    draft_past_key_values.reorder_cache(
        torch.tensor([0] * len(window_sizes), device=device)  # type: ignore
    )

    with torch.no_grad():
        draft_outputs = draft_model(
            input_ids=input_ids.repeat(len(window_sizes), 1),
            attention_mask=window_mask,
            position_ids=position_ids.repeat(len(window_sizes), 1),
            cache_position=cache_position,
            past_key_values=draft_past_key_values,
            use_cache=True,
        )

    draft_logits: Tensor = draft_outputs.logits[:, -1:]

    best_mask_idx = None
    for i in range(len(window_sizes)):
        correct: bool = (draft_logits[i].argmax() == true_draft_logits.argmax()).item()  # type: ignore
        kl_div = F.kl_div(
            draft_logits[i].log_softmax(dim=-1),
            true_draft_logits.log_softmax(dim=-1),
            reduction="batchmean",
            log_target=True,
        ).item()

        if i == len(window_sizes) - 1:
            done = True
        elif greedy:
            done = correct and (kl_div <= kl_div_thresh)
        else:
            done = kl_div <= kl_div_thresh

        if done:
            best_mask_idx = i
            draft_past_key_values.reorder_cache(torch.tensor([i], device=device))  # type: ignore
            break

    assert best_mask_idx is not None

    with torch.no_grad():
        full_outputs = full_model(
            input_ids=input_ids,
            attention_mask=window_mask[[best_mask_idx]],
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=full_past_key_values,
            use_cache=True,
        )

    return full_outputs.logits[:, -1:], window_mask[best_mask_idx]
