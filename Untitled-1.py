import torch
from transformers import AutoModelForCausalLM, DynamicCache  # type: ignore

from dynamic_attention_sinks import TokenDroppingCache
from dynamic_attention_sinks.indices_util import get_cache_update_indices, get_indices
from dynamic_attention_sinks.llama_util import (
    update_llama_model_for_dynamic_attention_sinks,
)

device = "mps"

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map=device,
    torch_dtype=torch.float32,
)

input_len = 17
block_size = 3
k = 3

attention_matrix = (
    (
        torch.randn(
            model.config.num_hidden_layers,
            1,
            model.config.num_key_value_heads,
            input_len + 1,
            input_len + 1,
        )
        - 3.4028e38 * torch.ones(input_len + 1, input_len + 1).triu(1)
    )
    .softmax(dim=3)
    .expand(
        model.config.num_hidden_layers, -1, model.config.num_key_value_heads, -1, -1
    )
)


reduced_attentions = attention_matrix[:, :, :, -1:, :-1].mean(dim=3)

input_ids = torch.arange(input_len, device=device)[None]
position_ids = (
    torch.arange(input_ids.shape[1], device=device)
    .view(1, -1)
    .expand(input_ids.shape[0], input_ids.shape[1])
)

prefill_input_len = input_ids.shape[1] - 1

cache_update_indices = get_cache_update_indices(
    reduced_attentions[..., :-1],
    k=k,
    block_size=block_size,
    reduce_heads=False,
    device=input_ids.device,  # type: ignore
)

past_key_values_1 = TokenDroppingCache()
block_logits = []

for block_idx in range((prefill_input_len + block_size - 1) // block_size):
    block_start = block_idx * block_size
    block_end = min((block_idx + 1) * block_size, prefill_input_len)
    block_input_ids = input_ids[:, block_start:block_end]
    block_position_ids = position_ids[:, block_start:block_end]

    with torch.no_grad():
        outputs = model(
            input_ids=block_input_ids,
            position_ids=block_position_ids,
            use_cache=True,
            past_key_values=past_key_values_1,
        )

    for layer_idx in range(model.config.num_hidden_layers):
        past_key_values_1.token_select_indices(
            cache_update_indices[block_idx][layer_idx],
            layer_idx=layer_idx,
        )

    block_logits.append(outputs.logits)


logits1 = torch.cat(block_logits, dim=1)

past_key_values_1.get_seq_length()

update_llama_model_for_dynamic_attention_sinks(model)

indices = get_indices(
    reduced_attentions[..., :-1],
    k=k,
    block_size=block_size,
).to(input_ids.device)

past_key_values_2 = DynamicCache()

with torch.no_grad():
    logits2 = model(
        input_ids=input_ids[:, :-1],
        dynamic_attention_sinks_block_size=block_size,
        dynamic_attention_sinks_indices=indices,
        past_key_values=past_key_values_2,
    ).logits

torch.allclose(logits1, logits2, rtol=1e-6, atol=1e-3)

indices[0, 0, -1]

for layer_idx in range(model.config.num_hidden_layers):
    assert torch.allclose(
        past_key_values_1[layer_idx][0].sort(dim=-2).values,  # type: ignore
        past_key_values_2[layer_idx][0].sort(dim=-2).values,  # type: ignore
        rtol=1e-6,
        atol=1e-3,
    )
    assert torch.allclose(
        past_key_values_1[layer_idx][1].sort(dim=-2).values,  # type: ignore
        past_key_values_2[layer_idx][1].sort(dim=-2).values,  # type: ignore
        rtol=1e-6,
        atol=1e-3,
    )
