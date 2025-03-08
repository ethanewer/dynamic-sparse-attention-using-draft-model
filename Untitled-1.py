# %%
import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
)

from dynamic_attention_sinks.indices_util import (
    get_indices,
    get_sink_indices,
)
from dynamic_attention_sinks.llama_util import (
    update_llama_model_for_dynamic_attention_sinks,
)

device = "cpu"

# %%
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map=device,
    torch_dtype=torch.float32,
)

# %%
input_len = 20
block_size = 2
k = 2

attention_matrix = (
    (
        torch.randn(
            1,
            1,
            model.config.num_key_value_heads,
            input_len + 1,
            input_len + 1,
        )
        - 3.4028e38 * torch.ones(input_len + 1, input_len + 1).triu(1)
    )
    .softmax(dim=3)
    .expand(model.config.num_hidden_layers, -1, -1, -1, -1)
)

# plt.imshow(attention_matrix[0, 0, 0], cmap="gray_r")
# plt.show()

reduced_attentions = attention_matrix[:, :, :, -1:, :-1].mean(dim=3)

sink_indices = get_sink_indices(
    reduced_attentions,
    k=k,
    block_size=block_size,
)

print(reduced_attentions[0, 0, 0].sort(descending=True).indices)

print(sink_indices[0, 0, 0], sink_indices.shape)

mask = torch.zeros(1, model.config.num_attention_heads, input_len, input_len)

for ib in range(0, input_len, block_size):
    mask[:, :, ib : ib + 2 * block_size, ib : ib + block_size] = 1

for layer_i in range(1):
    for batch_i in range(1):
        for head_i in range(model.config.num_key_value_heads):
            for block_i, indices in enumerate(sink_indices[layer_i, batch_i, head_i]):
                block_start = (block_i + 1) * block_size
                block_end = (block_i + 2) * block_size
                mask[batch_i, head_i, block_start:block_end, indices] = 1

            mask[batch_i, head_i].tril_()


# plt.imshow(mask[0, 0], cmap="gray_r", extent=(0, mask.shape[3], 0, mask.shape[2]))

# plt.xticks(torch.arange(0, mask.shape[3] + 1, 1), labels=" " * (mask.shape[3] + 1))
# plt.yticks(torch.arange(0, mask.shape[2] + 1, 1), labels=" " * (mask.shape[2] + 1))
# plt.grid()
# plt.show()

mask = -3.4028e38 * (1 - mask).to(device)
input_ids = torch.arange(input_len, device=device)[None]
position_ids = (
    torch.arange(input_ids.shape[1], device=device)
    .view(1, -1)
    .expand(input_ids.shape[0], input_ids.shape[1])
)

# %%
with torch.no_grad():
    logits1 = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=position_ids,
    ).logits

# %%
update_llama_model_for_dynamic_attention_sinks(model)

indices = get_indices(reduced_attentions, k, block_size).to(device)

print(indices[0, 0, 0])

with torch.no_grad():
    logits2 = model(
        input_ids=input_ids,
        position_ids=position_ids,
        dynamic_attention_sinks_block_size=block_size,
        dynamic_attention_sinks_indices=indices,
    ).logits

# %%
