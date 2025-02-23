import mlx.core as mx
from mlx_lm.models import llama, qwen2
from mlx_lm.models.cache import KVCache

from .llama_util import reset_llama_model, update_llama_model_to_output_attns
from .qwen2_util import reset_qwen2_model, update_qwen2_model_to_output_attns


def sample(logits, temp):
    if temp == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temp))


def generate_reduced_attentions(
    model: llama.Model | qwen2.Model,
    inputs: mx.array,
    max_new_tokens: int,
    temp: float = 0,
    eos_token_ids: set[int] = set(),
) -> tuple[mx.array, mx.array]:
    input_len = inputs.shape[1]
    sequences = mx.array(inputs)
    logits = []
    attention_scores = []

    cache = [KVCache() for _ in range(len(model.model.layers))]
    attn_log = None
    for i in range(max_new_tokens):
        outputs = model(inputs=inputs, cache=cache)

        inputs = sample(outputs[:, -1:], temp)
        sequences = mx.concatenate([sequences, inputs], axis=1)

        if i == 0:
            if isinstance(model, llama.Model):
                attn_log = update_llama_model_to_output_attns(model)
            elif isinstance(model, qwen2.Model):
                attn_log = update_qwen2_model_to_output_attns(model)
            else:
                raise NotImplementedError
        else:
            logits.append(outputs[0, -1:])
            assert attn_log is not None
            attention_scores.append(
                mx.concatenate([a[0, :, :, :input_len] for a in attn_log])
                .square()
                .sum(axis=0)
            )
            attn_log.clear()

        if inputs[0, -1].item() in eos_token_ids:
            break

    if isinstance(model, llama.Model):
        reset_llama_model(model)
    elif isinstance(model, qwen2.Model):
        reset_qwen2_model(model)
    else:
        raise NotImplementedError

    reduced_attentions = mx.concatenate(attention_scores, axis=0).sum(axis=0).sqrt()

    return mx.concatenate(logits), reduced_attentions


def dynamic_attention_sinks_generate(
    model: llama.Model | qwen2.Model,
    input_ids: mx.array,
    generated_ids: mx.array,
    reduced_attentions: mx.array,
    block_size: int,
    k: int,
    temp: float = 0,
): ...


#     if isinstance(model, LlamaForCausalLM):
#         for layer in model.model.layers:
#             assert isinstance(layer.self_attn, LlamaAttention)
#     elif isinstance(model, Qwen2ForCausalLM):
#         for layer in model.model.layers:
#             assert isinstance(layer.self_attn, Qwen2Attention)
#     else:
#         raise NotImplementedError()

#     assert input_ids.shape[0] == 1
#     input_len = input_ids.shape[1]
#     position_ids = torch.arange(input_ids, device=input_ids.device)[None]  # type: ignore

#     k = min(k, input_len - block_size)
#     sink_indices = reduced_attentions[: input_len - block_size].topk(k).indices.tolist()
#     cache_seq_indices = []
#     past_key_values = TokenDroppingCache()

#     for i in range(0, input_ids.shape[1] - 1, block_size):
#         j = min(i + block_size, input_ids.shape[1] - 1)
#         block_input_ids = input_ids[:, i:j]
#         block_position_ids = position_ids[:, i:j]

#         with torch.no_grad():
#             _ = model(
#                 input_ids=block_input_ids,
#                 position_ids=block_position_ids,
#                 use_cache=True,
#                 past_key_values=past_key_values,
#             )

#         cache_seq_indices += list(range(i, j))
#         selected_indices = []
#         new_cache_seq_indices = []
#         for cache_idx, seq_idx in enumerate(cache_seq_indices):
#             if seq_idx in sink_indices or seq_idx >= j - block_size:
#                 selected_indices.append(cache_idx)
#                 new_cache_seq_indices.append(seq_idx)

#         past_key_values.token_select_indices(
#             torch.tensor(selected_indices, device=input_ids.device)
#         )
#         cache_seq_indices = new_cache_seq_indices

#     cache_size = min(block_size + k, input_len - 1)
#     assert past_key_values.get_seq_length() == cache_size

#     generated_ids = model.generate(  # type: ignore
#         input_ids=input_ids[:, -cache_size - 1 :],
#         attention_mask=torch.ones_like(input_ids),
#         use_cache=True,
#         past_key_values=past_key_values,
#         **generation_kwargs,
#     )

#     return torch.cat((input_ids[:, : -cache_size - 1], generated_ids), dim=1)
