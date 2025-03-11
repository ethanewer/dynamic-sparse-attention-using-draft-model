import gc
import tracemalloc

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    LlamaForCausalLM,
)

from dynamic_attention_sinks import (
    dynamic_attention_sinks_generate_v3,
    generate_reduced_attentions,
)
from reduced_attention_mapping import KLDivAttentionMapping

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


draft_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=device,
)

full_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=device,
)

attention_mapping = KLDivAttentionMapping(
    "reduced_attention_mapping/kl-div-qwen-0.5b-to-3b.pt"
)


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()


def print_mean_times(d, s="", skip_first=True):
    for k, v in d.items():
        if skip_first:
            s += f"{k}: {sum(v[1:]) / len(v[1:]):.4f}s, "
        else:
            s += f"{k}: {sum(v) / len(v):.4f}s, "
    print(s[:-2])


num_new_tokens = 32

generation_kwargs = dict(
    max_length=None,
    max_new_tokens=num_new_tokens,
    min_new_tokens=num_new_tokens,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    eos_token_id=None,
    pad_token_id=None,
)

clear_cache()
for input_size in [512] + list(range(512, 2048, 512)):  # range(8192, 65536 + 1, 8192):
    input_ids: Tensor = torch.randint(8192, (1, input_size), device=device)

    draft_model = draft_model.to(device)  # type: ignore

    reduced_attentions = generate_reduced_attentions(
        model=draft_model,
        input_ids=input_ids,
        reduction="mean",
        generation_kwargs=generation_kwargs,
    )[1].cpu()

    draft_model = draft_model.cpu()

    # reduced_attentions = attention_mapping(reduced_attentions)

    reduced_attentions = F.avg_pool1d(
        reduced_attentions.reshape(-1, reduced_attentions.shape[-1]),
        kernel_size=5,
        stride=1,
        padding=2,
        count_include_pad=False,
    ).view(reduced_attentions.shape)
    clear_cache()

    print("INPUT SIZE:", input_size)

    tracemalloc.start()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dynamic_attention_sinks_generate_v3(
        model=full_model,
        input_ids=input_ids,
        reduced_attentions=reduced_attentions,
        block_size=input_size // 16,
        k=input_size // 16,
        generation_kwargs=generation_kwargs,
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"    Max CPU Memory Usage: {peak / 1024**2:.2f} MB")
    if device == "cuda":
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
        max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
        print(f"    Max GPU Memory Allocated: {max_memory_allocated:.2f} MB")
        print(f"    Max GPU Memory Reserved: {max_memory_reserved:.2f} MB")

    print()
    clear_cache()
