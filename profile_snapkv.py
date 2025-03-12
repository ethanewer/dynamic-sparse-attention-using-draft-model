import gc
import tracemalloc

import torch
from torch import Tensor
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    LlamaForCausalLM,
)

from reduced_attention_mapping import LinearAttentionMapping
from snapkv import lookahead_snapkv_generate

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

attention_mapping = LinearAttentionMapping(
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


num_new_tokens = 20

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
    attention_mask = torch.ones_like(input_ids)

    draft_model = draft_model.to(device)  # type: ignore

    lookahead_ids: Tensor = draft_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,  # type: ignore
    )

    draft_model = draft_model.cpu()

    tracemalloc.start()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    lookahead_snapkv_generate(
        model=full_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        lookahead_ids=lookahead_ids,
        window_size=input_size // 32,
        max_capacity_prompt=input_size // 8,
        generation_kwargs=generation_kwargs,
    )

    print("INPUT SIZE:", input_size)

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
