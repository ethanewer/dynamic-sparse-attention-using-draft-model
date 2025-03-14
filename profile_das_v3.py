import gc
from collections import defaultdict
import json
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from dynamic_attention_sinks import dynamic_attention_sinks_generate_v3

assert torch.cuda.is_available()
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=device,
)

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()


generation_kwargs = dict(
    max_length=None,
    max_new_tokens=32,
    min_new_tokens=32,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    eos_token_id=None,
    pad_token_id=None,
)

dynamic_attention_sinks_generate_v3(
    model=model,
    input_ids=torch.randint(8192, (1, 2048), device=device),
    reduced_attentions=torch.randn(24, 1, 2, 2048, dtype=torch.bfloat16),
    block_size=2048 // 16,
    k=2048 // 16,
    generation_kwargs=generation_kwargs,
)


max_memory_allocated_before = torch.cuda.max_memory_allocated() / 1024**2
max_memory_reserved_before = torch.cuda.max_memory_reserved() / 1024**2

results = defaultdict(list)

for input_size in range(2048, 50000, 2048):
    input_ids: Tensor = torch.randint(8192, (1, input_size), device=device)
    reduced_attentions = torch.randn(24, 1, 2, input_size, dtype=torch.bfloat16)

    clear_cache()

    print("\nINPUT SIZE:", input_size)

    torch.cuda.reset_peak_memory_stats()

    dynamic_attention_sinks_generate_v3(
        model=model,
        input_ids=input_ids,
        reduced_attentions=reduced_attentions,
        block_size=input_size // 16,
        k=input_size // 16,
        generation_kwargs=generation_kwargs,
    )

    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
    max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
    max_memory_allocated_dif = max_memory_allocated - max_memory_allocated_before
    max_memory_reserved_dif = max_memory_reserved - max_memory_reserved_before
    print(f"    Max GPU Memory Allocated: {max_memory_allocated:.2f} MB")
    print(f"    Max GPU Memory Reserved: {max_memory_reserved:.2f} MB")
    print(f"    Max New GPU Memory Allocated: {max_memory_allocated_dif:.2f} MB")
    print(f"    Max New GPU Memory Reserved: {max_memory_reserved_dif:.2f} MB")

    clear_cache()

    results["max_memory_allocated"].append(max_memory_allocated)
    results["max_memory_reserved"].append(max_memory_reserved)
    results["max_memory_allocated_dif"].append(max_memory_allocated_dif)
    results["max_memory_reserved_dif"].append(max_memory_reserved_dif)

with open("das-memory-benchmark.json", "w") as f:
    json.dump(results, f)


# INPUT SIZE: 49152
#     Max GPU Memory Allocated: 16286.92 MB
#     Max GPU Memory Reserved: 20792.00 MB
#     Max New GPU Memory Allocated: 14723.22 MB
#     Max New GPU Memory Reserved: 19098.00 MB