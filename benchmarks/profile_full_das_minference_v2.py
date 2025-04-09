import gc
import json
import time
from collections import defaultdict

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore

from das_minference import das_minference_generate, generate_reduced_attentions
from reduced_attention_mapping import AverageAttentionMapping

assert torch.cuda.is_available()
device = "cuda"

draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
)

full_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    attn_implementation="flash_attention_2",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
)

attention_mapping = AverageAttentionMapping(device=device)
attention_mapping.w = torch.ones(
    draft_model.config.num_hidden_layers,
    draft_model.config.num_key_value_heads,
    full_model.config.num_hidden_layers,
    full_model.config.num_key_value_heads,
    device=device,
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

reduced_attentions = generate_reduced_attentions(
    draft_model,
    input_ids=torch.randint(8192, (1, 2048), device=device),
    generation_kwargs=generation_kwargs,
)[1]

das_minference_generate(
    model=full_model,
    input_ids=torch.randint(8192, (1, 16384), device=device),
    reduced_attentions=attention_mapping(reduced_attentions),
    window_size=16384 // 32,
    max_capacity_prompt=16384 // 8,
    generation_kwargs=generation_kwargs,
)

clear_cache()

max_memory_allocated_before = torch.cuda.max_memory_allocated() / 1024**2
max_memory_reserved_before = torch.cuda.max_memory_reserved() / 1024**2

results = defaultdict(list)

for input_size in range(2048, 100000, 2048):
    input_ids: Tensor = torch.randint(8192, (1, input_size), device=device)

    clear_cache()

    print("\nINPUT SIZE:", input_size)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    reduced_attentions = generate_reduced_attentions(
        draft_model,
        input_ids=input_ids,
        generation_kwargs=generation_kwargs,
    )[1]

    das_minference_generate(
        model=full_model,
        input_ids=input_ids,
        reduced_attentions=attention_mapping(reduced_attentions),
        window_size=64,
        max_capacity_prompt=1024,
        generation_kwargs=generation_kwargs,
    )

    dt = time.time() - t0
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
    max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
    max_memory_allocated_dif = max_memory_allocated - max_memory_allocated_before
    max_memory_reserved_dif = max_memory_reserved - max_memory_reserved_before
    print(f"    Time: {dt:.4f}s")
    print(f"    Max GPU Memory Allocated: {max_memory_allocated:.2f} MB")
    print(f"    Max GPU Memory Reserved: {max_memory_reserved:.2f} MB")
    print(f"    Max New GPU Memory Allocated: {max_memory_allocated_dif:.2f} MB")
    print(f"    Max New GPU Memory Reserved: {max_memory_reserved_dif:.2f} MB")

    clear_cache()

    results["time"].append(dt)
    results["max_memory_allocated"].append(max_memory_allocated)
    results["max_memory_reserved"].append(max_memory_reserved)
    results["max_memory_allocated_dif"].append(max_memory_allocated_dif)
    results["max_memory_reserved_dif"].append(max_memory_reserved_dif)
    results["input_size"].append(input_size)

with open("quantized-llama-das-full-benchmark.json", "w") as f:
    json.dump(results, f)
