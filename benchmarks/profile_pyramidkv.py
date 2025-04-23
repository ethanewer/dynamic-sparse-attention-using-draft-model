import gc
import json
import time
from collections import defaultdict

import torch
from pyramidkv.monkeypatch import replace_llama  # type: ignore
from torch import Tensor
from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore

MODEL = "llama"
ALG = "streamingllm"
NUM_FULL_TOKENS = 64

replace_llama(ALG)

assert MODEL == "llama"
assert torch.cuda.is_available()
device = "cuda"


full_model_name = (
    "meta-llama/Llama-3.1-8B-Instruct"
    if MODEL == "llama"
    else "Qwen/Qwen2.5-14B-Instruct"
)
draft_model_name = (
    "meta-llama/Llama-3.2-1B-Instruct"
    if MODEL == "llama"
    else "Qwen/Qwen2.5-0.5B-Instruct"
)

model = AutoModelForCausalLM.from_pretrained(
    full_model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
)


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()


generation_kwargs = dict(
    max_length=None,
    max_new_tokens=NUM_FULL_TOKENS,
    min_new_tokens=NUM_FULL_TOKENS,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    eos_token_id=None,
    pad_token_id=None,
)

model.generate(
    input_ids=torch.randint(8192, (1, 8192), device=device),
    attention_mask=torch.ones(1, 8192, device=device),
    **generation_kwargs,
)

clear_cache()

max_memory_allocated_before = torch.cuda.max_memory_allocated() / 1024**2
max_memory_reserved_before = torch.cuda.max_memory_reserved() / 1024**2

results = defaultdict(list)

for input_size in range(8192, 100000 if MODEL == "llama" else 82000, 8192):
    input_ids: Tensor = torch.randint(8192, (1, input_size), device=device)
    attention_mask = torch.ones_like(input_ids)

    clear_cache()

    print("\nINPUT SIZE:", input_size)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
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


with open(
    f"quantized-{MODEL}-pyramidkv-{ALG}-benchmark({NUM_FULL_TOKENS}).json", "w"
) as f:
    json.dump(results, f)
