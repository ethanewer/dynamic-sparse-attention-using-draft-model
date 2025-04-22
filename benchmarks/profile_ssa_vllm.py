import gc
import json
import os
import time
from collections import defaultdict

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore

from speculative_sparse_attention import speculative_sparse_attention_generate

assert torch.cuda.is_available()
device = "cuda"


NUM_PREFILL = 1024
MAX_CAPACITY_PROMPT = 1024
MODEL = "llama"
NUM_FULL_TOKENS = 64
NUM_DRAFT_TOKENS = 64

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

if __name__ == "__main__":
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    print("Loading full model with Transformers...")
    full_model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        attn_implementation="flash_attention_2",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )
    print("Full model loaded successfully.")

    print("Loading draft model with vLLM...")
    print("Attempting to load draft model without quantization...")
    draft_model_vllm = LLM(
        model=draft_model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=90112,
        rope_scaling={"rope_type": "dynamic", "factor": 3.0},
        gpu_memory_utilization=0.625,
    )
    print("Draft model loaded without vLLM quantization.")

    def clear_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    draft_sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=NUM_DRAFT_TOKENS,
        min_tokens=NUM_DRAFT_TOKENS,
        skip_special_tokens=False,
        ignore_eos=True,
    )

    print("Performing warmup...")
    draft_model_vllm.generate(
        prompt_token_ids=torch.randint(8192, (1, 2048)).tolist(),
        sampling_params=draft_sampling_params,
        use_tqdm=False,
    )
    speculative_sparse_attention_generate(
        model=full_model,
        input_ids=torch.randint(8192, (1, 8192), device=device),
        attention_mask=torch.ones(1, 2048, device=device),
        lookahead_ids=torch.randint(8192, (1, 8192 + 32), device=device),
        num_vertical=NUM_PREFILL,
        prefill_window_size=NUM_PREFILL,
        max_capacity_prompt=MAX_CAPACITY_PROMPT,
        generation_kwargs=generation_kwargs,
    )
    clear_cache()
    print("Warmup complete.")

    max_memory_allocated_before = torch.cuda.max_memory_allocated() / 1024**2
    max_memory_reserved_before = torch.cuda.max_memory_reserved() / 1024**2

    results = defaultdict(list)

    for input_size in range(8192, 100000 if MODEL == "llama" else 82000, 8192):
        input_ids_tensor: Tensor = torch.randint(8192, (1, input_size), device=device)
        attention_mask_tensor = torch.ones_like(input_ids_tensor)

        input_ids_list = input_ids_tensor.tolist()[0]

        clear_cache()

        print(f"\nINPUT SIZE: {input_size}")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        draft_outputs = draft_model_vllm.generate(
            prompt_token_ids=[input_ids_list],
            sampling_params=draft_sampling_params,
            use_tqdm=False,
        )

        generated_ids_list = draft_outputs[0].outputs[0].token_ids
        full_sequence_list = input_ids_list + generated_ids_list
        lookahead_ids_tensor = torch.tensor([full_sequence_list], device=device)

        speculative_sparse_attention_generate(
            model=full_model,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            lookahead_ids=lookahead_ids_tensor,
            num_vertical=NUM_PREFILL,
            prefill_window_size=NUM_PREFILL,
            max_capacity_prompt=MAX_CAPACITY_PROMPT,
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

        results["time"].append(dt)
        results["max_memory_allocated"].append(max_memory_allocated)
        results["max_memory_reserved"].append(max_memory_reserved)
        results["max_memory_allocated_dif"].append(max_memory_allocated_dif)
        results["max_memory_reserved_dif"].append(max_memory_reserved_dif)
        results["input_size"].append(input_size)
        results["generated_length"].append(len(generated_ids_list))

        clear_cache()

    output_filename = f"quantized-{MODEL}-ssa[{NUM_PREFILL}, {MAX_CAPACITY_PROMPT}]-benchmark({NUM_DRAFT_TOKENS}, {NUM_FULL_TOKENS}).json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBenchmark complete. Results saved to {output_filename}")
