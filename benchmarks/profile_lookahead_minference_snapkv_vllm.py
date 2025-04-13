import gc
import json
import os
import time
from collections import defaultdict

import torch
from minference import MInference  # type: ignore
from torch import Tensor
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from vllm import LLM, SamplingParams  # type: ignore

from snapkv import lookahead_minference_snapkv_generate

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = "cuda"
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    print("Loading full model with Transformers...")
    full_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )

    minference_patch = MInference(
        "minference",
        "Qwen/Qwen2.5-7B-Instruct",
        kv_type="snapkv",
    )

    full_model = minference_patch(full_model)
    print("Full model loaded successfully.")

    print("Loading draft model with vLLM...")
    print("Attempting to load draft model without quantization...")
    draft_model_vllm = LLM(
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
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
        max_new_tokens=32,
        min_new_tokens=32,
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
        max_tokens=32,
        min_tokens=32,
        skip_special_tokens=False,
        ignore_eos=True,
    )

    print("Performing warmup...")
    draft_model_vllm.generate(
        prompt_token_ids=torch.randint(8192, (1, 2048)).tolist(),
        sampling_params=draft_sampling_params,
        use_tqdm=False,
    )
    lookahead_minference_snapkv_generate(
        model=full_model,
        minference_config=minference_patch.config,
        input_ids=torch.randint(8192, (1, 2048), device=device),
        attention_mask=torch.ones(1, 2048, device=device),
        lookahead_ids=torch.randint(8192, (1, 2080), device=device),
        window_size=2048 // 32,
        max_capacity_prompt=2048 // 8,
        generation_kwargs=generation_kwargs,
    )
    clear_cache()
    print("Warmup complete.")

    max_memory_allocated_before = torch.cuda.max_memory_allocated() / 1024**2
    max_memory_reserved_before = torch.cuda.max_memory_reserved() / 1024**2

    results = defaultdict(list)

    for input_size in range(8192, 82000, 8192):
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

        lookahead_minference_snapkv_generate(
            model=full_model,
            minference_config=minference_patch.config,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            lookahead_ids=lookahead_ids_tensor,
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

        results["time"].append(dt)
        results["max_memory_allocated"].append(max_memory_allocated)
        results["max_memory_reserved"].append(max_memory_reserved)
        results["max_memory_allocated_dif"].append(max_memory_allocated_dif)
        results["max_memory_reserved_dif"].append(max_memory_reserved_dif)
        results["input_size"].append(input_size)
        results["generated_length"].append(len(generated_ids_list))

        clear_cache()

    output_filename = (
        "quantized-7b-vllm-draft-lookahead-minference-snapkv-benchmark.json"
    )
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBenchmark complete. Results saved to {output_filename}")
