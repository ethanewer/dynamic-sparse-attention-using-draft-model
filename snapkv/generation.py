from typing import Any

from torch import Tensor
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .llama_util import update_llama_model_for_snapkv
from .qwen2_util import update_qwen2_model_for_snapkv


def snapkv_generate(
    model: LlamaForCausalLM | Qwen2ForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    max_capacity_prompt: int,
    window_size: int = 32,
    kernel_size: int = 5,
    generation_kwargs: dict[str, Any] = {},
) -> Tensor:
    if isinstance(model, LlamaForCausalLM):
        update_llama_model_for_snapkv(model)
    elif isinstance(model, Qwen2ForCausalLM):
        update_qwen2_model_for_snapkv(model)
    else:
        raise NotImplementedError()

    model.config.max_capacity_prompt = max_capacity_prompt
    model.config.window_size = window_size
    model.config.kernel_size = kernel_size

    return model.generate(  # type: ignore
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        **generation_kwargs,
    )
