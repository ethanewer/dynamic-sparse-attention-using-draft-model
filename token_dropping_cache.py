import torch
from transformers import DynamicCache  # type: ignore


class TokenDroppingCache(DynamicCache):
    def token_select_indices(self, indices: torch.Tensor | list[int]):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., indices, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., indices, :]
