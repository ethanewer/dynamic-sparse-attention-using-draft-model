from typing import Optional

from torch import Tensor
from transformers.cache_utils import DynamicCache


class TokenDroppingCache(DynamicCache):
    def trim_1d(self, indices: Tensor, layer_idx: Optional[int]):
        assert indices.ndim == 1
        if layer_idx is None:
            for layer_idx in range(len(self)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, indices]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, indices]
        else:
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, indices]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, indices]

    def trim_4d(self, indices: Tensor, layer_idx: Optional[int]):
        assert (
            indices.shape[:2] == self.key_cache[layer_idx if layer_idx else 0].shape[:2]
            and indices.shape[-1]
            == self.key_cache[layer_idx if layer_idx else 0].shape[-1]
        ), (indices.shape, self.key_cache[layer_idx if layer_idx else 0].shape)

        if layer_idx is None:
            for layer_idx in range(len(self)):  # type: ignore
                self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(
                    dim=2,
                    index=indices,
                )
                self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(
                    dim=2,
                    index=indices,
                )

        else:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(
                dim=2,
                index=indices,
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(
                dim=2,
                index=indices,
            )

    def token_select_indices(
        self,
        indices: Tensor,
        layer_idx: Optional[int] = None,
    ) -> None:
        num_key_value_heads = self.key_cache[layer_idx if layer_idx else 0].shape[1]
        head_dim = self.key_cache[layer_idx if layer_idx else 0].shape[-1]

        if indices.ndim == 1:
            self.trim_1d(indices, layer_idx)
        elif indices.ndim == 2:
            if indices.shape[0] == 1:
                self.trim_1d(indices[0], layer_idx)
            else:
                indices = indices[:, None, :, None].expand(
                    -1,
                    num_key_value_heads,
                    -1,
                    head_dim,
                )
                self.trim_4d(indices, layer_idx)
        elif indices.ndim == 3:
            indices = indices[..., None].expand(-1, -1, -1, head_dim)
            self.trim_4d(indices, layer_idx)
        else:
            self.trim_4d(indices, layer_idx)
