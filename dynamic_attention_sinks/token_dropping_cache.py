from transformers import DynamicCache  # type: ignore
from typing import Optional
from torch import Tensor


class TokenDroppingCache(DynamicCache):
    def token_select_indices(
        self,
        indices: Tensor,
        layer_idx: Optional[int] = None,
    ) -> None:
        batch_size = self.key_cache[layer_idx if layer_idx else 0].shape[0]
        num_key_value_heads = self.key_cache[layer_idx if layer_idx else 0].shape[1]
        head_dim = self.key_cache[layer_idx if layer_idx else 0].shape[-1]

        # print(indices.shape)
        if indices.ndim == 1:
            indices = indices[None, None, :, None].expand(
                batch_size,
                num_key_value_heads,
                -1,
                head_dim,
            )
        elif indices.ndim == 2:
            indices = indices[:, None, :, None].expand(
                -1,
                num_key_value_heads,
                -1,
                head_dim,
            )
        elif indices.ndim == 3:
            indices = indices[..., None].expand(-1, -1, -1, head_dim)

        # print(indices.shape)

        #  print(indices[0, 0, :, 0])

        assert (
            indices.shape[:2] == self.key_cache[layer_idx if layer_idx else 0].shape[:2]
            and indices.shape[-1]
            == self.key_cache[layer_idx if layer_idx else 0].shape[-1]
        ), (indices.shape, self.key_cache[layer_idx if layer_idx else 0].shape)

        if layer_idx is None:
            # p = self.key_cache[0].shape
            for layer_idx in range(len(self)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(
                    dim=2,
                    index=indices,
                )
                self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(
                    dim=2,
                    index=indices,
                )

            # print(p, "->", self.key_cache[0].shape, indices.shape)
        else:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(
                dim=2,
                index=indices,
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(
                dim=2,
                index=indices,
            )
