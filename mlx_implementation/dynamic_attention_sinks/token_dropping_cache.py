import mlx.core as mx
from mlx_lm.models.cache import KVCache


class TokenDroppingKVCache(KVCache):
    def __init__(self):
        super().__init__()
        self.true_offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.true_offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.true_offset += keys.shape[2]
        self.keys[..., prev : self.true_offset, :] = keys
        self.values[..., prev : self.true_offset, :] = values
        return (
            self.keys[..., : self.true_offset, :],
            self.values[..., : self.true_offset, :],
        )

    @property
    def state(self):
        if self.true_offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.true_offset, :],
                self.values[..., : self.true_offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]
        self.true_offset = self.keys.shape[2]

    def is_trimmable(self):  # type: ignore
        return False

    def trim(self, n):
        raise NotImplementedError

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError

    def token_select_indices(self, indices: list[int]):
        assert self.keys is not None and self.values is not None
        self.true_offset = len(indices)
        self.keys[..., : self.true_offset, :] = self.keys[..., indices, :]
        self.values[..., : self.true_offset, :] = self.values[..., indices, :]
