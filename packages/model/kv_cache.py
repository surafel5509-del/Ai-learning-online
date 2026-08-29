"""KV cache for efficient autoregressive inference.

One cache per (layer, kv_head). Stores key/value tensors in fp32 (cast at use).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class KVCache:
    """Append-only key/value cache for grouped-query attention."""

    def __init__(self, batch_size: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int, device: torch.device, dtype: torch.dtype):
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        # Pre-allocate buffer; track how many positions are filled.
        self.keys = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.values = torch.zeros_like(self.keys)
        self.length = 0

    def update(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new k/v (B, num_kv_heads, S, head_dim) and return full k/v up to length."""
        s = new_keys.shape[2]
        end = self.length + s
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: {end} > max_seq_len {self.max_seq_len}"
            )
        self.keys[:, :, self.length:end, :] = new_keys.to(self.dtype)
        self.values[:, :, self.length:end, :] = new_values.to(self.dtype)
        self.length = end
        return (
            self.keys[:, :, :end, :],
            self.values[:, :, :end, :],
        )

    @property
    def filled_keys(self) -> torch.Tensor:
        return self.keys[:, :, : self.length, :]

    @property
    def filled_values(self) -> torch.Tensor:
        return self.values[:, :, : self.length, :]

    def reset(self) -> None:
        self.length = 0

    def trim(self, length: int) -> None:
        if length < 0:
            length = 0
        self.length = min(length, self.length)
