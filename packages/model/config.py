"""Model architecture configuration for the decoder-only Transformer LM.

Every field maps to a real architectural knob of the model in transformer.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Configurable architecture for a RoPE + RMSNorm + SwiGLU + GQA decoder LM."""

    vocab_size: int = 256
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 4          # query attention heads
    num_kv_heads: int = 2       # KV heads for grouped-query attention (<= num_heads)
    intermediate_size: int = 512  # SwiGLU FFN hidden dim
    max_seq_len: int = 256      # context length
    rope_theta: float = 10000.0
    rope_pct: float = 0.25      # fraction of head dim that is rotary
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    # dtype used for parameters at construction; mixed precision handled in trainer
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.num_kv_heads > self.num_heads:
            raise ValueError("num_kv_heads must be <= num_heads")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict:
        d = asdict(self)
        d["head_dim"] = self.head_dim
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)
