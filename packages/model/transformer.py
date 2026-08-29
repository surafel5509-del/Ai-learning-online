"""Real decoder-only Transformer language model.

Components:
- Token embedding (+ optional untied LM head)
- RoPE (rotary position embeddings)
- RMSNorm
- Grouped-Query Attention with KV cache support
- SwiGLU feed-forward
- Pre-norm residual blocks
- Causal LM head

All operations are real PyTorch; no placeholders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .kv_cache import KVCache


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


def build_rope(head_dim: int, max_seq_len: int, theta: float, rope_pct: float,
               device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (max_seq_len, head_dim). Only rope_pct of dims rotate."""
    rot_dim = max(1, int(head_dim * rope_pct))
    half = rot_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, half)
    # duplicate to full rot_dim
    freqs = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, rot_dim)
    cos = freqs.cos()
    sin = freqs.sin()
    # pad remaining (non-rotary) dims with ones/zeros
    if rot_dim < head_dim:
        pad = head_dim - rot_dim
        cos = torch.cat([cos, torch.ones(max_seq_len, pad, device=device, dtype=torch.float32)], dim=-1)
        sin = torch.cat([sin, torch.zeros(max_seq_len, pad, device=device, dtype=torch.float32)], dim=-1)
    return cos.to(dtype), sin.to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). cos/sin: (T, D)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.group_size = cfg.num_heads // cfg.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: Optional[KVCache] = None, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            # GQA: repeat kv heads to match q heads
            if self.group_size > 1:
                k = k.repeat_interleave(self.group_size, dim=1)
                v = v.repeat_interleave(self.group_size, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Causal mask: query at position i sees keys <= i+start_pos
            Tq = q.shape[2]
            Tk = k.shape[2]
            causal = torch.tril(
                torch.ones(Tq, Tk, device=x.device, dtype=torch.bool),
                diagonal=Tk - Tq,
            )
            scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # (B,H,Tq,D)
        else:
            # Full training attention with GQA repeat
            if self.group_size > 1:
                k = k.repeat_interleave(self.group_size, dim=1)
                v = v.repeat_interleave(self.group_size, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: Optional[KVCache] = None, start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only causal language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm_f = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None  # use tied weights
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        cos, sin = build_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
                               cfg.rope_pct, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(x)
        # tied embeddings: transpose embed weight (V,H) -> (H,V)
        return F.linear(x, self.embed.weight)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None,
                start_pos: int = 0,
                reduction: str = "mean") -> dict:
        """Returns dict with logits and (if targets) loss.

        With kv_caches provided, input_ids is the new chunk (length T) appended.
        start_pos is where this chunk begins (used to slice RoPE).
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, cos, sin, kv_cache=cache, start_pos=start_pos)
        x = self.norm_f(x)
        logits = self._head(x)  # (B, T, V)
        out = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def prefill_kv_caches(self, input_ids: torch.Tensor, max_new_len: int) -> list:
        """Run a prefill to populate per-layer KV caches for generation."""
        caches = [
            KVCache(input_ids.shape[0], self.cfg.num_kv_heads, self.cfg.head_dim,
                    self.cfg.max_seq_len, input_ids.device, torch.float32)
            for _ in range(self.cfg.num_layers)
        ]
        self.forward(input_ids, kv_caches=caches, start_pos=0)
        return caches

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
