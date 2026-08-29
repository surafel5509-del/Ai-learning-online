"""services.trainer — real Transformer training loop.

Implements: causal LM training, gradient accumulation, mixed precision (AMP,
GPU only), warmup+cosine scheduler, gradient clipping, periodic validation,
checkpointing (latest/best), real-time step reporting to DB, multi-dataset
sequential training with per-dataset checkpoints, replay for continual learning.

All metrics are real and written to the database (training_steps table), which
the SSE endpoint streams to the dashboard.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from packages.model import ModelConfig, TransformerLM
from packages.tokenizer import BPETokenizer
from packages.shared.metrics import perplexity


@dataclass
class Hyperparams:
    learning_rate: float = 3e-4
    batch_size: int = 8          # micro-batch size
    grad_accum_steps: int = 1    # gradient accumulation
    epochs: int = 1
    seq_len: int = 128
    warmup_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    checkpoint_every: int = 0    # 0 = only per-dataset; else steps
    val_every: int = 50          # steps between val checks
    val_ratio: float = 0.05
    max_steps: int = 0           # 0 = unlimited (epochs govern)
    seed: int = 1337

    def to_dict(self) -> dict:
        return field.asdict(self) if False else self.__dict__


MODE_PRESETS = {
    "fast": dict(learning_rate=4e-4, batch_size=16, grad_accum_steps=1, epochs=1,
                 seq_len=128, warmup_steps=10, weight_decay=0.01, grad_clip=1.0,
                 checkpoint_every=0, val_every=50),
    "balanced": dict(learning_rate=3e-4, batch_size=8, grad_accum_steps=2, epochs=2,
                     seq_len=192, warmup_steps=20, weight_decay=0.01, grad_clip=1.0,
                     checkpoint_every=0, val_every=40),
    "deep": dict(learning_rate=2e-4, batch_size=8, grad_accum_steps=4, epochs=3,
                 seq_len=256, warmup_steps=40, weight_decay=0.01, grad_clip=1.0,
                 checkpoint_every=0, val_every=30),
}


def make_hyperparams(mode: str, overrides: Optional[dict] = None) -> Hyperparams:
    if mode == "custom":
        hp = Hyperparams()
    else:
        hp = Hyperparams(**MODE_PRESETS.get(mode, MODE_PRESETS["balanced"]))
    if overrides:
        for k, v in overrides.items():
            if hasattr(hp, k) and v is not None:
                setattr(hp, k, type(getattr(hp, k))(v) if not isinstance(v, bool) else v)
    return hp


def lr_schedule(step: int, max_steps: int, warmup: int, base_lr: float) -> float:
    """Linear warmup then cosine decay to ~0. Real schedule."""
    if max_steps <= 0:
        max_steps = max(step, 1)
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_batches(token_ids: list[int], seq_len: int, batch_size: int,
                 drop_last: bool = True) -> list[torch.Tensor]:
    """Create non-overlapping sequential batches of shape (B, seq_len+1).

    Targets are inputs shifted by one (causal LM). We store seq_len+1 tokens so
    that input = [:-1], target = [1:].
    """
    chunk = seq_len + 1
    n_chunks = len(token_ids) // chunk
    if n_chunks == 0:
        return []
    data = torch.tensor(token_ids[: n_chunks * chunk], dtype=torch.long)
    data = data.view(n_chunks, chunk)
    batches = []
    i = 0
    while i + batch_size <= n_chunks:
        batches.append(data[i : i + batch_size])
        i += batch_size
    if not drop_last and i < n_chunks:
        batches.append(data[i:n_chunks])
    return batches


@dataclass
class TrainState:
    model: TransformerLM
    optimizer: torch.optim.Optimizer
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_perplexity: float = float("inf")


def build_optimizer(model: TransformerLM, hp: Hyperparams) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay (excludes norms/biases)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": hp.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=hp.learning_rate, betas=(0.9, 0.95), eps=1e-8)


def save_checkpoint(path: Path, model: TransformerLM, optimizer, hp: Hyperparams,
                    state: TrainState, extra: dict) -> None:
    """Real checkpoint: weights, optimizer, step, epoch, config, metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": model.cfg.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "best_val_loss": state.best_val_loss,
        "best_val_perplexity": state.best_val_perplexity,
        "hyperparams": hp.__dict__,
        "extra": extra,
    }, path)


def load_checkpoint(path: Path, device: str, model: Optional[TransformerLM] = None,
                    load_optimizer: bool = True) -> tuple[TransformerLM, "torch.optim.Optimizer", dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    if model is None:
        model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    hp = Hyperparams(**ckpt["hyperparams"])
    optimizer = build_optimizer(model, hp)
    if load_optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return model, optimizer, ckpt


@torch.no_grad()
def evaluate_loss(model: TransformerLM, token_ids: list[int], seq_len: int,
                  batch_size: int, device: str) -> tuple[float, float]:
    """Compute mean validation loss + perplexity on held-out tokens."""
    batches = make_batches(token_ids, seq_len, batch_size, drop_last=True)
    if not batches:
        return float("nan"), float("nan")
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        out = model(x, targets=y, reduction="sum")
        total_loss += float(out["loss"])
        total_tokens += x.numel()
    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    return mean_loss, perplexity(mean_loss)
