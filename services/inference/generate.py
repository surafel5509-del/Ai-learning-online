"""services.inference — real autoregressive generation with KV cache + sampling.

Supports: streaming token-by-token, temperature, top-k, top-p, repetition
penalty, max tokens, BOS priming. Uses the KV cache for O(1) per-step after the
prefill. All generation stats (tokens, latency, speed) are measured.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from packages.model import TransformerLM
from packages.tokenizer import BPETokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled
    repetition_penalty: float = 1.0
    do_sample: bool = True

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class GenerationResult:
    text: str
    tokens: list[int]
    num_tokens: int
    latency_ms: float
    tokens_per_sec: float
    settings: dict


def _apply_logits_processors(logits: torch.Tensor, generated_ids: list[int],
                             repetition_penalty: float) -> torch.Tensor:
    """Repetition penalty: lower the logit of already-generated tokens."""
    if repetition_penalty == 1.0 or not generated_ids:
        return logits
    for tid in set(generated_ids):
        if 0 <= tid < logits.shape[-1]:
            logits[tid] = logits[tid] / repetition_penalty
    return logits


def _sample(logits: torch.Tensor, cfg: GenerationConfig,
            generated_ids: list[int]) -> int:
    """Temperature -> top-k -> top-p -> sample. Greedy if not sampling."""
    logits = logits.clone().float()
    logits = _apply_logits_processors(logits, generated_ids, cfg.repetition_penalty)
    if not cfg.do_sample or cfg.temperature <= 0:
        return int(torch.argmax(logits).item())
    if cfg.temperature != 1.0:
        logits = logits / cfg.temperature
    # top-k
    if cfg.top_k and cfg.top_k > 0:
        k = min(cfg.top_k, logits.shape[-1])
        vals, _ = torch.topk(logits, k)
        thresh = vals[-1]
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    # top-p (nucleus)
    if cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > cfg.top_p
        # keep at least one token
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(model: TransformerLM, tokenizer: BPETokenizer, prompt: str,
             cfg: GenerationConfig, device: str = "cpu",
             stream: bool = False) -> "GenerationResult | Iterator[str]":
    """Generate text from a prompt.

    If stream=False returns a GenerationResult. If stream=True returns an
    iterator yielding text chunks (decoded incrementally) plus a final
    GenerationResult via the `result` attribute on the iterator.
    """
    model.eval()
    bos = tokenizer.bos_id
    eos = tokenizer.eos_id
    prompt_ids = [bos] + tokenizer.encode(prompt)
    # Respect context window
    max_ctx = model.cfg.max_seq_len
    if len(prompt_ids) > max_ctx - 1:
        prompt_ids = prompt_ids[-(max_ctx - 1):]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # Prefill
    caches = model.prefill_kv_caches(input_ids, max_new_len=cfg.max_new_tokens)
    generated: list[int] = []
    last_id = prompt_ids[-1]
    start = time.perf_counter()

    def step_one() -> Optional[int]:
        nonlocal last_id
        if len(generated) >= cfg.max_new_tokens:
            return None
        if caches[0].length >= max_ctx:
            return None
        cur = torch.tensor([[last_id]], dtype=torch.long, device=device)
        start_pos = caches[0].length
        out = model(cur, kv_caches=caches, start_pos=start_pos)
        logits = out["logits"][0, -1]
        nxt = _sample(logits, cfg, generated + prompt_ids)
        if nxt == eos:
            return None
        generated.append(nxt)
        last_id = nxt
        return nxt

    if not stream:
        while step_one() is not None:
            pass
        elapsed = (time.perf_counter() - start) * 1000
        text = tokenizer.decode(generated)
        tps = len(generated) / (elapsed / 1000) if elapsed > 0 else 0.0
        return GenerationResult(text=text, tokens=generated, num_tokens=len(generated),
                                latency_ms=elapsed, tokens_per_sec=tps,
                                settings=cfg.to_dict())

    # Streaming generator
    def _gen():
        # yield prompt first as context marker? No — yield only new tokens.
        prev_decoded_len = 0
        full_text = ""
        while True:
            nxt = step_one()
            if nxt is None:
                break
            # Decode incrementally; BPE merges may change earlier chars, so we
            # re-decode the full generated list and yield the delta.
            new_text = tokenizer.decode(generated)
            if len(new_text) > prev_decoded_len:
                delta = new_text[prev_decoded_len:]
                prev_decoded_len = len(new_text)
                full_text = new_text
                yield delta
        elapsed = (time.perf_counter() - start) * 1000
        tps = len(generated) / (elapsed / 1000) if elapsed > 0 else 0.0
        _gen.result = GenerationResult(  # type: ignore[attr-defined]
            text=full_text, tokens=generated, num_tokens=len(generated),
            latency_ms=elapsed, tokens_per_sec=tps, settings=cfg.to_dict())
    return _gen()
