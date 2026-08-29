"""Real metrics: loss, perplexity, retention, growth score, knowledge score.

Every value is computed from actual data. Composite scores are clearly labeled.
"""
from __future__ import annotations

import math
from typing import Optional


def perplexity(loss: float) -> float:
    """Perplexity = exp(loss). Real, from cross-entropy loss."""
    if loss is None or math.isnan(loss) or math.isinf(loss):
        return float("inf")
    return math.exp(min(loss, 50.0))  # guard overflow for very high loss


def knowledge_retention_score(before: dict, after: dict) -> tuple[float, dict]:
    """Compute a real Knowledge Retention Score from before/after benchmark results.

    before/after: dict mapping test_id -> score in [0,1].
    Retention = mean(after/before) over tests where before>0, plus credit for
    tests maintained. Returns (score in 0..1, detail dict).
    """
    ratios = []
    maintained = 0
    total = 0
    for k, b in before.items():
        a = after.get(k, 0.0)
        total += 1
        if b <= 0:
            # couldn't do before; only credit if it can now (growth, not retention)
            continue
        r = a / b
        r = max(0.0, min(1.0, r))
        ratios.append(r)
        if a >= b:
            maintained += 1
    if not ratios:
        score = 0.0 if total == 0 else (sum(after.values()) / total if total else 0.0)
    else:
        score = sum(ratios) / len(ratios)
    detail = {
        "tests": total,
        "maintained_or_improved": maintained,
        "degraded": total - maintained,
        "before_mean": (sum(before.values()) / len(before)) if before else 0.0,
        "after_mean": (sum(after.values()) / len(after)) if after else 0.0,
    }
    return score, detail


def degradation_flagged(retention_score: float, threshold: float = 0.85) -> bool:
    """True if old knowledge significantly degraded (below threshold)."""
    return retention_score < threshold


def evaluation_score(results: list[dict]) -> float:
    """Mean score over evaluation results (each {score: float 0..1})."""
    if not results:
        return 0.0
    return sum(r.get("score", 0.0) for r in results) / len(results)


def ai_growth_score(
    evaluation: float,
    retention: float,
    validation: float,
    vocab_coverage: float,
    training_progress: float,
    task_performance: float,
) -> tuple[float, dict]:
    """Composite AI Growth Score. NOT a scientific intelligence measure.

    Weighted blend of measurable signals, each in [0,1]:
      evaluation (0.25), retention (0.25), validation (0.20),
      vocab_coverage (0.10), training_progress (0.10), task_performance (0.10).
    Returns (score 0..100, breakdown). Training time alone never raises it.
    """
    def c(x, lo=0.0, hi=1.0):
        if x is None:
            return 0.0
        return max(lo, min(hi, float(x)))
    weights = {
        "evaluation": 0.25,
        "retention": 0.25,
        "validation": 0.20,
        "vocab_coverage": 0.10,
        "training_progress": 0.10,
        "task_performance": 0.10,
    }
    vals = {
        "evaluation": c(evaluation),
        "retention": c(retention),
        "validation": c(validation),
        "vocab_coverage": c(vocab_coverage),
        "training_progress": c(training_progress),
        "task_performance": c(task_performance),
    }
    raw = sum(vals[k] * weights[k] for k in weights)
    score = round(raw * 100, 1)
    return score, {"raw": round(raw, 4), "components": vals, "weights": weights,
                   "note": "Composite of measurable signals; not a scientific intelligence measure."}


def estimate_words_from_tokens(num_tokens: int, tokens_per_word: float = 1.3) -> int:
    """Rough word estimate from token count (English ~1.3 tokens/word). Labeled estimate."""
    if num_tokens <= 0:
        return 0
    return int(num_tokens / tokens_per_word)
