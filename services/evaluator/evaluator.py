"""services.evaluator — real evaluation against benchmarks & custom tests.

Generates benchmark Q/A from dataset facts (simple pattern extraction), runs
custom user tests, scores responses with exact-match / substring / token-overlap
criteria, and computes the knowledge retention score from before/after results.

Never declares the model "knows" something just because loss decreased — uses
actual generation + answer checking.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import torch

from packages.model import TransformerLM
from packages.tokenizer import BPETokenizer
from packages.shared.metrics import knowledge_retention_score, evaluation_score
from services.inference import GenerationConfig, generate


# Patterns to extract candidate Q/A facts from text for benchmark generation.
# These are heuristics that produce *candidate* tests; humans can curate them.
FACT_PATTERNS = [
    # "X is the capital of Y." / "Y's capital is X."
    (re.compile(r"([A-Z][\w\s]{1,40}?)\s+is\s+the\s+capital\s+of\s+([A-Z][\w\s]{1,40}?)[.\n]", re.I),
     lambda m: (f"What is the capital of {m.group(2).strip()}?", m.group(1).strip())),
    # "X is the capital of Y." swapped: "The capital of Y is X."
    (re.compile(r"the\s+capital\s+of\s+([A-Z][\w\s]{1,40}?)\s+is\s+([A-Z][\w\s]{1,40}?)[.\n]", re.I),
     lambda m: (f"What is the capital of {m.group(1).strip()}?", m.group(2).strip())),
    # "X is a Y." -> definition
    (re.compile(r"^([A-Z][\w-]{1,40}?)\s+is\s+(a|an)\s+([^.]{2,60})[.]", re.I),
     lambda m: (f"What is {m.group(1).strip()}?", m.group(3).strip())),
]


def extract_benchmark_tests(docs: list[str], max_tests: int = 20) -> list[dict]:
    """Extract candidate benchmark Q/A tests from dataset documents."""
    tests: list[dict] = []
    seen_q: set[str] = set()
    for doc in docs:
        for pat, fn in FACT_PATTERNS:
            for m in pat.finditer(doc):
                try:
                    q, a = fn(m)
                except Exception:
                    continue
                q = q.strip()
                a = a.strip().rstrip(".")
                if not q or not a or len(a) > 60:
                    continue
                if q.lower() in seen_q:
                    continue
                seen_q.add(q.lower())
                tests.append({
                    "question": q,
                    "expected_answer": a,
                    "criteria": "contains",
                })
                if len(tests) >= max_tests:
                    return tests
    return tests


def score_response(response: str, expected: str, criteria: str = "contains") -> tuple[float, bool]:
    """Score a response against expected answer. Returns (score 0..1, passed)."""
    resp = response.strip().lower()
    exp = expected.strip().lower()
    if not exp:
        # No ground truth: give neutral credit if a non-empty answer is produced
        return (0.5, bool(resp))
    if criteria == "exact":
        passed = resp == exp
        return (1.0 if passed else 0.0, passed)
    if criteria == "contains":
        passed = exp in resp
        return (1.0 if passed else 0.0, passed)
    if criteria == "similarity":
        # Token-overlap (Jaccard) — real, simple
        rset = set(re.findall(r"\w+", resp))
        eset = set(re.findall(r"\w+", exp))
        if not eset:
            return (0.5, bool(resp))
        union = rset | eset
        inter = rset & eset
        score = len(inter) / len(union) if union else 0.0
        return (score, score >= 0.5)
    # default contains
    passed = exp in resp
    return (1.0 if passed else 0.0, passed)


@dataclass
class EvalTest:
    test_id: str
    question: str
    expected_answer: str
    criteria: str = "contains"


def run_tests(model: TransformerLM, tokenizer: BPETokenizer, tests: list[EvalTest],
              device: str = "cpu",
              gen_config: Optional[GenerationConfig] = None) -> list[dict]:
    """Run a list of tests against a model. Returns results with real scores/latency."""
    if gen_config is None:
        gen_config = GenerationConfig(max_new_tokens=64, temperature=0.3, top_k=40,
                                      top_p=0.9, repetition_penalty=1.15, do_sample=True)
    results: list[dict] = []
    for t in tests:
        start = time.perf_counter()
        res = generate(model, tokenizer, t.question + "\nAnswer:", gen_config, device=device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        score, passed = score_response(res.text, t.expected_answer, t.criteria)
        results.append({
            "test_id": t.test_id,
            "question": t.question,
            "expected": t.expected_answer,
            "response": res.text,
            "score": score,
            "passed": passed,
            "latency_ms": elapsed_ms,
            "tokens_generated": res.num_tokens,
        })
    return results


def compute_retention(before_results: list[dict], after_results: list[dict]) -> tuple[float, dict]:
    """Map test_id -> score for before/after and compute retention score."""
    before = {r["test_id"]: r["score"] for r in before_results}
    after = {r["test_id"]: r["score"] for r in after_results}
    return knowledge_retention_score(before, after)
