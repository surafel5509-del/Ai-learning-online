"""Tests for evaluation, retention scoring, and model promotion logic."""
import torch

from packages.model import ModelConfig, TransformerLM
from packages.tokenizer import BPETokenizer
from services.evaluator import run_tests, EvalTest
from packages.shared.metrics import (
    perplexity, knowledge_retention_score, ai_growth_score, estimate_words_from_tokens,
)


def test_perplexity():
    import math
    assert abs(perplexity(2.0) - math.exp(2.0)) < 1e-6
    assert perplexity(0.0) == 1.0


def test_retention_score_no_change():
    before = {"t1": 0.8, "t2": 0.6}
    after = {"t1": 0.8, "t2": 0.6}
    score, detail = knowledge_retention_score(before, after)
    assert score == 1.0  # no forgetting


def test_retention_score_full_forgetting():
    before = {"t1": 1.0}
    after = {"t1": 0.0}
    score, detail = knowledge_retention_score(before, after)
    assert score < 0.3  # severe forgetting flagged


def test_retention_score_partial():
    before = {"t1": 1.0, "t2": 1.0}
    after = {"t1": 1.0, "t2": 0.5}
    score, _ = knowledge_retention_score(before, after)
    assert 0.6 < score < 1.0


def test_growth_score_never_zero_with_data():
    gs, detail = ai_growth_score(evaluation=0.5, retention=0.8, validation=0.5,
                                 vocab_coverage=0.6, training_progress=1.0, task_performance=0.5)
    assert 0 < gs <= 100
    assert "note" in detail


def test_growth_score_labelled_estimate():
    _, detail = ai_growth_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert "estimate" in detail["note"].lower() or "composite" in detail["note"].lower() or "not" in detail["note"].lower()


def test_estimate_words():
    w = estimate_words_from_tokens(1000)
    assert 500 < w < 1500  # heuristic ratio


def test_run_tests_generation():
    cfg = ModelConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4,
                      num_kv_heads=2, intermediate_size=128, max_seq_len=32)
    model = TransformerLM(cfg)
    tok = BPETokenizer(version="0.1")
    tok.train(["hello world"] * 10, target_vocab_size=300)
    tests = [EvalTest(test_id="t1", question="What is", expected_answer="hello", criteria="contains")]
    from services.inference import GenerationConfig
    gcfg = GenerationConfig(max_new_tokens=10, do_sample=False)
    results = run_tests(model, tok, tests, "cpu", gcfg)
    assert len(results) == 1
    assert "score" in results[0]
    assert "response" in results[0]
