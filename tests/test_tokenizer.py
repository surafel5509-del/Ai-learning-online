"""Tests for the BPE tokenizer, including Amharic/Ethiopic Unicode."""
import pytest

from packages.tokenizer import BPETokenizer


def test_train_and_basic_roundtrip():
    tok = BPETokenizer(version="0.1")
    text = "hello world hello there world world"
    info = tok.train([text], target_vocab_size=300)
    assert tok.vocab_size >= 256
    assert info.num_merges > 0
    ids = tok.encode("hello world")
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert "hello" in decoded


def test_encode_decode_consistency():
    tok = BPETokenizer(version="0.1")
    tok.train(["the quick brown fox jumps over the lazy dog"] * 10, target_vocab_size=350)
    original = "the quick brown fox"
    ids = tok.encode(original)
    decoded = tok.decode(ids)
    # decoded should contain the original words (byte-level preserves content)
    assert all(w in decoded for w in original.split())


def test_save_load(tmp_path):
    tok = BPETokenizer(version="0.1")
    tok.train(["alpha beta gamma delta epsilon"] * 8, target_vocab_size=320)
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode("alpha beta") == tok.encode("alpha beta")


def test_amharic_ethiopic_unicode():
    """Tokenizer must handle Amharic (Ethiopic script) without crashing
    and preserve the characters through encode/decode."""
    tok = BPETokenizer(version="0.1")
    amharic = "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።"  # "Addis Ababa is the capital of Ethiopia."
    text = (amharic + " ") * 20
    tok.train([text], target_vocab_size=320)
    ids = tok.encode(amharic)
    assert len(ids) > 0
    decoded = tok.decode(ids)
    # Ethiopic chars should survive (byte-level UTF-8 preserves them)
    assert "አዲስ" in decoded or "አበባ" in decoded


def test_unicode_coverage():
    tok = BPETokenizer(version="0.1")
    text = "English text አማርኛ 123 numbers"
    cov = tok.unicode_coverage([text])
    assert isinstance(cov, dict)
    assert len(cov) > 0


def test_empty_input():
    tok = BPETokenizer(version="0.1")
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_vocab_size_minimum():
    tok = BPETokenizer(version="0.1")
    # below the 256 byte floor should still produce at least 256
    tok.train(["ab"], target_vocab_size=260)
    assert tok.vocab_size >= 256
