"""Tests for dataset processing: parsing, cleaning, dedup, analysis, split."""
import os
import tempfile

from packages.shared.dataset import (
    parse_file, clean_text, deduplicate, analyze_documents, split_tokens,
    write_token_bin, read_token_bin, file_checksum,
)
from packages.shared.security import allowed_file_type


def test_parse_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("First paragraph.\n\nSecond paragraph.")
    docs = parse_file(f, "txt")
    assert len(docs) == 2
    assert "First" in docs[0]


def test_parse_jsonl(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"text":"doc one"}\n{"text":"doc two"}\n')
    docs = parse_file(f, "jsonl")
    assert len(docs) == 2
    assert "doc one" in docs[0]


def test_parse_csv(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("text\n\"hello\"\n\"world\"\n")
    docs = parse_file(f, "csv")
    assert len(docs) == 2


def test_clean_text():
    raw = "Hello\x00World\r\n  spaces  "
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "Hello" in cleaned


def test_deduplicate():
    docs = ["same", "same", "different", "same", "unique"]
    result = deduplicate(docs)
    deduped = result[0] if isinstance(result, tuple) else result
    assert len(deduped) == 3


def test_analyze_documents():
    from packages.tokenizer import BPETokenizer
    tok = BPETokenizer(version="0.1")
    tok.train(["one two three four five"] * 5, target_vocab_size=280)
    docs = ["one two three", "four five six seven"]
    a = analyze_documents(docs, tok)
    assert a.num_documents == 2
    assert a.estimated_words >= 7
    assert a.raw_chars > 0
    assert a.num_tokens > 0


def test_split_tokens():
    tokens = list(range(1000))
    train, val = split_tokens(tokens, val_ratio=0.1)
    assert len(train) == 900
    assert len(val) == 100
    assert set(train).isdisjoint(set(val))


def test_token_bin_roundtrip(tmp_path):
    tokens = [1, 2, 3, 255, 1000, 70000]
    path = tmp_path / "tokens.bin"
    write_token_bin(path, tokens)
    loaded = read_token_bin(path)
    assert loaded == tokens


def test_file_checksum(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("content")
    h1 = file_checksum(f)
    f.write_text("content2")
    h2 = file_checksum(f)
    assert h1 != h2


def test_allowed_file_type():
    assert allowed_file_type("doc.txt")
    assert allowed_file_type("doc.json")
    assert allowed_file_type("doc.jsonl")
    assert allowed_file_type("doc.csv")
    assert allowed_file_type("doc.md")
    assert not allowed_file_type("malware.exe")
    assert not allowed_file_type("script.sh")
