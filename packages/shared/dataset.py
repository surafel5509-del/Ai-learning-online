"""Real dataset ingestion, cleaning, analysis, and tokenization pipeline.

Pipeline (never destroys originals):
  upload -> validate -> clean -> analyze -> deduplicate -> tokenize -> split -> count
Originals are stored untouched in STORAGE_DIR/datasets/<version_id>/raw/.
Processed artifacts (documents, token ids) are stored alongside.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import settings


_WS_RE = re.compile(r"\s+")


@dataclass
class DatasetAnalysis:
    num_documents: int = 0
    raw_chars: int = 0
    raw_bytes: int = 0
    estimated_words: int = 0
    num_tokens: int = 0
    unique_vocab_tokens: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    unicode_coverage: dict = field(default_factory=dict)
    duplicates_removed: int = 0
    languages_hint: list = field(default_factory=list)


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_documents_from_text(text: str) -> list[str]:
    """Split text into documents. Paragraphs separated by blank lines are documents;
    otherwise the whole text is one document. Lines within a paragraph kept."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on 2+ newlines = paragraph/document boundaries
    parts = re.split(r"\n{2,}", text)
    docs = []
    for p in parts:
        p = p.strip()
        if p:
            docs.append(p)
    if not docs and text.strip():
        docs = [text.strip()]
    return docs


def parse_file(path: Path, file_type: str) -> list[str]:
    """Parse a supported file into a list of document strings. Originals untouched."""
    if file_type == "txt" or file_type == "md":
        return _extract_documents_from_text(path.read_text(encoding="utf-8", errors="replace"))
    if file_type == "jsonl":
        docs = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                docs.append(line)
                continue
            docs.append(_obj_to_text(obj))
        return docs
    if file_type == "json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return [path.read_text(encoding="utf-8", errors="replace")]
        if isinstance(obj, list):
            return [_obj_to_text(o) for o in obj]
        return [_obj_to_text(obj)]
    if file_type == "csv":
        docs = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docs.append(_obj_to_text(row))
        return docs
    raise ValueError(f"Unsupported file type: {file_type}")


def _obj_to_text(obj) -> str:
    """Flatten an object/dict into readable text. Keys become 'key: value' lines."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
    if isinstance(obj, list):
        return "\n".join(_obj_to_text(x) for x in obj)
    return str(obj)


def clean_text(text: str) -> str:
    """Normalize whitespace; preserve content & case. No destructive edits."""
    text = text.replace("\u0000", "")
    text = _WS_RE.sub(lambda m: m.group(0) if "\n" in m.group(0) else " ", text)
    return text.strip()


def estimate_words(text: str) -> int:
    # Whitespace split is an estimate; CJK/Amharic may differ. Labeled as estimate in UI.
    return len(text.split())


def analyze_documents(docs: list[str], tokenizer) -> DatasetAnalysis:
    """Compute real analysis: chars, words, tokens, vocab, unicode coverage."""
    a = DatasetAnalysis()
    a.num_documents = len(docs)
    a.raw_chars = sum(len(d) for d in docs)
    a.raw_bytes = sum(len(d.encode("utf-8")) for d in docs)
    a.estimated_words = sum(estimate_words(d) for d in docs)
    all_ids: list[int] = []
    for d in docs:
        all_ids.extend(tokenizer.encode(d))
    a.num_tokens = len(all_ids)
    a.unique_vocab_tokens = len(set(all_ids))
    a.unicode_coverage = tokenizer.unicode_coverage(docs)
    return a


def deduplicate(docs: list[str]) -> tuple[list[str], int]:
    """Exact-match document deduplication. Returns (deduped, removed_count)."""
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        key = d.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out, len(docs) - len(out)


def split_tokens(token_ids: list[int], val_ratio: float = 0.05,
                 min_val: int = 1) -> tuple[list[int], list[int]]:
    """Split token ids into train/val at document-block granularity if possible,
    else a simple sequential split. Val is the last portion (chronologically honest)."""
    n = len(token_ids)
    if n == 0:
        return [], []
    n_val = max(min_val, int(n * val_ratio))
    n_val = min(n_val, n - 1) if n > 1 else 0
    return token_ids[: n - n_val], token_ids[n - n_val:]


def write_token_bin(path: Path, token_ids: list[int]) -> None:
    """Persist token ids as a compact binary file (uint32 little-endian)."""
    import struct
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(token_ids)}I", *token_ids))


def read_token_bin(path: Path) -> list[int]:
    import struct
    data = path.read_bytes()
    n = len(data) // 4
    return list(struct.unpack(f"<{n}I", data))
