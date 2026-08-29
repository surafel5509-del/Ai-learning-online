"""services.memory — explicit memory + RAG retrieval.

Embeddings use a lightweight real model: a bag-of-character-n-grams hashed into
a fixed dim, L2-normalized. This is a real, deterministic embedding (no external
deps) — sufficient for retrieval over a small knowledge base and clearly an
estimate of semantic similarity, not a frontier embedding model. Vector search
is exact cosine over stored vectors (works in SQLite; pgvector optional in prod).

RAG flow: query -> embed -> cosine search top-k -> build context -> return.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

EMBED_DIM = 256


def _ngrams(text: str, ns: tuple[int, ...] = (2, 3, 4)) -> list[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    chars = text
    out: list[str] = []
    for n in ns:
        for i in range(max(0, len(chars) - n + 1)):
            out.append(chars[i : i + n])
    # also word unigrams/bigrams
    words = text.split()
    out.extend(words)
    for i in range(max(0, len(words) - 1)):
        out.append(words[i] + "_" + words[i + 1])
    return out


def embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Real hashed n-gram embedding, L2-normalized."""
    vec = np.zeros(dim, dtype=np.float32)
    for ng in _ngrams(text):
        h = hashlib.md5(ng.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def chunk_text(text: str, max_chars: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of ~max_chars by sentence boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
            # handle very long sentences by hard split
            while len(cur) > max_chars:
                chunks.append(cur[:max_chars])
                cur = cur[max_chars - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    score: float
    document_id: Optional[str] = None
    title: Optional[str] = None


def build_context(hits: list[RetrievalHit], max_chars: int = 1200) -> str:
    """Concatenate retrieved chunks into a context string for the model."""
    parts: list[str] = []
    total = 0
    for h in hits:
        if total + len(h.text) > max_chars:
            break
        parts.append(h.text)
        total += len(h.text)
    return "\n\n".join(parts)
