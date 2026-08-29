"""Tests for memory/RAG: embedding, vector search, retrieval."""
import numpy as np
import pytest

from services.memory.memory import embed_text


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    a = np.array([1.0, 1.0])
    b = np.array([-1.0, -1.0])
    assert abs(cosine_similarity(a, b) + 1.0) < 1e-6


def test_embedding_dim():
    from services.memory.memory import embed_text
    v = embed_text("hello world")
    assert isinstance(v, np.ndarray)
    assert v.ndim == 1 and v.shape[0] > 0


def test_vector_search_retrieves_relevant():
    """Documents semantically similar to query should rank higher."""
    from services.memory.memory import embed_text
    docs = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "Water boils at 100 degrees Celsius.",
    ]
    doc_vecs = [embed_text(d) for d in docs]
    query = embed_text("What is the capital of France?")
    scores = [cosine_similarity(query, dv) for dv in doc_vecs]
    best_idx = int(np.argmax(scores))
    assert best_idx == 0  # the Paris doc should be most similar
