"""Pytest configuration: shared fixtures and path setup."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ensure repo root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tmp_storage(tmp_path_factory):
    """Redirect storage to a temp dir for the whole test session."""
    base = tmp_path_factory.mktemp("ai_storage")
    (base / "datasets").mkdir()
    (base / "tokenizers").mkdir()
    (base / "checkpoints").mkdir()
    (base / "models").mkdir()
    (base / "uploads").mkdir()
    return base


@pytest.fixture
def small_config():
    from packages.model import ModelConfig
    return ModelConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4,
                       num_kv_heads=2, intermediate_size=128, max_seq_len=64)
