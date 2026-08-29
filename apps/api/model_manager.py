"""apps.api.model_manager — load & cache trained models for inference.

Loads a ModelVersion's checkpoint + tokenizer into memory, cached by version id.
Used by chat, test lab, evaluation, and comparison endpoints.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import torch

from packages.model import ModelConfig, TransformerLM
from packages.tokenizer import BPETokenizer
from packages.shared import db_models
from packages.shared.config import settings
from services.trainer.core import load_checkpoint


class LoadedModel:
    def __init__(self, model: TransformerLM, tokenizer: BPETokenizer,
                 model_version: db_models.ModelVersion, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_version = model_version
        self.device = device
        self.loaded_at = time.time()

    @property
    def id(self) -> str:
        return self.model_version.id


class ModelManager:
    """In-process cache of loaded models. Thread-safe via a lock."""

    def __init__(self, max_cached: int = 4):
        self._cache: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        self._max = max_cached
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def device(self) -> str:
        return self._device

    def get(self, db, model_version_id: str) -> LoadedModel:
        with self._lock:
            if model_version_id in self._cache:
                return self._cache[model_version_id]
        mv = db.get(db_models.ModelVersion, model_version_id)
        if mv is None:
            raise ValueError(f"Model version {model_version_id} not found")
        if not mv.checkpoint_path or not Path(mv.checkpoint_path).exists():
            raise ValueError(f"Model version {model_version_id} has no checkpoint on disk")
        # tokenizer
        tok = None
        if mv.tokenizer_version_id:
            tv = db.get(db_models.TokenizerVersion, mv.tokenizer_version_id)
            if tv and Path(tv.storage_path).exists():
                tok = BPETokenizer.load(tv.storage_path)
        if tok is None:
            raise ValueError(f"Tokenizer for model version {model_version_id} not found")
        model, _, _ = load_checkpoint(Path(mv.checkpoint_path), self._device,
                                      load_optimizer=False)
        model.eval()
        lm = LoadedModel(model, tok, mv, self._device)
        with self._lock:
            if len(self._cache) >= self._max:
                # evict oldest
                oldest = min(self._cache.values(), key=lambda x: x.loaded_at)
                self._cache.pop(oldest.id, None)
            self._cache[model_version_id] = lm
        return lm

    def invalidate(self, model_version_id: str) -> None:
        with self._lock:
            self._cache.pop(model_version_id, None)

    def list_cached(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())


model_manager = ModelManager()
