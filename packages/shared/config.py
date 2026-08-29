"""Application configuration (env-driven, with sane dev defaults).

Defaults are chosen so the platform runs with zero external services:
SQLite file DB, local filesystem object storage, in-process + DB-polling worker.
Switch DATABASE_URL to a PostgreSQL DSN for production.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    # --- Storage ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{REPO_ROOT}/data/ai_platform.db")
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", REPO_ROOT / "data" / "storage"))
    CHECKPOINT_DIR: Path = Path(os.getenv("CHECKPOINT_DIR", REPO_ROOT / "data" / "checkpoints"))
    TOKENIZER_DIR: Path = Path(os.getenv("TOKENIZER_DIR", REPO_ROOT / "data" / "tokenizers"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", REPO_ROOT / "data" / "logs"))

    # --- Security ---
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-only-change-me-in-production-32chars!!")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = int(os.getenv("JWT_EXPIRE_HOURS", "168"))
    # bcrypt-ish cost handled by passlib if present, else sha256+salt fallback
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))  # 200MB
    RATE_LIMIT_PER_MIN: int = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

    # --- Training defaults ---
    DEFAULT_DEVICE: str = os.getenv("DEFAULT_DEVICE", "auto")  # auto|cpu|gpu
    WORKER_POLL_SECONDS: float = float(os.getenv("WORKER_POLL_SECONDS", "1.0"))
    SSE_HEARTBEAT_SECONDS: float = float(os.getenv("SSE_HEARTBEAT_SECONDS", "15"))

    # --- Paths ---
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    def ensure_dirs(self) -> None:
        for d in (self.STORAGE_DIR, self.CHECKPOINT_DIR, self.TOKENIZER_DIR, self.LOG_DIR,
                  self.STORAGE_DIR.parent):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
