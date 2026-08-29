"""Full database schema for the AI platform.

Tables: users, datasets, dataset_versions, dataset_files, training_jobs,
training_steps, models, model_versions, checkpoints, memories, documents,
document_chunks, embeddings, conversations, messages, feedback, evaluations,
evaluation_results, workers, schedules, corrections, tokenizer_versions.

Relationships, indexes, and JSON columns are used throughout. All timestamps
are UTC. Status fields use constrained strings.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey,
    JSON, Index, Enum as SAEnum, LargeBinary,
)
from sqlalchemy.orm import relationship

from .database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid.uuid4().hex


# ---------- Enums ----------
class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EVALUATING = "evaluating"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeviceType(str, enum.Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class TrainingMode(str, enum.Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
    CUSTOM = "custom"


# ---------- Users ----------
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=_uid)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now)

    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


# ---------- Datasets ----------
class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    knowledge_category = Column(String(64), default="General Knowledge", index=True)
    language = Column(String(32), default="auto")
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    user = relationship("User", back_populates="datasets")
    versions = relationship("DatasetVersion", back_populates="dataset",
                            cascade="all, delete-orphan", order_by="DatasetVersion.version")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id = Column(String, primary_key=True, default=_uid)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    notes = Column(Text, default="")
    # Aggregated analysis (real, computed at ingestion)
    num_files = Column(Integer, default=0)
    num_documents = Column(Integer, default=0)
    raw_chars = Column(Integer, default=0)
    raw_bytes = Column(Integer, default=0)
    num_tokens = Column(Integer, default=0)           # after tokenization
    estimated_words = Column(Integer, default=0)       # whitespace-token estimate
    unique_vocab_tokens = Column(Integer, default=0)
    tokenizer_version_id = Column(String, ForeignKey("tokenizer_versions.id"), nullable=True)
    train_tokens = Column(Integer, default=0)
    val_tokens = Column(Integer, default=0)
    analysis = Column(JSON, default=dict)             # unicode coverage, dedup stats, etc.
    created_at = Column(DateTime, default=_now)

    dataset = relationship("Dataset", back_populates="versions")
    files = relationship("DatasetFile", back_populates="version", cascade="all, delete-orphan")
    tokenizer_version = relationship("TokenizerVersion")


class DatasetFile(Base):
    __tablename__ = "dataset_files"
    id = Column(String, primary_key=True, default=_uid)
    version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(16), nullable=False)    # txt|json|jsonl|csv|md
    storage_path = Column(String(1024), nullable=False)  # relative to STORAGE_DIR
    size_bytes = Column(Integer, default=0)
    num_documents = Column(Integer, default=0)
    num_tokens = Column(Integer, default=0)
    checksum = Column(String(64), default="")
    created_at = Column(DateTime, default=_now)

    version = relationship("DatasetVersion", back_populates="files")


# ---------- Tokenizer versions ----------
class TokenizerVersion(Base):
    __tablename__ = "tokenizer_versions"
    id = Column(String, primary_key=True, default=_uid)
    version = Column(String(32), nullable=False, index=True)   # semantic version
    vocab_size = Column(Integer, nullable=False)
    num_merges = Column(Integer, default=0)
    storage_path = Column(String(1024), nullable=False)        # JSON file path
    training_tokens = Column(Integer, default=0)
    unicode_coverage = Column(JSON, default=dict)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now)


# ---------- Training jobs ----------
class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    # Queue of dataset versions to train on, in order
    dataset_version_ids = Column(JSON, default=list)  # ordered list
    current_dataset_index = Column(Integer, default=0)
    # Parent model to continue from (continual learning). None = from scratch.
    parent_model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    base_model_config = Column(JSON, default=dict)    # ModelConfig if from scratch
    # Training spec
    mode = Column(String(16), default="balanced")
    device = Column(String(8), default="auto")
    hyperparams = Column(JSON, default=dict)
    tokenizer_version_id = Column(String, ForeignKey("tokenizer_versions.id"), nullable=True)
    # Replay config (continual learning / forgetting prevention)
    replay_ratio = Column(Float, default=0.0)
    replay_dataset_version_ids = Column(JSON, default=list)
    # Resulting model version (candidate)
    output_model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    # Runtime state
    status = Column(String(16), default=JobStatus.QUEUED.value, index=True)
    worker_id = Column(String, nullable=True, index=True)
    progress_pct = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_loss = Column(Float, nullable=True)
    best_val_loss = Column(Float, nullable=True)
    best_val_perplexity = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    final_val_loss = Column(Float, nullable=True)
    final_perplexity = Column(Float, nullable=True)
    tokens_processed = Column(Integer, default=0)
    tokens_per_sec = Column(Float, default=0.0)
    elapsed_seconds = Column(Float, default=0.0)
    retention_score = Column(Float, nullable=True)
    evaluation_score = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    steps = relationship("TrainingStep", back_populates="job",
                         cascade="all, delete-orphan", order_by="TrainingStep.id")
    output_model_version = relationship("ModelVersion", foreign_keys=[output_model_version_id])
    parent_model_version = relationship("ModelVersion", foreign_keys=[parent_model_version_id])


class TrainingStep(Base):
    """Per-step metric samples written by the worker (the real-time feed source)."""
    __tablename__ = "training_steps"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("training_jobs.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    epoch = Column(Integer, default=0)
    dataset_index = Column(Integer, default=0)
    loss = Column(Float, nullable=False)
    learning_rate = Column(Float, default=0.0)
    tokens_processed = Column(Integer, default=0)
    tokens_per_sec = Column(Float, default=0.0)
    grad_norm = Column(Float, nullable=True)
    memory_mb = Column(Float, nullable=True)
    is_validation = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now, index=True)

    job = relationship("TrainingJob", back_populates="steps")


# ---------- Models ----------
class Model(Base):
    """A model family/lineage root (e.g. 'my-llm'). Versions hang off it."""
    __tablename__ = "models"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=_now)

    versions = relationship("ModelVersion", back_populates="model",
                            cascade="all, delete-orphan", order_by="ModelVersion.version")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(String, primary_key=True, default=_uid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    version = Column(String(32), nullable=False)      # e.g. 1.0.0
    parent_model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    architecture = Column(JSON, nullable=False)        # ModelConfig.to_dict()
    parameter_count = Column(Integer, default=0)
    vocab_size = Column(Integer, default=0)
    tokenizer_version_id = Column(String, ForeignKey("tokenizer_versions.id"), nullable=True)
    training_dataset_version_ids = Column(JSON, default=list)
    training_tokens = Column(Integer, default=0)
    status = Column(String(16), default=ModelStatus.TRAINING.value, index=True)
    checkpoint_path = Column(String(1024), nullable=True)
    evaluation_metrics = Column(JSON, default=dict)
    retention_metrics = Column(JSON, default=dict)
    growth_score = Column(Float, default=0.0)
    # Promotion gates
    promotion_passed = Column(Boolean, default=False)
    promotion_reason = Column(Text, default="")
    created_at = Column(DateTime, default=_now)

    model = relationship("Model", back_populates="versions")
    parent = relationship("ModelVersion", remote_side="ModelVersion.id", uselist=False)
    checkpoints = relationship("Checkpoint", back_populates="model_version",
                               cascade="all, delete-orphan")
    tokenizer_version = relationship("TokenizerVersion")


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    id = Column(String, primary_key=True, default=_uid)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False, index=True)
    job_id = Column(String, ForeignKey("training_jobs.id"), nullable=True)
    step = Column(Integer, default=0)
    epoch = Column(Integer, default=0)
    path = Column(String(1024), nullable=False)
    val_loss = Column(Float, nullable=True)
    val_perplexity = Column(Float, nullable=True)
    metrics = Column(JSON, default=dict)
    is_latest = Column(Boolean, default=False)
    is_best = Column(Boolean, default=False)
    is_previous_production = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now)

    model_version = relationship("ModelVersion", back_populates="checkpoints")


# ---------- Memory / Knowledge (RAG) ----------
class Document(Base):
    """A source document in the knowledge base (can come from datasets)."""
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    source = Column(String(255), default="manual")    # manual|dataset|correction
    source_ref = Column(String(255), default="")
    title = Column(String(512), default="")
    content = Column(Text, default="")
    language = Column(String(32), default="auto")
    knowledge_category = Column(String(64), default="General Knowledge")
    created_at = Column(DateTime, default=_now)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True, default=_uid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=True)   # float32 numpy bytes
    embedding_dim = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_now)

    document = relationship("Document", back_populates="chunks")


class Memory(Base):
    """Explicit conversational / factual memory entries."""
    __tablename__ = "memories"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    kind = Column(String(32), default="fact")        # fact|preference|correction|note
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_now)


class Correction(Base):
    """User-submitted corrections (Q, incorrect A, correct A). Becomes eval/training data."""
    __tablename__ = "corrections"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    incorrect_answer = Column(Text, default="")
    correct_answer = Column(Text, nullable=False)
    context = Column(Text, default="")
    status = Column(String(16), default="pending")   # pending|used_for_eval|used_for_training
    created_at = Column(DateTime, default=_now)


# ---------- Conversations / chat ----------
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), default="New Conversation")
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation",
                            cascade="all, delete-orphan", order_by="Message.id")
    model_version = relationship("ModelVersion")


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=_uid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)        # user|assistant|system
    content = Column(Text, nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    generation_settings = Column(JSON, default=dict)
    tokens_generated = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=_now)

    conversation = relationship("Conversation", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message", cascade="all, delete-orphan")


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(String, primary_key=True, default=_uid)
    message_id = Column(String, ForeignKey("messages.id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)         # -1|0|1
    comment = Column(Text, default="")
    created_at = Column(DateTime, default=_now)

    message = relationship("Message", back_populates="feedback")


# ---------- Evaluations ----------
class Evaluation(Base):
    """A named evaluation suite (benchmark or custom tests)."""
    __tablename__ = "evaluations"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    kind = Column(String(32), default="custom")      # benchmark|custom|retention
    source_dataset_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=True)
    created_at = Column(DateTime, default=_now)

    tests = relationship("EvaluationTest", back_populates="evaluation", cascade="all, delete-orphan")


class EvaluationTest(Base):
    __tablename__ = "evaluation_tests"
    id = Column(String, primary_key=True, default=_uid)
    evaluation_id = Column(String, ForeignKey("evaluations.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    expected_answer = Column(Text, default="")
    criteria = Column(Text, default="")              # e.g. contains, similarity
    source_doc_id = Column(String, ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=_now)

    evaluation = relationship("Evaluation", back_populates="tests")
    results = relationship("EvaluationResult", back_populates="test", cascade="all, delete-orphan")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(String, primary_key=True, default=_uid)
    test_id = Column(String, ForeignKey("evaluation_tests.id"), nullable=False, index=True)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False, index=True)
    response = Column(Text, default="")
    score = Column(Float, default=0.0)               # 0..1
    passed = Column(Boolean, default=False)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=_now)

    test = relationship("EvaluationTest", back_populates="results")
    model_version = relationship("ModelVersion")


# ---------- Workers & schedules ----------
class Worker(Base):
    __tablename__ = "workers"
    id = Column(String, primary_key=True, default=_uid)
    hostname = Column(String(255), default="")
    device = Column(String(16), default="cpu")
    device_name = Column(String(255), default="")
    status = Column(String(16), default="idle")      # idle|busy|dead
    current_job_id = Column(String, ForeignKey("training_jobs.id"), nullable=True)
    last_heartbeat = Column(DateTime, default=_now)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_now)


class Schedule(Base):
    __tablename__ = "schedules"
    id = Column(String, primary_key=True, default=_uid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), default="Scheduled Learning")
    enabled = Column(Boolean, default=False)
    interval_seconds = Column(Integer, default=3600)
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True, index=True)
    auto_promote = Column(Boolean, default=False)
    config = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_now)


# ---------- Indexes ----------
# (Most columns are indexed inline via index=True above. Additional composite
# indexes are declared here with unique names to avoid SQLAlchemy auto-name clashes.)
Index("ix_training_jobs_status_created", TrainingJob.status, TrainingJob.created_at)
Index("ix_training_steps_job_step", TrainingStep.job_id, TrainingStep.step)
Index("ix_doc_chunks_doc", DocumentChunk.document_id)
