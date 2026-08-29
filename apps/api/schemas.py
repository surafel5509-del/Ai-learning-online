"""Pydantic schemas for API request/response validation."""
from __future__ import annotations

from typing import Optional, Any
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    email: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    is_admin: bool = False


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    knowledge_category: str = "General Knowledge"
    language: str = "auto"


class DatasetVersionCreate(BaseModel):
    notes: str = ""
    deduplicate: bool = True


class PasteText(BaseModel):
    text: str = Field(min_length=1)
    filename: str = "pasted.txt"


class TrainingJobCreate(BaseModel):
    dataset_version_ids: list[str]
    parent_model_version_id: Optional[str] = None
    base_model_config: Optional[dict] = None
    mode: str = "balanced"
    device: str = "auto"
    hyperparams: Optional[dict] = None
    tokenizer_version_id: Optional[str] = None
    replay_ratio: float = 0.0
    replay_dataset_version_ids: list[str] = []


class TrainingJobUpdate(BaseModel):
    action: str  # pause|resume|cancel


class EvaluationCreate(BaseModel):
    name: str
    kind: str = "custom"
    source_dataset_version_id: Optional[str] = None


class EvaluationTestCreate(BaseModel):
    question: str
    expected_answer: str = ""
    criteria: str = "contains"


class RunEvaluationRequest(BaseModel):
    evaluation_id: str
    model_version_id: str
    max_new_tokens: int = 64
    temperature: float = 0.3


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    model_version_id: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.15
    use_rag: bool = True


class GenerateRequest(BaseModel):
    prompt: str
    model_version_id: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.15
    do_sample: bool = True


class CompareRequest(BaseModel):
    prompt: str
    model_version_id_a: str
    model_version_id_b: str
    max_new_tokens: int = 128
    temperature: float = 0.3


class CorrectionCreate(BaseModel):
    question: str
    incorrect_answer: str = ""
    correct_answer: str
    context: str = ""


class MemoryCreate(BaseModel):
    kind: str = "fact"
    content: str
    metadata: dict = {}


class DocumentCreate(BaseModel):
    title: str = ""
    content: str
    knowledge_category: str = "General Knowledge"
    language: str = "auto"


class ScheduleCreate(BaseModel):
    name: str = "Scheduled Learning"
    enabled: bool = False
    interval_seconds: int = 3600
    auto_promote: bool = False
    config: dict = {}


class ModelCreate(BaseModel):
    name: str
    description: str = ""
    base_model_config: Optional[dict] = None


class PromoteRequest(BaseModel):
    model_version_id: str
