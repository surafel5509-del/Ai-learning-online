# Architecture

## Overview

The platform is a monorepo with a clear separation between the browser (control
interface) and the backend (training + persistent storage).

```
Browser (React SPA)
    │  REST + Server-Sent Events
    ▼
FastAPI Backend  ──►  PostgreSQL (metadata, models, metrics, memory, pgvector)
    │  enqueues jobs        ──►  Object/file storage (datasets, tokenizers, checkpoints)
    ▼
Training Queue (DB-backed)
    │
    ▼
CPU/GPU Worker  ──►  Real Transformer (PyTorch)  ──►  Checkpoints
    │  writes live step metrics + SSE events
    ▼
Evaluator  ──►  Retention Test  ──►  Model Registry  ──►  Production Model
```

## Components

### `packages/model` — Real Transformer
- Decoder-only causal LM (`TransformerLM`).
- RoPE rotary positional embeddings, RMSNorm, SwiGLU FFN, Grouped-Query Attention.
- KV cache for fast autoregressive generation (`prefill_kv_caches` + decode).
- Configurable: vocab size, hidden size, layers, heads, KV heads, context length,
  intermediate size, dtype, RoPE theta/base.
- Mixed-precision (AMP) support when CUDA is available.

### `packages/tokenizer` — Versioned BPE
- Byte-level BPE trained on demand; vocab ≥ 256 (byte alphabet).
- Full Unicode coverage including Amharic/Ethiopic (UTF-8 byte-level preserves all codepoints).
- Versioned: each trained tokenizer gets a version; models record the tokenizer version
  they were trained with. The tokenizer is **not** changed after every small job.

### `packages/shared`
- `config` — settings (DB URL, storage dir, secrets) from env.
- `database` — SQLAlchemy engine/session, `Base`, all ORM models.
- `db_models` — full schema (users, datasets, versions, files, jobs, steps, models,
  model_versions, checkpoints, memories, documents, embeddings, conversations, messages,
  feedback, evaluations, results, workers, schedules).
- `hardware` — real CPU/GPU detection + live utilization (psutil/torch).
- `security` — password hashing (bcrypt), JWT auth, file-type allowlist.
- `dataset` — parsing (TXT/JSON/JSONL/CSV/MD), cleaning, dedup, analysis, tokenization,
  train/val split, binary token storage.
- `metrics` — perplexity, knowledge retention score, AI growth score, word estimates.

### `services/trainer`
- Real training loop: forward + cross-entropy loss + backward + optimizer step.
- Gradient accumulation, warmup + cosine LR schedule, gradient clipping, weight decay.
- Mixed precision on GPU.
- Multi-dataset queue: checkpoint → evaluate → record metrics → lineage → next dataset.
- Replay data mixing for continual learning.
- Checkpointing: weights + optimizer + scheduler + step + metrics.

### `services/inference`
- KV-cache generation with temperature, top-p, top-k, repetition penalty.
- Streaming (token-by-token) generation.
- Model comparison (two models, same prompt).

### `services/evaluator`
- Auto-generate benchmark Q/A from dataset facts.
- Custom user tests (question + expected answer + criteria).
- Run tests against current/candidate/previous models.
- **Knowledge Retention Score**: compare before vs after test scores.

### `services/memory` + `services/retrieval`
- Explicit memory store.
- Document → chunk → embedding → vector search (pgvector / cosine).
- RAG: query → embed → retrieve → context → model → response.

### `services/scheduler`
- Scheduled learning cycles (30m / 1h / 6h / daily / custom).
- Auto-learning: new data → validate → queue → train → replay → evaluate → retention →
  candidate → promote if quality gates pass.

### `apps/api` — FastAPI
- 11 routers: auth, datasets, tokenizers, training, models, evaluations, memory,
  inference, chat, dashboard, schedules.
- JWT auth on all protected routes; user isolation.
- SSE endpoints for live training progress and streaming generation.
- Serves the built SPA.

### `apps/web` — React + Vite + TS
- 16 pages: Dashboard, Chat, AI Growth, Training, Training Queue, Datasets, Knowledge,
  Vocabulary, Memory, Evaluations, Models, Checkpoints, Workers, Performance, Settings,
  Model Test Lab.
- Real-time updates via SSE / polling from real API data.

## Data flow: a training job

1. User uploads/pastes text → `datasets` validates, cleans, dedups, analyzes, tokenizes,
   splits → stores a version + binary token file. Original is never destroyed.
2. User requests a training plan → backend computes steps/epochs from token counts.
3. User starts a job → row in `training_jobs` (status QUEUED).
4. Worker polls the queue, loads the model checkpoint (continual learning: parent model),
   mixes replay data, runs the loop, writes `training_steps` rows + emits SSE events.
5. After each dataset: checkpoint → evaluate on benchmarks + custom tests → compute
   retention vs the parent model → record metrics → continue.
6. Result becomes a **CANDIDATE** model version. Promotion to PRODUCTION only happens if
   quality gates (retention threshold, no severe regression) pass — controlled by the
   scheduler/auto-learning config.

## Persistence

All state lives in PostgreSQL (SQLite for dev): datasets, versions, jobs, steps, models,
model versions, checkpoints, metrics, conversations, memories, evaluations, schedules.
The browser closing never erases anything; jobs survive API/worker restarts and resume
from checkpoints where possible.
