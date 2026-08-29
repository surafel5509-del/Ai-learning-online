# AGENTS.md — AI Platform Repository Memory

## Project: Online Continual-Learning AI Platform
A real, trainable decoder-only Transformer LM platform with persistent backend,
CPU/GPU training, continual learning, model versioning, memory/RAG, evaluation,
and a real-time growth dashboard. **No fake metrics.**

## Environment (audited 2026-08-29)
- OS: Linux container, 4 CPU cores, ~15GB RAM, no swap.
- Python 3.13.14 at /usr/local/bin/python
- Node/npm/npx present. Docker present. **No** postgres/redis/psql binaries on host.
- **No GPU / no CUDA.** CPU training path must work standalone.
- torch was NOT installed; installing CPU wheel in background.
- Repo was **empty** (only `.git`).

## Architecture decisions
- Backend: **FastAPI + SQLAlchemy + SQLite** (file-based, zero external deps for dev;
  schema is portable to PostgreSQL by switching the DSN). DB-backed job queue avoids
  needing Redis/Celery. Workers poll the DB for jobs.
- Model: **packages/model** — decoder-only Transformer with RoPE, RMSNorm, SwiGLU,
  GQA, KV cache, optional AMP. Real PyTorch causal-LM training.
- Tokenizer: **packages/tokenizer** — real trainable byte-level BPE (GPT-2 style),
  versioned, with full Unicode incl. Amharic/Ethiopic coverage.
- Frontend: **React + Vite + TypeScript** SPA (no Next.js SSR needed; simpler deploy).
  Communicates via REST + Server-Sent Events for live training updates.
- Hardware: real detection via torch and psutil; never faked. GPU path guarded by
  torch.cuda.is_available().

## Key conventions
- Every public metric must be computed from real data; estimates labeled "estimate"
  or "composite".
- Never overwrite production model — candidates first, then gated promotion.
- Monorepo layout under: apps/{web,api} services/* packages/* training/* tests/* docs/*.
- Run backend tests from repo root with: `python -m pytest tests -q`
- Start API: `python -m apps.api.main` (or uvicorn). Worker: `python -m apps.api.worker`.

## Notes
- See docs/ for detailed docs. Run order: DB → datasets → model → training →
  checkpoints → registry → evaluation → continual learning → dashboard → test lab →
  chat → memory/RAG → scheduler → monitoring → tests.

## Verified state (2026-08-29)
- Full stack verified end-to-end via `scripts/e2e_test.py`: register → train BPE
  tokenizer → create dataset + paste version → training plan → training job (real loss
  decrease, real tokens/sec) → evaluation (retention score) → promote to production →
  generate (real model output) → benchmark eval → growth dashboard.
- Frontend builds clean (`tsc --noEmit` + `vite build`); all 16 pages render; served by
  the API at `/` when `apps/web/dist` exists.
- Test suite: **51 tests pass** (`python -m pytest tests -q`) covering tokenizer (incl.
  Amharic/Ethiopic), Transformer forward/backward, KV cache, training loss decrease,
  checkpoint save/load/resume, evaluation & retention scoring, dataset pipeline,
  memory/RAG vector search, and the FastAPI API.
- API exposes 76 routes across 11 routers. SSE for live training + streaming generation.
- Docker: `docker-compose.yml` (db/redis/api/worker/web) + `Dockerfile.api`,
  `Dockerfile.web`, `Dockerfile.worker.gpu`. `.env.example` + `.gitignore` added.
- Docs: README, ARCHITECTURE, TRAINING, CONTINUAL_LEARNING, MODEL_REGISTRY, DATASETS,
  DEPLOYMENT, API, SECURITY.
- Added endpoints: `/dashboard/hardware`, `/schedules/active`, simplified `/schedules`
  POST for the Settings form. Fixed `_sched_dict` reference in dashboard.py.

## Gotchas
- `allowed_file_type` lives in `packages/shared/security.py`, NOT `dataset.py`.
- `analyze_documents(docs, tokenizer)` requires a tokenizer argument.
- `EvalTest` uses `expected_answer` (not `expected`). `ai_growth_score` uses param names
  `evaluation/retention/validation/vocab_coverage/training_progress/task_performance`.
- embed dim is 256 (`EMBED_DIM` in services/memory/memory.py).
- `TransformerLM.forward` uses `kv_caches` (list) + `start_pos`; use
  `prefill_kv_caches()` then decode. There is no `clear_kv_cache`/`use_cache` flag.
