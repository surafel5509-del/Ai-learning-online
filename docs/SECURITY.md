# Security

## Authentication & authorization
- Passwords hashed with **bcrypt** via passlib; raw passwords are never stored.
- **JWT** bearer tokens (`python-jose`) for API auth; short-lived access tokens.
- Every protected route requires a valid token; user isolation enforced (each user only
  sees their own datasets, jobs, models, conversations, memories).

## File upload safety
- Extension **allowlist**: `.txt .json .jsonl .csv .md` only.
- Size limits enforced.
- Files stored **outside** the web root under `STORAGE_DIR` — never served directly.
- Uploaded files are **never executed**. They are only parsed as text/data.

## Input validation
- Pydantic schemas validate all request bodies.
- SQL injection mitigated by SQLAlchemy parameterized queries (no raw string SQL).
- Path traversal prevented by validated identifiers + storage root confinement.

## Secrets
- `SECRET_KEY` read from environment; never hardcoded.
- See `.env.example`. Rotate secrets in production.
- Secrets are masked in logs and never echoed (see the platform's secret-handling policy).

## Rate limiting & exposure
- Designed to sit behind a reverse proxy (nginx/traefik) for TLS termination and rate
  limiting in production.
- CORS configurable via `CORS_ORIGINS` (default permissive for dev; restrict in prod).
- Detailed errors are not leaked to clients in production.

## No-fake policy
- Hardware stats (CPU/GPU/VRAM) come from real system calls (psutil, `torch.cuda`).
- Training metrics come from real forward/backward passes.
- Evaluation scores come from real model generation + criteria checks.
- The **AI Growth Score** is explicitly labelled a composite estimate; its components are
  shown. It never increases merely because training time increased.

## Corrections & memory
- Corrections (Q / wrong answer / correct answer) are stored as structured data and may
  later become training/evaluation data. A single correction **never** immediately
  modifies model weights.
- RAG retrieval augments generation; it does not alter weights.

## Failure recovery
- State is persisted in the database, so API/worker restarts, browser closes, and crashes
  never erase datasets, jobs, models, checkpoints, or conversations. Jobs resume from
  checkpoints where possible.
