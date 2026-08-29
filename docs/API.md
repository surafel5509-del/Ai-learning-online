# API Reference

Base URL: `http://localhost:8000`. All protected routes require
`Authorization: Bearer <jwt>`.

## Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/register` | Register `{username, password}` → `{access_token}` |
| POST | `/auth/login` | Login → `{access_token}` |
| GET | `/auth/me` | Current user |

## Tokenizers

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tokenizers/train` | Train BPE from `{texts}` or `{dataset_version_ids}`, `target_vocab_size` |
| GET | `/tokenizers` | List tokenizers |
| GET | `/tokenizers/active` | Active tokenizer |
| POST | `/tokenizers/{id}/activate` | Set active |

## Datasets

| Method | Path | Description |
|--------|------|-------------|
| GET | `/datasets` | List |
| POST | `/datasets` | Create `{name, knowledge_category}` |
| GET/DELETE | `/datasets/{id}` | Get / delete |
| GET | `/datasets/{id}/versions` | List versions |
| POST | `/datasets/{id}/versions` | Upload file version |
| POST | `/datasets/{id}/versions/paste` | Paste text version |
| GET | `/datasets/{id}/versions/{vid}/preview` | Preview documents |

## Training

| Method | Path | Description |
|--------|------|-------------|
| GET | `/training/hardware` | Real CPU/GPU detection + stats |
| GET | `/training/plan` | Compute training plan for dataset versions |
| GET | `/training/jobs` | List jobs |
| POST | `/training/jobs` | Create job |
| GET | `/training/jobs/{id}` | Job detail |
| GET | `/training/jobs/{id}/steps` | Step history |
| GET | `/training/jobs/{id}/stream` | **SSE** live progress |
| POST | `/training/jobs/{id}/control` | Pause/resume/cancel |

## Models / Registry

| Method | Path | Description |
|--------|------|-------------|
| GET | `/models` | List models |
| GET | `/models/{id}` | Model detail |
| GET | `/models/registry/all` | Full registry with versions |
| GET | `/models/versions/{mv_id}` | Version detail |
| POST | `/models/{mv_id}/promote` | Promote to production |
| POST | `/models/{mv_id}/rollback` | Rollback |
| PUT | `/models/production/{model_id}` | Set production |

## Evaluations

| Method | Path | Description |
|--------|------|-------------|
| GET | `/evaluations` | List suites |
| POST | `/evaluations/run` | Run a suite against a model version |
| GET | `/evaluations/{id}` | Suite detail |
| GET/POST | `/evaluations/{id}/tests` | List / add custom tests |
| GET | `/evaluations/{id}/results/{mv_id}` | Results for a version |

## Inference / Test Lab

| Method | Path | Description |
|--------|------|-------------|
| POST | `/inference/generate` | Generate (config: temp, top-p, top-k, rep penalty, max tokens) |
| GET/POST | `/inference/generate/stream` | **SSE** streaming generation |
| POST | `/inference/compare` | Compare two model versions on one prompt |

## Chat

| Method | Path | Description |
|--------|------|-------------|
| GET | `/chat/conversations` | List conversations |
| POST | `/chat/send` | Send message (with memory + RAG) |
| GET/POST | `/chat/send/stream` | **SSE** streaming chat |
| POST | `/chat/messages/{id}/feedback` | Feedback / correction |

## Memory / RAG

| Method | Path | Description |
|--------|------|-------------|
| GET/POST | `/memory/documents` | List / add knowledge documents |
| DELETE | `/memory/documents/{id}` | Delete |
| GET/POST | `/memory/memories` | Explicit memories |
| POST | `/memory/retrieve` | Vector retrieve for a query |
| GET/POST | `/memory/corrections` | Corrections (Q, wrong A, correct A) |

## Dashboard

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard/status` | Global AI status + live worker info |
| GET | `/dashboard/growth` | AI Growth score + components |
| GET | `/dashboard/growth/charts?range=7d` | Historical charts (24h/7d/30d/all) |
| GET | `/dashboard/knowledge` | Knowledge tracker by category |
| GET | `/dashboard/vocabulary` | Vocabulary tracker |
| GET | `/dashboard/checkpoints` | Checkpoints |
| GET | `/dashboard/workers` | Workers |
| GET | `/dashboard/performance` | Performance stats |
| GET | `/dashboard/schedules` | Schedules |
| GET | `/dashboard/hardware` | Live hardware stats |

## Schedules

| Method | Path | Description |
|--------|------|-------------|
| GET | `/schedules` | List |
| GET | `/schedules/active` | Primary schedule settings |
| POST | `/schedules` | Create/update primary schedule |
| PATCH/DELETE | `/schedules/{id}` | Update / delete |

## Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | `{"status":"ok"}` |
