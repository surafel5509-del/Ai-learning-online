# Deployment

## Option 1: Docker Compose (recommended)

```bash
cp .env.example .env
# edit POSTGRES_PASSWORD and SECRET_KEY
docker compose up --build -d
```

Services:
- `db` (PostgreSQL 16) — metadata + pgvector-ready
- `redis` — task queue coordination
- `api` (FastAPI) — port 8000
- `worker` — training worker
- `web` (Vite preview) — port 5173

The API also serves the built SPA at `/`, so you can use `http://localhost:8000/`
directly and skip the `web` container if desired.

## Option 2: Local dev (no Docker)

See the [README](../README.md#quick-start-local-no-docker).

## GPU deployment

For GPU training, build the worker from `Dockerfile.worker.gpu` (based on the official
PyTorch CUDA image) and run with GPU access:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
The platform auto-detects CUDA at runtime; no code changes are needed.

## Configuration (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///data/ai_platform.db` | DB connection (use Postgres in prod) |
| `REDIS_URL` | `redis://localhost:6379/0` | Queue coordination |
| `STORAGE_DIR` | `./data` | File storage root |
| `SECRET_KEY` | — | JWT signing secret (required) |
| `CORS_ORIGINS` | `*` | Allowed origins |
| `DEFAULT_DEVICE` | `CPU` | Default device for new jobs |

## PostgreSQL with pgvector (production)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The schema stores embeddings; vector similarity search uses cosine distance. SQLite is
used for dev and falls back to in-memory/row scan for retrieval.

## Scaling

- Run multiple `worker` containers (CPU or GPU) — they poll the same queue.
- Put the API behind a reverse proxy (nginx/traefik) with TLS.
- Use S3-compatible object storage for large datasets by mounting/pointing `STORAGE_DIR`
  accordingly.
- For distributed GPU training, extend `services/trainer` with DDP (future work).

## Failure recovery

- Browser close: no effect — state is server-side.
- API restart: jobs persist in DB; reconnect SSE.
- Worker crash: job marked FAILED; resume from last checkpoint.
- GPU crash: same as worker crash; CPU fallback available.

## Health check

```
GET /api/health  →  {"status": "ok"}
```
