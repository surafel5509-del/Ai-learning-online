# AI Continual-Learning Platform

A real, end-to-end online AI platform that trains a genuine decoder-only Transformer
language model, supports continual learning, and lets you watch the model grow in real
time through a professional dashboard. This is **not** a chatbot wrapper and **not** fake
training — every metric (loss, perplexity, tokens/sec, GPU/CPU usage, evaluation,
retention) is computed from real system data.

> **AI limitation:** This is an independent Transformer-based system inspired by publicly
> known ML techniques. A small model cannot become a frontier model merely by repeated
> training. The platform is designed to scale later through larger data, larger models,
> more compute, distributed GPU training, and improved evaluation.

---

## What it does

```
DATA  →  TRAIN  →  CHECKPOINT  →  EVALUATE  →  RETENTION TEST  →  MODEL VERSION
   →  PROMOTION  →  MORE DATA  →  MORE TRAINING  →  MORE EVALUATION  →  CONTINUOUS IMPROVEMENT
```

- **Real Transformer**: decoder-only, RoPE, RMSNorm, SwiGLU, Grouped-Query Attention,
  KV cache, mixed precision — implemented in PyTorch (`packages/model`).
- **Real trainable tokenizer**: versioned byte-level BPE with full Unicode (including
  Amharic/Ethiopic) support (`packages/tokenizer`).
- **CPU & GPU training**: auto-detects CUDA, monitors real VRAM/utilization, falls back
  to optimized CPU training. Never fakes hardware stats.
- **Continual learning**: replay data + old-knowledge tests + validation; checkpoints
  and model versioning; candidate models must pass quality gates before promotion.
- **Forgetting prevention**: computes a real **Knowledge Retention Score** comparing
  before/after evaluation; flags candidates when old knowledge degrades.
- **Real-time dashboard**: WebSocket/SSE live training progress, loss, perplexity,
  tokens/sec, ETA, hardware usage.
- **AI Growth dashboard**: model growth over time, historical charts, growth score.
- **Model Test Lab**: prompt the trained model with streaming, temperature, top-p,
  top-k, repetition penalty; model comparison.
- **Memory + RAG**: explicit memory, vector knowledge (pgvector), conversation history —
  retrieval augments generation without forcing everything into weights.
- **Chat**: streaming, memory, RAG, model selection, feedback, corrections, export.
- **Persistent backend**: PostgreSQL (SQLite fallback), datasets, jobs, models,
  checkpoints, metrics, conversations — closing the browser never erases anything.

---

## Repository structure

```
ai-platform/
├── apps/
│   ├── api/            # FastAPI backend (routers, auth, worker, schemas)
│   └── web/            # React + Vite + TypeScript frontend (16 pages)
├── services/
│   ├── trainer/        # CPU/GPU training loop, checkpoints, replay, continual learning
│   ├── inference/      # KV-cache generation, streaming, comparison
│   ├── evaluator/      # benchmarks from datasets, custom tests, retention scoring
│   ├── scheduler/      # scheduled + automatic learning cycles
│   ├── memory/         # explicit memory + vector embeddings + RAG retrieval
│   └── retrieval/      # vector search helpers
├── packages/
│   ├── model/          # TransformerLM, RoPE, RMSNorm, SwiGLU, GQA, KV cache
│   ├── tokenizer/      # versioned byte-level BPE tokenizer
│   └── shared/         # config, DB models, hardware, security, dataset, metrics
├── training/
│   ├── pretraining/
│   ├── finetuning/
│   └── continual_learning/
├── infrastructure/     # deployment notes
├── tests/              # 51 tests: tokenizer, model, training, eval, dataset, memory, API
├── scripts/            # e2e test harness
├── docs/               # architecture & feature docs
├── Dockerfile.api      # backend image
├── Dockerfile.web      # frontend image
├── Dockerfile.worker.gpu  # GPU worker image
├── docker-compose.yml  # db + redis + api + worker + web
└── requirements.txt
```

---

## Quick start (local, no Docker)

### Prerequisites
- Python 3.11+
- Node.js 20+
- (Optional) a CUDA GPU for GPU training; CPU always works.

### 1. Backend

```bash
cd /workspace/project
pip install -r requirements.txt

# start the API
python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# in a second terminal, start the training worker
python -m apps.api.worker
```

The API serves the built frontend at `/` when `apps/web/dist` exists.

### 2. Frontend

```bash
cd apps/web
npm install
npm run build      # production build (served by the API at /)
# or for dev with HMR:
npm run dev        # http://localhost:5173 (proxies /api to :8000)
```

### 3. Use it

Open `http://localhost:8000/` (or `:5173` in dev mode), register an account, then:

1. **Tokenizers** → train a BPE tokenizer from sample text (or paste text directly).
2. **Datasets** → create a dataset, paste/upload text to create a version.
3. **Training** → generate a training plan, start a job, watch it live.
4. **Evaluations** → run benchmark tests, check the retention score.
5. **Models** → promote a candidate to production.
6. **Model Test Lab** → prompt your trained model.
7. **Chat** → chat with the production model (with memory + RAG).

---

## Quick start (Docker)

```bash
cp .env.example .env   # edit secrets
docker compose up --build
```

- API:        `http://localhost:8000`
- Web:        `http://localhost:5173`
- PostgreSQL: `localhost:5432`
- Redis:      `localhost:6379`

For GPU training, build the worker with `Dockerfile.worker.gpu` and run with
`--gpus all` (requires the NVIDIA Container Toolkit).

---

## Documentation

| Doc | Topic |
|-----|-------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture & data flow |
| [TRAINING.md](docs/TRAINING.md) | Training modes, hyperparameters, CPU/GPU |
| [CONTINUAL_LEARNING.md](docs/CONTINUAL_LEARNING.md) | Replay, retention, promotion gates |
| [MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md) | Model versioning & statuses |
| [DATASETS.md](docs/DATASETS.md) | Dataset pipeline & formats |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker & production deployment |
| [API.md](docs/API.md) | REST API reference |
| [SECURITY.md](docs/SECURITY.md) | Security model |

---

## Testing

```bash
python -m pytest tests/ -q          # 51 tests
python scripts/e2e_test.py          # full HTTP end-to-end pipeline
```

Tests cover: tokenizer (incl. Amharic/Ethiopic), Transformer forward/backward, KV cache,
training loss decrease, checkpoint save/load/resume, evaluation & retention scoring,
dataset parsing/cleaning/dedup/split, memory/RAG vector search, and the FastAPI API.

---

## No fake features

Every metric is computed from real data. Where something cannot be scientifically
measured directly (e.g. the AI Growth Score), it is clearly labelled as a **composite
estimate** with its underlying components shown. See [SECURITY.md → No-fake policy](docs/SECURITY.md).
