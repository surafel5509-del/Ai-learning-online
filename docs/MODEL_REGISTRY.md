# Model Registry

Every model is versioned and tracked. The production model is never overwritten.

## Model fields

| Field | Description |
|-------|-------------|
| `id` | Model ID (architecture family) |
| `version` | Semantic version, e.g. `1.0.0`, `1.1.0` |
| `parent_model_version_id` | Parent model version (for continual learning lineage) |
| `architecture` | Model config (layers, heads, hidden, etc.) |
| `parameter_count` | Real parameter count from `model.num_parameters()` |
| `tokenizer_version` | Tokenizer version trained with |
| `training_datasets` | Datasets used in this version |
| `dataset_versions` | Specific dataset versions |
| `training_tokens` | Total tokens processed |
| `checkpoints` | Checkpoint references |
| `evaluation_metrics` | Benchmark + custom test scores |
| `retention_metrics` | Knowledge retention score + breakdown |
| `creation_date` | When created |
| `status` | Lifecycle status (below) |

## Statuses

```
TRAINING → CANDIDATE → VALIDATED → PRODUCTION
                ↘ FAILED
PRODUCTION → ARCHIVED   (when superseded)
```

- **TRAINING** — currently being trained.
- **CANDIDATE** — training finished; awaiting evaluation/gates.
- **VALIDATED** — passed quality gates; eligible for promotion.
- **PRODUCTION** — the live model used by Chat / Test Lab / inference by default.
- **ARCHIVED** — superseded by a newer production model; retained for rollback.
- **FAILED** — training failed or quality gates violated; not promoted.

## Lineage

Each version records its parent, forming a lineage tree. The AI Growth dashboard and
Models page render this lineage so you can trace how the model evolved.

## Promotion & rollback

- `POST /models/{mv_id}/promote` — promote a validated candidate to PRODUCTION; the
  previous production model is automatically ARCHIVED (not deleted).
- `POST /models/{mv_id}/rollback` — roll production back to a prior version/checkpoint.

## Checkpoints

Each model version references checkpoints (latest / best / previous-production) storing
weights, optimizer, scheduler, step, epoch, config, tokenizer version, dataset position,
and metrics. See `apps/api/routers/dashboard.py` (`/dashboard/checkpoints`).
