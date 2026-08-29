# Continual Learning

The model supports repeated learning without simply overwriting the production model.

```
Model v1  →  new data  →  train  →  evaluate  →  Model v1.1
Model v1.1  →  more data  →  train  →  evaluate  →  Model v1.2
```

## How it works

1. A new training job always starts from the **parent model** (the current production or
   selected checkpoint) — weights are loaded, not reinitialized.
2. New training data is mixed with **replay data** drawn from previously learned
   datasets, so old patterns stay present during optimization.
3. After training, the model is evaluated against:
   - **Old knowledge tests** (benchmarks derived from prior datasets),
   - **Validation** on held-out splits,
   - **Custom user tests**.
4. A **Knowledge Retention Score** is computed by comparing test scores **before** the
   training (parent model) vs **after** (candidate model).

## Knowledge Retention Score

For each test `t` with before-score `b_t` and after-score `a_t`:

```
retention_t = max(0, a_t) / max(eps, b_t)      if b_t > 0
retention_t = 1.0                                if b_t == 0 and a_t == 0
retention_t = 0.0                                if b_t == 0 and a_t > 0  (cannot verify)

Retention Score = mean(retention_t) over all tests
```

- `1.0` = no forgetting.
- Low scores indicate catastrophic forgetting and **flag the candidate**.

## Quality gates & promotion

A candidate model is promoted to PRODUCTION only when:
- retention score ≥ configured threshold (default `0.7`),
- no severe regression on old-knowledge tests,
- (optional, auto-learning) validation loss did not worsen beyond a margin.

If gates are violated, the candidate is marked FAILED/ARCHIVED and the previous
production model stays. Rollback to any checkpoint is always available.

## Replay data

Replay samples are drawn proportionally from each previously learned dataset's token
stream and interleaved with new data during batching. This is the primary mechanism that
**strongly reduces** (not mathematically eliminates) catastrophic forgetting.

## Never overwrite production

Production is never overwritten in place. Every change creates a new model version with:
parent pointer, training datasets, dataset versions, training tokens, checkpoints,
evaluation metrics, retention metrics, creation date, and a status. See
[MODEL_REGISTRY.md](MODEL_REGISTRY.md).

## Regression testing

Each promotion cycle records before/after scores historically, so you can observe whether
the model is actually improving or regressing over versions.
