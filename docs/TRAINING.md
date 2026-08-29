# Training

Training is **real**: a genuine PyTorch forward/backward pass over the Transformer with
cross-entropy language-modeling loss. Every reported number (loss, perplexity,
tokens/sec, step, ETA, hardware usage) is computed from the actual run.

## Training modes

| Mode | Use case | Behavior |
|------|----------|----------|
| **FAST** | Quick experimentation | Fewer epochs, smaller batches, minimal replay/eval |
| **BALANCED** | Quality/performance tradeoff | Moderate epochs, replay, validation each epoch |
| **DEEP** | Best quality | More epochs, full replay, evaluation + retention tests |
| **CUSTOM** | User-controlled | All hyperparameters set manually |

## Configurable hyperparameters (CUSTOM)

- `learning_rate`
- `batch_size`
- `gradient_accumulation_steps`
- `epochs`
- `sequence_length` (max context)
- `warmup_steps` / `warmup_ratio`
- `scheduler` (cosine)
- `weight_decay`
- `gradient_clipping`
- `checkpoint_frequency`

## CPU and GPU

- **AUTO**: detect CUDA; use GPU if available, else CPU.
- **GPU**: detect CUDA, GPU model, VRAM; monitor utilization; use AMP mixed precision.
- **CPU**: detect cores/threads; optimize dataloading; gradient accumulation for
  memory efficiency.

Hardware stats (`/training/hardware` and `/dashboard/hardware`) come from real system
calls (psutil for CPU, `torch.cuda` for GPU). They are never fabricated.

## Gradient accumulation

When the effective batch is larger than memory allows, micro-batches are accumulated and
the optimizer steps once per `gradient_accumulation_steps`. Loss is scaled accordingly.

## Learning rate schedule

Linear warmup from ~0 to `learning_rate` over `warmup_steps`, then cosine decay to a
small fraction of the peak LR. This stabilizes continual-learning fine-tuning.

## Checkpoints

Saved at `checkpoint_frequency` and at the end of each dataset:
- model weights
- optimizer state
- scheduler state
- training step + epoch
- configuration
- tokenizer version
- dataset position
- metrics

Kept: latest, best (by validation loss), previous production. Support resume, rollback,
export, import.

## Multi-dataset training

Queue many dataset versions:

```
Dataset A → checkpoint → evaluate → record → lineage
Dataset B → checkpoint → evaluate → record → lineage
Dataset C → ...
```

The browser may close without destroying the job; the worker continues and the job state
is persisted. Closing the browser never erases training state.

## Real-time dashboard

Live training progress is streamed via Server-Sent Events (`/training/jobs/{id}/stream`):
current dataset, epoch, step, total steps, progress %, loss, validation loss, perplexity,
learning rate, tokens/sec, tokens processed, elapsed, ETA, CPU/GPU, memory, checkpoint
status. Percentages are derived from `step / total_steps` — never faked.

## Training report

After every job, a report is generated: dataset, version, files, tokens, estimated words,
steps, epochs, duration, CPU/GPU, tokens/sec, initial loss, final loss, validation loss,
perplexity, retention score, evaluation score, checkpoint, resulting model version.
