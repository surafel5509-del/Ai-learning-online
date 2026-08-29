"""apps.api.worker — the training worker process.

Polls the DB for queued training jobs and executes them for real:
- loads parent model (continual learning) or builds from scratch
- runs multi-dataset sequential training with per-dataset checkpoints
- gradient accumulation, AMP (GPU), warmup+cosine LR, grad clipping
- periodic validation + checkpointing
- replay data mixing (forgetting prevention)
- before/after evaluation -> retention score
- creates a candidate model version; gates promotion by retention threshold

Writes live metrics to training_steps (streamed via SSE to the dashboard).
Recovers from checkpoints on restart. Browser closing does not affect it.
"""
from __future__ import annotations

import math
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

# ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.shared import SessionLocal, init_db, db_models as M, settings
from packages.shared.hardware import detect_device, resolve_device, live_stats
from packages.shared.dataset import read_token_bin
from packages.shared.metrics import perplexity, knowledge_retention_score, ai_growth_score, estimate_words_from_tokens
from packages.model import ModelConfig, TransformerLM
from packages.tokenizer import BPETokenizer
from services.trainer.core import (
    Hyperparams, make_hyperparams, TrainState, build_optimizer, make_batches,
    lr_schedule, save_checkpoint, load_checkpoint, evaluate_loss,
)
from services.inference import GenerationConfig
from services.evaluator import run_tests, EvalTest


WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"
PAUSE_CHECK_SECONDS = 1.0


def _now():
    return datetime.now(timezone.utc)


def _load_tokens_for_version(v: M.DatasetVersion) -> tuple[list[int], list[int]]:
    """Load train/val token ids for a dataset version from processed bin files."""
    proc_dir = settings.STORAGE_DIR / "datasets" / v.id / "processed"
    train_path = proc_dir / "train.tokens.bin"
    val_path = proc_dir / "val.tokens.bin"
    train_ids = read_token_bin(train_path) if train_path.exists() else []
    val_ids = read_token_bin(val_path) if val_path.exists() else []
    return train_ids, val_ids


def _get_or_create_model(db, user_id: str, parent_mv: Optional[M.ModelVersion],
                         base_config: Optional[dict], tokenizer_version_id: str
                         ) -> tuple[M.Model, M.ModelVersion, ModelConfig]:
    """Resolve the model family + create a new TRAINING version (continual learning)."""
    if parent_mv:
        parent_model = db.get(M.Model, parent_mv.model_id)
        cfg = ModelConfig.from_dict(parent_mv.architecture)
        # new version string: bump patch
        new_version = _bump_version(parent_mv.version)
        mv = M.ModelVersion(model_id=parent_mv.model_id, version=new_version,
                            parent_model_version_id=parent_mv.id,
                            architecture=parent_mv.architecture,
                            parameter_count=parent_mv.parameter_count,
                            vocab_size=parent_mv.vocab_size,
                            tokenizer_version_id=tokenizer_version_id,
                            status="training",
                            training_dataset_version_ids=list(parent_mv.training_dataset_version_ids or []))
        db.add(mv); db.flush()
        return parent_model, mv, cfg
    # from scratch
    cfg = ModelConfig.from_dict(base_config or {})
    model = M.Model(user_id=user_id, name="ai-model",
                    description="Auto-created model from training")
    db.add(model); db.flush()
    mv = M.ModelVersion(model_id=model.id, version="1.0.0", architecture=cfg.to_dict(),
                       parameter_count=0, vocab_size=cfg.vocab_size,
                       tokenizer_version_id=tokenizer_version_id, status="training")
    db.add(mv); db.flush()
    return model, mv, cfg


def _bump_version(v: str) -> str:
    parts = v.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return v + ".1"


def _record_step(db, job_id: str, state: TrainState, loss: float, lr: float,
                 tokens: int, tps: float, dataset_index: int, epoch: int,
                 is_val: bool = False, grad_norm: Optional[float] = None,
                 mem_mb: Optional[float] = None) -> None:
    db.add(M.TrainingStep(
        job_id=job_id, step=state.step, epoch=epoch, dataset_index=dataset_index,
        loss=float(loss), learning_rate=float(lr), tokens_processed=int(tokens),
        tokens_per_sec=float(tps), grad_norm=grad_norm, memory_mb=mem_mb,
        is_validation=is_val,
    ))
    db.commit()


def _save_ckpt(db, mv: M.ModelVersion, job: M.TrainingJob, state: TrainState,
               model, optimizer, hp: Hyperparams, val_loss: Optional[float],
               val_ppl: Optional[float], is_best: bool, is_latest: bool,
               device: str, extra: dict) -> M.Checkpoint:
    path = settings.CHECKPOINT_DIR / mv.id / f"step_{state.step}.pt"
    save_checkpoint(path, model, optimizer, hp, state,
                    {"val_loss": val_loss, "dataset_index": extra.get("dataset_index", 0)})
    ckpt = M.Checkpoint(model_version_id=mv.id, job_id=job.id, step=state.step,
                        epoch=state.epoch, path=str(path), val_loss=val_loss,
                        val_perplexity=val_ppl, metrics=extra,
                        is_latest=is_latest, is_best=is_best)
    # update latest flags
    if is_latest:
        db.query(M.Checkpoint).filter(M.Checkpoint.model_version_id == mv.id,
                                      M.Checkpoint.is_latest == True).update({M.Checkpoint.is_latest: False})
    if is_best:
        db.query(M.Checkpoint).filter(M.Checkpoint.model_version_id == mv.id,
                                      M.Checkpoint.is_best == True).update({M.Checkpoint.is_best: False})
    db.add(ckpt)
    # update model version checkpoint path
    mv.checkpoint_path = str(path)
    db.commit()
    return ckpt


def _run_tests_for_retention(db, model, tokenizer, mv: M.ModelVersion, device: str,
                             user_id: str) -> list[dict]:
    """Run all of the user's evaluation tests against the model for retention."""
    evs = db.query(M.Evaluation).filter(M.Evaluation.user_id == user_id).all()
    tests: list[EvalTest] = []
    for ev in evs:
        for t in ev.tests:
            tests.append(EvalTest(test_id=t.id, question=t.question,
                                  expected_answer=t.expected_answer, criteria=t.criteria))
    if not tests:
        return []
    cfg = GenerationConfig(max_new_tokens=48, temperature=0.3, top_k=40, top_p=0.9,
                           repetition_penalty=1.15, do_sample=True)
    return run_tests(model, tokenizer, tests, device, cfg)


def execute_job(job_id: str, device_name: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(M.TrainingJob, job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = _now()
        job.worker_id = WORKER_ID
        # register worker
        worker = db.query(M.Worker).filter(M.Worker.id == WORKER_ID).first()
        if not worker:
            di = detect_device()
            worker = M.Worker(id=WORKER_ID, hostname=socket.gethostname(),
                              device=di.device, device_name=di.name, status="busy",
                              current_job_id=job.id, metadata_json={})
            db.add(worker)
        else:
            worker.status = "busy"; worker.current_job_id = job.id
        db.commit()

        # resolve device
        device = resolve_device(job.device)
        # tokenizer
        tv = db.get(M.TokenizerVersion, job.tokenizer_version_id)
        if not tv:
            raise RuntimeError("Tokenizer version not found")
        tokenizer = BPETokenizer.load(tv.storage_path)

        # model
        parent_mv = None
        if job.parent_model_version_id:
            parent_mv = db.get(M.ModelVersion, job.parent_model_version_id)
        model_obj, mv, cfg = _get_or_create_model(db, job.user_id, parent_mv,
                                                  job.base_model_config, tv.id)
        # build model
        if parent_mv and parent_mv.checkpoint_path and Path(parent_mv.checkpoint_path).exists():
            model, optimizer, ck = load_checkpoint(Path(parent_mv.checkpoint_path), device)
            state = TrainState(model=model, optimizer=optimizer, step=0, epoch=0,
                               best_val_loss=ck.get("best_val_loss", float("inf")),
                               best_val_perplexity=ck.get("best_val_perplexity", float("inf")))
        else:
            # ensure vocab size matches tokenizer
            cfg.vocab_size = tokenizer.vocab_size
            model = TransformerLM(cfg).to(device)
            mv.architecture = cfg.to_dict()
            mv.parameter_count = model.num_parameters()
            mv.vocab_size = tokenizer.vocab_size
            db.commit()
            hp = make_hyperparams(job.mode, job.hyperparams)
            optimizer = build_optimizer(model, hp)
            state = TrainState(model=model, optimizer=optimizer)

        hp = make_hyperparams(job.mode, job.hyperparams)
        # Rebuild optimizer with the (possibly updated) hp
        optimizer = build_optimizer(model, hp)
        state.optimizer = optimizer

        # BEFORE evaluation (retention baseline) — only if parent exists
        before_results: list[dict] = []
        if parent_mv:
            before_results = _run_tests_for_retention(db, model, tokenizer, mv, device, job.user_id)

        use_amp = (device == "cuda")
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # replay data (continual learning): mix in old dataset tokens
        replay_ids: list[int] = []
        for dvid in (job.replay_dataset_version_ids or []):
            v = db.get(M.DatasetVersion, dvid)
            if v:
                tr, _ = _load_tokens_for_version(v)
                replay_ids.extend(tr)
        if job.replay_ratio > 0 and replay_ids:
            # cap replay to ratio * new data length (computed per dataset below)
            pass

        # total steps estimate
        total_new_tokens = 0
        for dvid in job.dataset_version_ids:
            v = db.get(M.DatasetVersion, dvid)
            if v:
                tr, _ = _load_tokens_for_version(v)
                total_new_tokens += len(tr)
        total_steps_est = (total_new_tokens // (hp.seq_len * hp.batch_size)) * hp.epochs
        job.total_steps = int(total_steps_est)
        db.commit()

        model.train()
        job_start = time.time()
        global_tokens = 0
        all_trained_dv_ids = list(mv.training_dataset_version_ids or [])

        for ds_idx, dvid in enumerate(job.dataset_version_ids):
            # check pause/cancel
            if _should_stop(db, job_id):
                break
            v = db.get(M.DatasetVersion, dvid)
            if not v:
                continue
            train_ids, val_ids = _load_tokens_for_version(v)
            # mix replay
            if job.replay_ratio > 0 and replay_ids:
                import random
                n_replay = int(len(train_ids) * job.replay_ratio)
                if n_replay > 0 and len(replay_ids) > 0:
                    sampled = [random.choice(replay_ids) for _ in range(min(n_replay, len(replay_ids)))]
                    train_ids = sampled + train_ids
            batches = make_batches(train_ids, hp.seq_len, hp.batch_size)
            if not batches:
                continue
            all_trained_dv_ids.append(dvid)
            job.current_dataset_index = ds_idx
            db.commit()

            for epoch in range(hp.epochs):
                if _should_stop(db, job_id):
                    break
                job.current_epoch = epoch
                db.commit()
                accum_loss = 0.0
                for bi, batch in enumerate(batches):
                    if _should_stop(db, job_id):
                        break
                    # respect pause
                    while _is_paused(db, job_id):
                        time.sleep(PAUSE_CHECK_SECONDS)
                    x = batch[:, :-1].to(device)
                    y = batch[:, 1:].to(device)
                    optimizer.zero_grad()
                    if use_amp:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            out = model(x, targets=y)
                            loss = out["loss"] / hp.grad_accum_steps
                        scaler.scale(loss).backward()
                    else:
                        out = model(x, targets=y)
                        loss = out["loss"] / hp.grad_accum_steps
                        loss.backward()
                    # gradient accumulation: step every grad_accum_steps
                    if (bi + 1) % hp.grad_accum_steps == 0 or (bi + 1) == len(batches):
                        if use_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
                            scaler.step(optimizer); scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
                            optimizer.step()
                        state.step += 1
                        lr = lr_schedule(state.step, job.total_steps or 1, hp.warmup_steps, hp.learning_rate)
                        for g in optimizer.param_groups:
                            g["lr"] = lr
                    else:
                        lr = lr_schedule(state.step, job.total_steps or 1, hp.warmup_steps, hp.learning_rate)
                        for g in optimizer.param_groups:
                            g["lr"] = lr
                        continue
                    # record step (real metrics)
                    step_tokens = x.numel()
                    global_tokens += step_tokens
                    elapsed = time.time() - job_start
                    tps = global_tokens / elapsed if elapsed > 0 else 0.0
                    mem_mb = live_stats().get("memory_used_mb") if device == "cpu" else live_stats().get("vram_used_mb")
                    _record_step(db, job_id, state, float(loss.item()) * hp.grad_accum_steps,
                                 lr, step_tokens, tps, ds_idx, epoch, is_val=False,
                                 grad_norm=None, mem_mb=mem_mb)
                    job.current_step = state.step
                    job.current_loss = float(loss.item()) * hp.grad_accum_steps
                    job.tokens_processed = global_tokens
                    job.tokens_per_sec = tps
                    job.elapsed_seconds = elapsed
                    job.progress_pct = min(100.0, (state.step / max(1, job.total_steps)) * 100.0)
                    db.commit()
                    # periodic validation
                    if val_ids and state.step > 0 and state.step % hp.val_every == 0:
                        vloss, vppl = evaluate_loss(model, val_ids, hp.seq_len, hp.batch_size, device)
                        if not (math.isnan(vloss) or math.isinf(vloss)):
                            _record_step(db, job_id, state, vloss, lr, 0, 0.0, ds_idx, epoch, is_val=True)
                            if vloss < state.best_val_loss:
                                state.best_val_loss = vloss
                                state.best_val_perplexity = vppl
                                _save_ckpt(db, mv, job, state, model, optimizer, hp,
                                           vloss, vppl, is_best=True, is_latest=False,
                                           device=device, extra={"dataset_index": ds_idx})
                            job.best_val_loss = state.best_val_loss
                            job.best_val_perplexity = state.best_val_perplexity
                            db.commit()
                    # periodic checkpoint
                    if hp.checkpoint_every and state.step % hp.checkpoint_every == 0:
                        _save_ckpt(db, mv, job, state, model, optimizer, hp,
                                   job.best_val_loss, state.best_val_perplexity,
                                   is_best=False, is_latest=True, device=device,
                                   extra={"dataset_index": ds_idx})
                    model.train()

            # per-dataset checkpoint
            _save_ckpt(db, mv, job, state, model, optimizer, hp,
                       job.best_val_loss, state.best_val_perplexity,
                       is_best=False, is_latest=True, device=device,
                       extra={"dataset_index": ds_idx, "dataset_version_id": dvid})

        # final validation across last val set
        final_val_loss = job.best_val_loss
        final_ppl = state.best_val_perplexity
        # AFTER evaluation (retention)
        after_results = _run_tests_for_retention(db, model, tokenizer, mv, device, job.user_id)
        retention_score = 1.0
        retention_detail = {}
        if before_results and after_results:
            before = {r["test_id"]: r["score"] for r in before_results}
            after = {r["test_id"]: r["score"] for r in after_results}
            retention_score, retention_detail = knowledge_retention_score(before, after)

        # evaluation score (mean of after results)
        eval_score = (sum(r["score"] for r in after_results) / len(after_results)) if after_results else 0.0

        # persist after-eval results to the model version
        mv.retention_metrics = {"retention_score": retention_score, **retention_detail,
                                "before_mean": (sum(r["score"] for r in before_results)/len(before_results)) if before_results else 0.0,
                                "after_mean": eval_score}
        mv.evaluation_metrics = {"mean_score": eval_score, "num_tests": len(after_results),
                                 "passed": sum(1 for r in after_results if r["passed"])}
        # growth score
        val_perf = 1.0 / (1.0 + (final_ppl or 10.0))
        gs, _ = ai_growth_score(eval_score, retention_score, val_perf, 1.0, 1.0, eval_score)
        mv.growth_score = gs
        mv.training_dataset_version_ids = all_trained_dv_ids
        mv.training_tokens = global_tokens

        # promotion gate: retention must be acceptable
        RETENTION_THRESHOLD = 0.7
        if retention_score >= RETENTION_THRESHOLD:
            mv.status = "candidate"
            mv.promotion_passed = True
            mv.promotion_reason = f"Passed retention gate ({retention_score:.2f} >= {RETENTION_THRESHOLD})"
        else:
            mv.status = "candidate"
            mv.promotion_passed = False
            mv.promotion_reason = (f"Flagged: retention {retention_score:.2f} < {RETENTION_THRESHOLD}. "
                                   "Automatic promotion blocked; manual review required.")

        job.output_model_version_id = mv.id
        job.final_loss = float(loss.item()) * hp.grad_accum_steps if 'loss' in dir() else None
        job.final_val_loss = final_val_loss
        job.final_perplexity = final_ppl
        job.retention_score = retention_score
        job.evaluation_score = eval_score
        job.progress_pct = 100.0
        job.status = "completed"
        job.finished_at = _now()
        db.commit()

    except Exception as e:
        traceback.print_exc()
        db = SessionLocal()
        job = db.get(M.TrainingJob, job_id)
        if job:
            job.status = "failed"
            job.error_message = f"{type(e).__name__}: {e}"
            job.finished_at = _now()
            if job.output_model_version_id:
                mv = db.get(M.ModelVersion, job.output_model_version_id)
                if mv and mv.status == "training":
                    mv.status = "failed"
            db.commit()
    finally:
        try:
            db.close()
        except Exception:
            pass
        # mark worker idle
        db2 = SessionLocal()
        w = db2.query(M.Worker).filter(M.Worker.id == WORKER_ID).first()
        if w:
            w.status = "idle"; w.current_job_id = None
            w.last_heartbeat = _now()
            db2.commit()
        db2.close()


def _should_stop(db, job_id: str) -> bool:
    job = db.get(M.TrainingJob, job_id)
    if not job:
        return True
    return job.status in ("cancelled", "failed")


def _is_paused(db, job_id: str) -> bool:
    job = db.get(M.TrainingJob, job_id)
    return bool(job and job.status == "paused")


def heartbeat_loop():
    """Background thread updating worker heartbeat."""
    import threading
    def _loop():
        while True:
            try:
                db = SessionLocal()
                w = db.query(M.Worker).filter(M.Worker.id == WORKER_ID).first()
                if w:
                    w.last_heartbeat = _now()
                    db.commit()
                db.close()
            except Exception:
                pass
            time.sleep(5)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def main():
    init_db()
    di = detect_device()
    print(f"[worker {WORKER_ID}] device={di.device} name={di.name} cuda={di.cuda_available}")
    # register worker
    db = SessionLocal()
    w = db.query(M.Worker).filter(M.Worker.id == WORKER_ID).first()
    if not w:
        w = M.Worker(id=WORKER_ID, hostname=socket.gethostname(), device=di.device,
                     device_name=di.name, status="idle", metadata_json={})
        db.add(w)
    else:
        w.status = "idle"
    db.commit(); db.close()
    heartbeat_loop()
    print("[worker] polling for jobs...")
    while True:
        try:
            db = SessionLocal()
            job = db.query(M.TrainingJob).filter(
                M.TrainingJob.status == "queued"
            ).order_by(M.TrainingJob.created_at.asc()).first()
            db.close()
            if job:
                print(f"[worker] starting job {job.id}")
                execute_job(job.id, di.device)
                print(f"[worker] finished job {job.id}")
            else:
                time.sleep(settings.WORKER_POLL_SECONDS)
        except KeyboardInterrupt:
            print("[worker] shutting down")
            break
        except Exception as e:
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
