"""Training router: create/list/control jobs, SSE live updates, training plan.

Job execution is handled by the worker process (apps/api/worker.py) which polls
the DB for queued jobs. This router only manages job lifecycle + streaming.
"""
from __future__ import annotations

import json
import math
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M
from apps.api.auth import auth
from packages.shared.config import settings
from packages.shared.hardware import detect_device, resolve_device
from apps.api.schemas import TrainingJobCreate, TrainingJobUpdate

router = APIRouter(prefix="/training", tags=["training"])


def _safe(v):
    """Convert non-finite floats (NaN/inf) to None for JSON compliance."""
    if v is None:
        return None
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    return v


def _job_dict(job: M.TrainingJob, db: Session) -> dict:
    mv = job.output_model_version
    parent = job.parent_model_version
    # dataset versions
    dvs = []
    for dvid in job.dataset_version_ids or []:
        v = db.get(M.DatasetVersion, dvid)
        if v:
            ds = db.get(M.Dataset, v.dataset_id)
            dvs.append({"id": v.id, "version": v.version,
                        "dataset_name": ds.name if ds else "?",
                        "num_tokens": v.num_tokens})
    return {
        "id": job.id, "status": job.status, "mode": job.mode, "device": job.device,
        "dataset_versions": dvs, "current_dataset_index": job.current_dataset_index,
        "parent_model_version_id": job.parent_model_version_id,
        "parent_model_version": ({"id": parent.id, "version": parent.version} if parent else None),
        "output_model_version_id": job.output_model_version_id,
        "output_model_version": ({"id": mv.id, "version": mv.version, "status": mv.status} if mv else None),
        "hyperparams": job.hyperparams, "replay_ratio": job.replay_ratio,
        "progress_pct": _safe(job.progress_pct), "current_epoch": job.current_epoch,
        "current_step": job.current_step, "total_steps": job.total_steps,
        "current_loss": _safe(job.current_loss), "best_val_loss": _safe(job.best_val_loss),
        "best_val_perplexity": _safe(job.best_val_perplexity),
        "final_loss": _safe(job.final_loss), "final_val_loss": _safe(job.final_val_loss),
        "final_perplexity": _safe(job.final_perplexity),
        "tokens_processed": job.tokens_processed, "tokens_per_sec": _safe(job.tokens_per_sec),
        "elapsed_seconds": _safe(job.elapsed_seconds),
        "retention_score": _safe(job.retention_score), "evaluation_score": _safe(job.evaluation_score),
        "error_message": job.error_message, "worker_id": job.worker_id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@router.post("/jobs")
def create_job(body: TrainingJobCreate, db: Session = Depends(get_db),
               user: M.User = Depends(auth)):
    # validate datasets exist
    for dvid in body.dataset_version_ids:
        v = db.get(M.DatasetVersion, dvid)
        if not v:
            raise HTTPException(404, f"Dataset version {dvid} not found")
    # validate parent model if provided
    if body.parent_model_version_id:
        pmv = db.get(M.ModelVersion, body.parent_model_version_id)
        if not pmv:
            raise HTTPException(404, "Parent model version not found")
    # resolve device
    try:
        resolve_device(body.device)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    # tokenizer
    tv_id = body.tokenizer_version_id
    if not tv_id:
        tv = db.query(M.TokenizerVersion).filter(M.TokenizerVersion.is_active == True).first()
        tv_id = tv.id if tv else None
    if not tv_id:
        raise HTTPException(400, "No tokenizer available. Train a tokenizer first.")
    job = M.TrainingJob(
        user_id=user.id, dataset_version_ids=body.dataset_version_ids,
        parent_model_version_id=body.parent_model_version_id,
        base_model_config=body.base_model_config or {},
        mode=body.mode, device=body.device,
        hyperparams=body.hyperparams or {},
        tokenizer_version_id=tv_id,
        replay_ratio=body.replay_ratio,
        replay_dataset_version_ids=body.replay_dataset_version_ids,
        status="queued",
    )
    db.add(job); db.commit(); db.refresh(job)
    return _job_dict(job, db)


@router.get("/jobs")
def list_jobs(status_filter: Optional[str] = None, db: Session = Depends(get_db),
              user: M.User = Depends(auth)):
    q = db.query(M.TrainingJob).filter(M.TrainingJob.user_id == user.id)
    if status_filter:
        q = q.filter(M.TrainingJob.status == status_filter)
    jobs = q.order_by(M.TrainingJob.created_at.desc()).all()
    return [_job_dict(j, db) for j in jobs]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    job = db.get(M.TrainingJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")
    return _job_dict(job, db)


@router.post("/jobs/{job_id}/control")
def control_job(job_id: str, body: TrainingJobUpdate, db: Session = Depends(get_db),
                user: M.User = Depends(auth)):
    job = db.get(M.TrainingJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")
    action = body.action
    if action == "pause" and job.status in ("running", "queued"):
        job.status = "paused"
    elif action == "resume" and job.status == "paused":
        job.status = "queued"
    elif action == "cancel" and job.status in ("running", "queued", "paused"):
        job.status = "cancelled"
    else:
        raise HTTPException(400, f"Cannot {action} job in status {job.status}")
    db.commit()
    return _job_dict(job, db)


@router.get("/jobs/{job_id}/steps")
def job_steps(job_id: str, limit: int = 200, db: Session = Depends(get_db),
              user: M.User = Depends(auth)):
    job = db.get(M.TrainingJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")
    steps = db.query(M.TrainingStep).filter(M.TrainingStep.job_id == job_id).order_by(
        M.TrainingStep.id.desc()).limit(limit).all()
    steps = list(reversed(steps))
    return [{
        "step": s.step, "epoch": s.epoch, "dataset_index": s.dataset_index,
        "loss": s.loss, "learning_rate": s.learning_rate,
        "tokens_processed": s.tokens_processed, "tokens_per_sec": s.tokens_per_sec,
        "grad_norm": s.grad_norm, "memory_mb": s.memory_mb,
        "is_validation": s.is_validation,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    } for s in steps]


@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Server-Sent Events stream of live training updates for a job.

    Polls the DB for job + latest steps and emits real events. Heartbeat keeps
    the connection alive. Ends when job reaches a terminal state.
    """
    job = db.get(M.TrainingJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")

    def event_stream():
        last_step_id = 0
        terminal = {"completed", "failed", "cancelled"}
        while True:
            db.refresh(job)
            # new steps
            new_steps = db.query(M.TrainingStep).filter(
                M.TrainingStep.job_id == job_id,
                M.TrainingStep.id > last_step_id,
            ).order_by(M.TrainingStep.id.asc()).all()
            for s in new_steps:
                last_step_id = s.id
                yield _sse("step", {
                    "step": s.step, "epoch": s.epoch, "dataset_index": s.dataset_index,
                    "loss": s.loss, "learning_rate": s.learning_rate,
                    "tokens_processed": s.tokens_processed,
                    "tokens_per_sec": s.tokens_per_sec, "grad_norm": s.grad_norm,
                    "memory_mb": s.memory_mb, "is_validation": s.is_validation,
                })
            # job status snapshot
            yield _sse("status", _job_dict(job, db))
            if job.status in terminal:
                yield _sse("done", {"status": job.status})
                break
            time.sleep(settings.WORKER_POLL_SECONDS)
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/plan")
def training_plan(dataset_version_ids: list[str] = Query(default=[]),
                  mode: str = "balanced", seq_len: int = 128,
                  batch_size: int = 8, epochs: int = 1,
                  db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Compute a real training plan: total tokens, steps, est. time, est. params."""
    total_tokens = 0
    datasets = []
    for dvid in dataset_version_ids:
        v = db.get(M.DatasetVersion, dvid)
        if not v:
            continue
        ds = db.get(M.Dataset, v.dataset_id)
        total_tokens += v.train_tokens
        datasets.append({"id": v.id, "name": ds.name if ds else "?",
                         "tokens": v.train_tokens})
    steps_per_epoch = total_tokens // (seq_len * batch_size) if total_tokens else 0
    total_steps = steps_per_epoch * epochs
    # rough CPU estimate: ~tokens/sec baseline (real, measured at runtime; here an estimate)
    est_tps = 4000
    est_seconds = total_tokens / est_tps * epochs if total_tokens else 0
    return {
        "datasets": datasets, "total_train_tokens": total_tokens,
        "seq_len": seq_len, "batch_size": batch_size, "epochs": epochs,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "estimated_seconds": est_seconds,
        "estimated_words": int(total_tokens / 1.3),
        "note": "estimated_words is a rough estimate (tokens != words). "
                "estimated_seconds is a coarse CPU estimate; actual measured at runtime.",
    }


@router.get("/hardware")
def hardware_info(user: M.User = Depends(auth)):
    """Real device detection: CPU/GPU, cores, VRAM."""
    return detect_device().to_dict()
