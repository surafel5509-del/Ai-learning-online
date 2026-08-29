"""Models & Model Registry router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M
from apps.api.auth import auth
from apps.api.model_manager import model_manager

router = APIRouter(prefix="/models", tags=["models"])


def _mv_dict(mv: M.ModelVersion, db: Session) -> dict:
    tok = None
    if mv.tokenizer_version_id:
        tv = db.get(M.TokenizerVersion, mv.tokenizer_version_id)
        if tv:
            tok = {"id": tv.id, "version": tv.version, "vocab_size": tv.vocab_size}
    return {
        "id": mv.id, "model_id": mv.model_id, "version": mv.version,
        "parent_model_version_id": mv.parent_model_version_id,
        "architecture": mv.architecture, "parameter_count": mv.parameter_count,
        "vocab_size": mv.vocab_size, "tokenizer": tok,
        "training_dataset_version_ids": mv.training_dataset_version_ids,
        "training_tokens": mv.training_tokens, "status": mv.status,
        "checkpoint_path": mv.checkpoint_path,
        "evaluation_metrics": mv.evaluation_metrics,
        "retention_metrics": mv.retention_metrics,
        "growth_score": mv.growth_score,
        "promotion_passed": mv.promotion_passed, "promotion_reason": mv.promotion_reason,
        "created_at": mv.created_at.isoformat() if mv.created_at else None,
    }


@router.get("")
def list_models(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    models = db.query(M.Model).filter(M.Model.user_id == user.id).order_by(M.Model.created_at.desc()).all()
    out = []
    for m in models:
        versions = db.query(M.ModelVersion).filter(M.ModelVersion.model_id == m.id).order_by(M.ModelVersion.created_at.desc()).all()
        out.append({
            "id": m.id, "name": m.name, "description": m.description,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "versions": [_mv_dict(v, db) for v in versions],
        })
    return out


@router.get("/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    m = db.get(M.Model, model_id)
    if not m or m.user_id != user.id:
        raise HTTPException(404, "Model not found")
    versions = db.query(M.ModelVersion).filter(M.ModelVersion.model_id == m.id).order_by(M.ModelVersion.created_at.desc()).all()
    return {
        "id": m.id, "name": m.name, "description": m.description,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "versions": [_mv_dict(v, db) for v in versions],
    }


@router.get("/versions/{mv_id}")
def get_version(mv_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    mv = db.get(M.ModelVersion, mv_id)
    if not mv:
        raise HTTPException(404, "Model version not found")
    m = db.get(M.Model, mv.model_id)
    if not m or m.user_id != user.id:
        raise HTTPException(404, "Model version not found")
    return _mv_dict(mv, db)


@router.post("/{mv_id}/promote")
def promote(mv_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Promote a validated/candidate model version to PRODUCTION.

    Archives the previous production version (never overwrite; rollback possible).
    """
    mv = db.get(M.ModelVersion, mv_id)
    if not mv:
        raise HTTPException(404, "Model version not found")
    m = db.get(M.Model, mv.model_id)
    if not m or m.user_id != user.id:
        raise HTTPException(404, "Model version not found")
    if mv.status not in ("validated", "candidate"):
        raise HTTPException(400, f"Cannot promote model in status {mv.status}")
    # archive previous production
    prev = db.query(M.ModelVersion).filter(
        M.ModelVersion.model_id == mv.model_id,
        M.ModelVersion.status == "production",
    ).all()
    for p in prev:
        p.status = "archived"
    mv.status = "production"
    # mark its checkpoint as previous-production for rollback
    db.commit()
    model_manager.invalidate(mv_id)
    return _mv_dict(mv, db)


@router.post("/{mv_id}/rollback")
def rollback(mv_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Roll back production to a previous (archived) model version."""
    mv = db.get(M.ModelVersion, mv_id)
    if not mv:
        raise HTTPException(404, "Model version not found")
    m = db.get(M.Model, mv.model_id)
    if not m or m.user_id != user.id:
        raise HTTPException(404, "Model version not found")
    if mv.status != "archived":
        raise HTTPException(400, "Can only rollback to an archived version")
    cur = db.query(M.ModelVersion).filter(
        M.ModelVersion.model_id == mv.model_id,
        M.ModelVersion.status == "production",
    ).all()
    for c in cur:
        c.status = "archived"
    mv.status = "production"
    db.commit()
    model_manager.invalidate(mv_id)
    return _mv_dict(mv, db)


@router.get("/production/{model_id}")
def get_production(model_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    mv = db.query(M.ModelVersion).filter(
        M.ModelVersion.model_id == model_id,
        M.ModelVersion.status == "production",
    ).first()
    if not mv:
        return {"production": None}
    return {"production": _mv_dict(mv, db)}


@router.get("/registry/all")
def registry(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Full model registry across all of the user's models."""
    out = []
    models = db.query(M.Model).filter(M.Model.user_id == user.id).all()
    for m in models:
        versions = db.query(M.ModelVersion).filter(M.ModelVersion.model_id == m.id).order_by(M.ModelVersion.created_at.desc()).all()
        for v in versions:
            d = _mv_dict(v, db)
            d["model_name"] = m.name
            out.append(d)
    return out
