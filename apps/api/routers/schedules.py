"""Schedules router: create/update scheduled learning cycles."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M
from apps.api.auth import auth


router = APIRouter(prefix="/schedules", tags=["schedules"])


@router.get("")
def list_schedules(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    items = db.query(M.Schedule).filter(M.Schedule.user_id == user.id).all()
    return [_dict(s) for s in items]


@router.get("/active")
def active_schedule(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Return the user's primary schedule settings as a single object
    (auto_learning, interval, retention_threshold). Returns defaults if none."""
    s = db.query(M.Schedule).filter(M.Schedule.user_id == user.id).first()
    if not s:
        return {"auto_learning": False, "interval": "off",
                "retention_threshold": 0.7, "last_run": None, "id": None}
    interval_map = {1800: "30m", 3600: "1h", 21600: "6h", 86400: "daily"}
    interval = interval_map.get(s.interval_seconds, "custom")
    return {
        "id": s.id, "auto_learning": s.enabled,
        "interval": interval if s.enabled else "off",
        "retention_threshold": (s.config or {}).get("retention_threshold", 0.7),
        "last_run": s.last_run_at.isoformat() if s.last_run_at else None,
    }


@router.post("")
def create_schedule(body: dict, db: Session = Depends(get_db),
                    user: M.User = Depends(auth)):
    """Create or update the primary schedule from the Settings form.
    Body keys: auto_learning, interval, retention_threshold."""
    interval_to_sec = {"off": 0, "30m": 1800, "1h": 3600, "6h": 21600, "daily": 86400, "custom": 3600}
    auto = bool(body.get("auto_learning", False))
    interval = body.get("interval", "off")
    secs = interval_to_sec.get(interval, 3600) if interval != "off" else 0
    retention = float(body.get("retention_threshold", 0.7))
    s = db.query(M.Schedule).filter(M.Schedule.user_id == user.id).first()
    if not s:
        s = M.Schedule(user_id=user.id, name="primary", enabled=auto,
                       interval_seconds=secs or 3600, auto_promote=auto, config={})
        db.add(s)
    else:
        s.enabled = auto
        s.auto_promote = auto
        if secs:
            s.interval_seconds = secs
    s.config = {"retention_threshold": retention}
    if auto and secs:
        s.next_run_at = datetime.now(timezone.utc) + timedelta(seconds=secs)
    else:
        s.next_run_at = None
    db.commit(); db.refresh(s)
    return _dict(s)


@router.patch("/{schedule_id}")
def update_schedule(schedule_id: str, body: dict, db: Session = Depends(get_db),
                    user: M.User = Depends(auth)):
    s = db.get(M.Schedule, schedule_id)
    if not s or s.user_id != user.id:
        raise HTTPException(404, "Schedule not found")
    for k in ("name", "enabled", "interval_seconds", "auto_promote", "config"):
        if k in body:
            setattr(s, k, body[k])
    if s.enabled:
        s.next_run_at = datetime.now(timezone.utc) + timedelta(seconds=s.interval_seconds)
    else:
        s.next_run_at = None
    db.commit(); db.refresh(s)
    return _dict(s)


@router.delete("/{schedule_id}")
def delete_schedule(schedule_id: str, db: Session = Depends(get_db),
                    user: M.User = Depends(auth)):
    s = db.get(M.Schedule, schedule_id)
    if not s or s.user_id != user.id:
        raise HTTPException(404, "Schedule not found")
    db.delete(s); db.commit()
    return {"ok": True}


def _dict(s: M.Schedule) -> dict:
    return {
        "id": s.id, "name": s.name, "enabled": s.enabled,
        "interval_seconds": s.interval_seconds, "auto_promote": s.auto_promote,
        "last_run_at": s.last_run_at.isoformat() if s.last_run_at else None,
        "next_run_at": s.next_run_at.isoformat() if s.next_run_at else None,
        "config": s.config,
    }
