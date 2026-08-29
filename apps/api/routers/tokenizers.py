"""Tokenizer router: train a new versioned tokenizer from datasets, list, set active."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M, settings
from apps.api.auth import auth
from packages.tokenizer import BPETokenizer
from packages.shared.dataset import parse_file

router = APIRouter(prefix="/tokenizers", tags=["tokenizers"])


@router.get("")
def list_tokenizers(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    items = db.query(M.TokenizerVersion).order_by(M.TokenizerVersion.created_at.desc()).all()
    return [{
        "id": t.id, "version": t.version, "vocab_size": t.vocab_size,
        "num_merges": t.num_merges, "training_tokens": t.training_tokens,
        "is_active": t.is_active, "unicode_coverage": t.unicode_coverage,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    } for t in items]


@router.post("/train")
def train_tokenizer(body: dict, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Train a new tokenizer version from dataset versions and/or raw text.

    body: {dataset_version_ids: [...], texts: [str,...], target_vocab_size: int, version: str}
    At least one of dataset_version_ids or texts must be non-empty.
    """
    dv_ids = body.get("dataset_version_ids", [])
    texts = list(body.get("texts", []))
    target = int(body.get("target_vocab_size", 512))
    if target < 260:
        raise HTTPException(400, "target_vocab_size must be >= 260")
    version = body.get("version", None)
    if not dv_ids and not texts:
        raise HTTPException(400, "Provide dataset_version_ids or texts")

    for dvid in dv_ids:
        v = db.get(M.DatasetVersion, dvid)
        if not v:
            raise HTTPException(404, f"Dataset version {dvid} not found")
        for f in v.files:
            path = settings.STORAGE_DIR / f.storage_path
            if path.exists():
                texts.extend(parse_file(path, f.file_type))
    if not texts:
        raise HTTPException(400, "No text found in selected datasets")

    # compute next version string
    last = db.query(M.TokenizerVersion).order_by(M.TokenizerVersion.created_at.desc()).first()
    if version is None:
        if last and last.version:
            parts = last.version.split(".")
            try:
                parts[-1] = str(int(parts[-1]) + 1)
                version = ".".join(parts)
            except ValueError:
                version = "0.1.0"
        else:
            version = "0.1.0"

    tok = BPETokenizer(version=version)
    info = tok.train(texts, target_vocab_size=target)
    coverage = tok.unicode_coverage(texts)
    path = settings.TOKENIZER_DIR / f"tokenizer-{version}-{tok.vocab_size}.json"
    tok.save(path)
    # deactivate previous
    db.query(M.TokenizerVersion).filter(M.TokenizerVersion.is_active == True).update({M.TokenizerVersion.is_active: False})
    tv = M.TokenizerVersion(version=version, vocab_size=tok.vocab_size,
                            num_merges=info.num_merges, storage_path=str(path),
                            training_tokens=info.created_from_tokens,
                            unicode_coverage=coverage, is_active=True)
    db.add(tv); db.commit(); db.refresh(tv)
    return {
        "id": tv.id, "version": tv.version, "vocab_size": tv.vocab_size,
        "num_merges": tv.num_merges, "training_tokens": tv.training_tokens,
        "unicode_coverage": coverage, "is_active": True,
    }


@router.post("/{tokenizer_id}/activate")
def activate_tokenizer(tokenizer_id: str, db: Session = Depends(get_db),
                       user: M.User = Depends(auth)):
    tv = db.get(M.TokenizerVersion, tokenizer_id)
    if not tv:
        raise HTTPException(404, "Tokenizer not found")
    db.query(M.TokenizerVersion).filter(M.TokenizerVersion.is_active == True).update({M.TokenizerVersion.is_active: False})
    tv.is_active = True
    db.commit()
    return {"ok": True, "id": tv.id, "is_active": True}


@router.get("/active")
def get_active(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    tv = db.query(M.TokenizerVersion).filter(M.TokenizerVersion.is_active == True).first()
    if not tv:
        return {"active": None}
    return {
        "active": True, "id": tv.id, "version": tv.version,
        "vocab_size": tv.vocab_size, "num_merges": tv.num_merges,
    }
