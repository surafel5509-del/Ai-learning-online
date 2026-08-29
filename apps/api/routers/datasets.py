"""Datasets router: upload, paste, list, inspect, version, analyze, tokenize.

Implements the full pipeline: upload -> validate -> clean -> dedupe -> analyze
-> tokenize -> split -> count. Originals never destroyed.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models, settings
from packages.shared.security import allowed_file_type
from apps.api.auth import auth
from packages.shared.dataset import (
    parse_file, clean_text, deduplicate, analyze_documents, split_tokens,
    write_token_bin, file_checksum, DatasetAnalysis,
)
from packages.tokenizer import BPETokenizer
from packages.shared import db_models as M

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _get_tokenizer_for_training(db: Session) -> db_models.TokenizerVersion:
    tv = db.query(M.TokenizerVersion).filter(M.TokenizerVersion.is_active == True).first()
    if tv is None:
        raise HTTPException(400, "No active tokenizer. Train a tokenizer first.")
    return tv


@router.post("")
def create_dataset(body: dict, db: Session = Depends(get_db),
                   user: M.User = Depends(auth)):
    ds = M.Dataset(user_id=user.id, name=body["name"],
                   description=body.get("description", ""),
                   knowledge_category=body.get("knowledge_category", "General Knowledge"),
                   language=body.get("language", "auto"))
    db.add(ds); db.commit(); db.refresh(ds)
    return _dataset_dict(ds)


@router.get("")
def list_datasets(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    items = db.query(M.Dataset).filter(M.Dataset.user_id == user.id).order_by(M.Dataset.created_at.desc()).all()
    return [_dataset_dict(d) for d in items]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    ds = db.get(M.Dataset, dataset_id)
    if not ds or ds.user_id != user.id:
        raise HTTPException(404, "Dataset not found")
    return _dataset_dict(ds)


@router.patch("/{dataset_id}")
def rename_dataset(dataset_id: str, body: dict, db: Session = Depends(get_db),
                   user: M.User = Depends(auth)):
    ds = db.get(M.Dataset, dataset_id)
    if not ds or ds.user_id != user.id:
        raise HTTPException(404, "Dataset not found")
    if "name" in body:
        ds.name = body["name"]
    if "description" in body:
        ds.description = body["description"]
    if "knowledge_category" in body:
        ds.knowledge_category = body["knowledge_category"]
    db.commit(); db.refresh(ds)
    return _dataset_dict(ds)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db),
                   user: M.User = Depends(auth)):
    ds = db.get(M.Dataset, dataset_id)
    if not ds or ds.user_id != user.id:
        raise HTTPException(404, "Dataset not found")
    db.delete(ds); db.commit()
    return {"ok": True}


def _dataset_dict(ds: M.Dataset) -> dict:
    return {
        "id": ds.id, "name": ds.name, "description": ds.description,
        "knowledge_category": ds.knowledge_category, "language": ds.language,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "versions": [_version_dict(v) for v in ds.versions],
    }


def _version_dict(v: M.DatasetVersion) -> dict:
    return {
        "id": v.id, "version": v.version, "notes": v.notes,
        "num_files": v.num_files, "num_documents": v.num_documents,
        "raw_chars": v.raw_chars, "raw_bytes": v.raw_bytes,
        "num_tokens": v.num_tokens, "estimated_words": v.estimated_words,
        "unique_vocab_tokens": v.unique_vocab_tokens,
        "train_tokens": v.train_tokens, "val_tokens": v.val_tokens,
        "tokenizer_version_id": v.tokenizer_version_id,
        "analysis": v.analysis, "created_at": v.created_at.isoformat() if v.created_at else None,
        "files": [{"id": f.id, "filename": f.filename, "file_type": f.file_type,
                   "size_bytes": f.size_bytes, "num_documents": f.num_documents,
                   "num_tokens": f.num_tokens} for f in v.files],
    }


@router.post("/{dataset_id}/versions")
async def create_version(dataset_id: str, notes: str = Form(""), deduplicate: bool = Form(True),
                         files: list[UploadFile] = File(default=[]),
                         db: Session = Depends(get_db), user: M.User = Depends(auth)):
    ds = db.get(M.Dataset, dataset_id)
    if not ds or ds.user_id != user.id:
        raise HTTPException(404, "Dataset not found")
    if not files:
        raise HTTPException(400, "At least one file is required")
    # next version number
    last = db.query(M.DatasetVersion).filter(M.DatasetVersion.dataset_id == dataset_id).order_by(M.DatasetVersion.version.desc()).first()
    next_v = (last.version + 1) if last else 1
    tv = _get_tokenizer_for_training(db)
    tokenizer = BPETokenizer.load(tv.storage_path)

    version = M.DatasetVersion(dataset_id=dataset_id, version=next_v, notes=notes,
                               tokenizer_version_id=tv.id)
    db.add(version); db.flush()

    raw_dir = settings.STORAGE_DIR / "datasets" / version.id / "raw"
    proc_dir = settings.STORAGE_DIR / "datasets" / version.id / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    all_docs: list[str] = []
    total_tokens = 0
    for uf in files:
        ftype = allowed_file_type(uf.filename)
        if ftype is None:
            raise HTTPException(400, f"Unsupported file type: {uf.filename}")
        data = await uf.read()
        if len(data) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large: {uf.filename}")
        raw_path = raw_dir / uf.filename
        raw_path.write_bytes(data)
        # parse + clean
        docs = [clean_text(d) for d in parse_file(raw_path, ftype)]
        docs = [d for d in docs if d]
        per_file_ids: list[int] = []
        for d in docs:
            per_file_ids.extend(tokenizer.encode(d))
        # write processed token bin for this file
        write_token_bin(proc_dir / f"{Path(uf.filename).stem}.tokens.bin", per_file_ids)
        cs = file_checksum(raw_path)
        df = M.DatasetFile(version_id=version.id, filename=uf.filename, file_type=ftype,
                           storage_path=str(raw_path.relative_to(settings.STORAGE_DIR)),
                           size_bytes=len(data), num_documents=len(docs),
                           num_tokens=len(per_file_ids), checksum=cs)
        db.add(df)
        all_docs.extend(docs)
        total_tokens += len(per_file_ids)

    # deduplicate across the version
    removed = 0
    if deduplicate:
        all_docs, removed = deduplicate(all_docs)

    analysis = analyze_documents(all_docs, tokenizer)
    all_ids = []
    for d in all_docs:
        all_ids.extend(tokenizer.encode(d))
    train_ids, val_ids = split_tokens(all_ids, val_ratio=0.05)
    write_token_bin(proc_dir / "train.tokens.bin", train_ids)
    write_token_bin(proc_dir / "val.tokens.bin", val_ids)

    version.num_files = len(files)
    version.num_documents = analysis.num_documents
    version.raw_chars = analysis.raw_chars
    version.raw_bytes = analysis.raw_bytes
    version.num_tokens = analysis.num_tokens
    version.estimated_words = analysis.estimated_words
    version.unique_vocab_tokens = analysis.unique_vocab_tokens
    version.train_tokens = len(train_ids)
    version.val_tokens = len(val_ids)
    version.analysis = {**{k: v for k, v in analysis.__dict__.items()},
                        "duplicates_removed": removed}
    db.commit(); db.refresh(version)
    return _version_dict(version)


@router.post("/{dataset_id}/versions/paste")
def create_version_paste(dataset_id: str, body: dict, db: Session = Depends(get_db),
                         user: M.User = Depends(auth)):
    ds = db.get(M.Dataset, dataset_id)
    if not ds or ds.user_id != user.id:
        raise HTTPException(404, "Dataset not found")
    text = body.get("text", "")
    filename = body.get("filename", "pasted.txt")
    if not text.strip():
        raise HTTPException(400, "Text is empty")
    import tempfile
    # Reuse the upload path by writing to a temp file and calling logic inline.
    last = db.query(M.DatasetVersion).filter(M.DatasetVersion.dataset_id == dataset_id).order_by(M.DatasetVersion.version.desc()).first()
    next_v = (last.version + 1) if last else 1
    tv = _get_tokenizer_for_training(db)
    tokenizer = BPETokenizer.load(tv.storage_path)
    version = M.DatasetVersion(dataset_id=dataset_id, version=next_v,
                               notes=body.get("notes", ""), tokenizer_version_id=tv.id)
    db.add(version); db.flush()
    raw_dir = settings.STORAGE_DIR / "datasets" / version.id / "raw"
    proc_dir = settings.STORAGE_DIR / "datasets" / version.id / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True); proc_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / filename
    raw_path.write_text(text, encoding="utf-8")
    docs = [clean_text(d) for d in parse_file(raw_path, "txt")]
    docs = [d for d in docs if d]
    if body.get("deduplicate", True):
        docs, removed = deduplicate(docs)
    else:
        removed = 0
    ids = []
    for d in docs:
        ids.extend(tokenizer.encode(d))
    write_token_bin(proc_dir / "train.tokens.bin", ids)
    analysis = analyze_documents(docs, tokenizer)
    train_ids, val_ids = split_tokens(ids, val_ratio=0.05)
    write_token_bin(proc_dir / "train.tokens.bin", train_ids)
    write_token_bin(proc_dir / "val.tokens.bin", val_ids)
    df = M.DatasetFile(version_id=version.id, filename=filename, file_type="txt",
                       storage_path=str(raw_path.relative_to(settings.STORAGE_DIR)),
                       size_bytes=len(text.encode("utf-8")), num_documents=len(docs),
                       num_tokens=len(ids), checksum=file_checksum(raw_path))
    db.add(df)
    version.num_files = 1; version.num_documents = analysis.num_documents
    version.raw_chars = analysis.raw_chars; version.raw_bytes = analysis.raw_bytes
    version.num_tokens = analysis.num_tokens; version.estimated_words = analysis.estimated_words
    version.unique_vocab_tokens = analysis.unique_vocab_tokens
    version.train_tokens = len(train_ids); version.val_tokens = len(val_ids)
    version.analysis = {**{k: v for k, v in analysis.__dict__.items()}, "duplicates_removed": removed}
    db.commit(); db.refresh(version)
    return _version_dict(version)


@router.get("/{dataset_id}/versions/{version_id}")
def get_version(dataset_id: str, version_id: str, db: Session = Depends(get_db),
                user: M.User = Depends(auth)):
    v = db.get(M.DatasetVersion, version_id)
    if not v or v.dataset_id != dataset_id:
        raise HTTPException(404, "Version not found")
    ds = db.get(M.Dataset, dataset_id)
    if ds.user_id != user.id:
        raise HTTPException(404, "Version not found")
    return _version_dict(v)


@router.get("/{dataset_id}/versions/{version_id}/preview")
def preview_version(dataset_id: str, version_id: str, db: Session = Depends(get_db),
                    user: M.User = Depends(auth)):
    """Preview first N documents of a version (read from raw files)."""
    v = db.get(M.DatasetVersion, version_id)
    if not v or v.dataset_id != dataset_id:
        raise HTTPException(404, "Version not found")
    ds = db.get(M.Dataset, dataset_id)
    if ds.user_id != user.id:
        raise HTTPException(404, "Version not found")
    docs: list[str] = []
    for f in v.files[:5]:
        path = settings.STORAGE_DIR / f.storage_path
        if path.exists():
            docs.extend(parse_file(path, f.file_type)[:5])
    return {"documents": docs[:10]}
