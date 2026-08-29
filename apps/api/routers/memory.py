"""Memory & Knowledge router: documents, chunks, embeddings, RAG retrieval.

Stores explicit memory and vector-indexed documents. Retrieval is real cosine
search over hashed n-gram embeddings.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M
from apps.api.auth import auth
from services.memory import embed_text, cosine, chunk_text, RetrievalHit, build_context
from apps.api.schemas import DocumentCreate, MemoryCreate

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/documents")
def list_documents(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    docs = db.query(M.Document).filter(M.Document.user_id == user.id).order_by(M.Document.created_at.desc()).all()
    return [{
        "id": d.id, "title": d.title, "source": d.source, "source_ref": d.source_ref,
        "knowledge_category": d.knowledge_category, "language": d.language,
        "content_preview": d.content[:200], "num_chunks": len(d.chunks),
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in docs]


@router.post("/documents")
def add_document(body: DocumentCreate, db: Session = Depends(get_db),
                 user: M.User = Depends(auth)):
    doc = M.Document(user_id=user.id, title=body.title, content=body.content,
                     source="manual", knowledge_category=body.knowledge_category,
                     language=body.language)
    db.add(doc); db.flush()
    # chunk + embed
    chunks = chunk_text(body.content)
    for i, ch in enumerate(chunks):
        emb = embed_text(ch)
        dc = M.DocumentChunk(document_id=doc.id, chunk_index=i, text=ch,
                             embedding=emb.tobytes(), embedding_dim=emb.shape[0])
        db.add(dc)
    db.commit(); db.refresh(doc)
    return {"id": doc.id, "num_chunks": len(chunks)}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    doc = db.get(M.Document, doc_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")
    db.delete(doc); db.commit()
    return {"ok": True}


@router.get("/memories")
def list_memories(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    items = db.query(M.Memory).filter(M.Memory.user_id == user.id).order_by(M.Memory.created_at.desc()).all()
    return [{
        "id": m.id, "kind": m.kind, "content": m.content,
        "metadata": m.metadata_json,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    } for m in items]


@router.post("/memories")
def add_memory(body: MemoryCreate, db: Session = Depends(get_db),
               user: M.User = Depends(auth)):
    mem = M.Memory(user_id=user.id, kind=body.kind, content=body.content,
                   metadata_json=body.metadata)
    db.add(mem); db.commit(); db.refresh(mem)
    return {"id": mem.id, "kind": mem.kind, "content": mem.content}


@router.post("/retrieve")
def retrieve(body: dict, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Real RAG retrieval: embed query -> cosine search top-k -> context."""
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))
    if not query.strip():
        return {"hits": [], "context": ""}
    q_emb = embed_text(query)
    chunks = db.query(M.DocumentChunk).all()
    scored: list[RetrievalHit] = []
    for ch in chunks:
        doc = db.get(M.Document, ch.document_id)
        if not doc or doc.user_id != user.id:
            continue
        if ch.embedding is None:
            continue
        v = np.frombuffer(ch.embedding, dtype=np.float32)
        score = cosine(q_emb, v)
        scored.append(RetrievalHit(chunk_id=ch.id, text=ch.text, score=score,
                                   document_id=doc.id, title=doc.title))
    scored.sort(key=lambda h: h.score, reverse=True)
    top = scored[:top_k]
    return {
        "hits": [{"chunk_id": h.chunk_id, "text": h.text, "score": h.score,
                  "document_id": h.document_id, "title": h.title} for h in top],
        "context": build_context(top),
    }


@router.post("/corrections")
def add_correction(body: dict, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Store a user correction (Q, incorrect, correct). Does NOT modify weights."""
    cor = M.Correction(user_id=user.id, question=body["question"],
                       incorrect_answer=body.get("incorrect_answer", ""),
                       correct_answer=body["correct_answer"],
                       context=body.get("context", ""), status="pending")
    db.add(cor); db.commit(); db.refresh(cor)
    return {"id": cor.id, "status": cor.status}


@router.get("/corrections")
def list_corrections(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    items = db.query(M.Correction).filter(M.Correction.user_id == user.id).order_by(M.Correction.created_at.desc()).all()
    return [{
        "id": c.id, "question": c.question, "incorrect_answer": c.incorrect_answer,
        "correct_answer": c.correct_answer, "context": c.context, "status": c.status,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    } for c in items]
