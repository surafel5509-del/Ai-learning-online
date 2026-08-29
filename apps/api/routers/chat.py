"""Chat router: streaming chat with RAG + memory + conversation history."""
from __future__ import annotations

import json
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M
from apps.api.auth import auth
from services.inference import GenerationConfig, generate
from services.memory import embed_text, cosine, build_context, RetrievalHit
from apps.api.model_manager import model_manager
from apps.api.schemas import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])


def _production_model(db: Session, user: M.User, model_version_id: str | None) -> M.ModelVersion:
    if model_version_id:
        mv = db.get(M.ModelVersion, model_version_id)
        if not mv:
            raise HTTPException(404, "Model version not found")
        return mv
    # default to user's production model
    mv = db.query(M.ModelVersion).join(M.Model).filter(
        M.Model.user_id == user.id, M.ModelVersion.status == "production",
    ).first()
    if not mv:
        raise HTTPException(400, "No production model available. Train and promote a model first.")
    return mv


def _build_prompt(history: list[M.Message], user_msg: str, context: str) -> str:
    """Build a simple prompt from history + optional RAG context."""
    parts = []
    if context:
        parts.append(f"Context:\n{context}\n")
    for m in history[-6:]:
        if m.role == "user":
            parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
    parts.append(f"User: {user_msg}")
    parts.append("Assistant:")
    return "\n".join(parts)


@router.get("/conversations")
def list_conversations(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    convs = db.query(M.Conversation).filter(M.Conversation.user_id == user.id).order_by(M.Conversation.updated_at.desc()).all()
    return [{
        "id": c.id, "title": c.title,
        "model_version_id": c.model_version_id,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        "num_messages": len(c.messages),
    } for c in convs]


@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    c = db.get(M.Conversation, conv_id)
    if not c or c.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    return {
        "id": c.id, "title": c.title, "model_version_id": c.model_version_id,
        "messages": [{
            "id": m.id, "role": m.role, "content": m.content,
            "model_version_id": m.model_version_id,
            "tokens_generated": m.tokens_generated, "latency_ms": m.latency_ms,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        } for m in c.messages],
    }


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    c = db.get(M.Conversation, conv_id)
    if not c or c.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    db.delete(c); db.commit()
    return {"ok": True}


@router.post("/send")
def send(body: ChatRequest, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Non-streaming chat. Returns the assistant reply with measured stats."""
    mv = _production_model(db, user, body.model_version_id)
    # get or create conversation
    conv = None
    if body.conversation_id:
        conv = db.get(M.Conversation, body.conversation_id)
        if not conv or conv.user_id != user.id:
            raise HTTPException(404, "Conversation not found")
    if conv is None:
        title = body.message[:40] + ("..." if len(body.message) > 40 else "")
        conv = M.Conversation(user_id=user.id, title=title, model_version_id=mv.id)
        db.add(conv); db.flush()
    # store user message
    db.add(M.Message(conversation_id=conv.id, role="user", content=body.message))
    db.flush()

    # RAG retrieval
    context = ""
    if body.use_rag:
        q_emb = embed_text(body.message)
        chunks = db.query(M.DocumentChunk).all()
        scored = []
        for ch in chunks:
            doc = db.get(M.Document, ch.document_id)
            if not doc or doc.user_id != user.id or ch.embedding is None:
                continue
            v = np.frombuffer(ch.embedding, dtype=np.float32)
            scored.append(RetrievalHit(chunk_id=ch.id, text=ch.text, score=cosine(q_emb, v)))
        scored.sort(key=lambda h: h.score, reverse=True)
        context = build_context(scored[:5])

    lm = model_manager.get(db, mv.id)
    prompt = _build_prompt(conv.messages[:-1], body.message, context)
    cfg = GenerationConfig(max_new_tokens=body.max_new_tokens,
                           temperature=body.temperature, top_k=body.top_k,
                           top_p=body.top_p, repetition_penalty=body.repetition_penalty,
                           do_sample=True)
    res = generate(lm.model, lm.tokenizer, prompt, cfg, lm.device, stream=False)
    msg = M.Message(conversation_id=conv.id, role="assistant", content=res.text,
                    model_version_id=mv.id, generation_settings=cfg.to_dict(),
                    tokens_generated=res.num_tokens, latency_ms=res.latency_ms)
    db.add(msg)
    conv.updated_at = conv.created_at  # trigger update
    db.commit(); db.refresh(msg)
    return {
        "conversation_id": conv.id, "message_id": msg.id,
        "content": res.text, "num_tokens": res.num_tokens,
        "latency_ms": res.latency_ms, "tokens_per_sec": res.tokens_per_sec,
        "model_version_id": mv.id, "model_version": mv.version,
        "rag_used": body.use_rag and bool(context),
    }


@router.post("/send/stream")
def send_stream(body: ChatRequest, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Streaming chat via SSE."""
    mv = _production_model(db, user, body.model_version_id)
    conv = None
    if body.conversation_id:
        conv = db.get(M.Conversation, body.conversation_id)
        if not conv or conv.user_id != user.id:
            raise HTTPException(404, "Conversation not found")
    if conv is None:
        title = body.message[:40] + ("..." if len(body.message) > 40 else "")
        conv = M.Conversation(user_id=user.id, title=title, model_version_id=mv.id)
        db.add(conv); db.flush()
    db.add(M.Message(conversation_id=conv.id, role="user", content=body.message))
    db.flush()
    context = ""
    if body.use_rag:
        q_emb = embed_text(body.message)
        chunks = db.query(M.DocumentChunk).all()
        scored = []
        for ch in chunks:
            doc = db.get(M.Document, ch.document_id)
            if not doc or doc.user_id != user.id or ch.embedding is None:
                continue
            v = np.frombuffer(ch.embedding, dtype=np.float32)
            scored.append(RetrievalHit(chunk_id=ch.id, text=ch.text, score=cosine(q_emb, v)))
        scored.sort(key=lambda h: h.score, reverse=True)
        context = build_context(scored[:5])
    lm = model_manager.get(db, mv.id)
    prompt = _build_prompt(conv.messages[:-1], body.message, context)
    cfg = GenerationConfig(max_new_tokens=body.max_new_tokens,
                           temperature=body.temperature, top_k=body.top_k,
                           top_p=body.top_p, repetition_penalty=body.repetition_penalty,
                           do_sample=True)
    conv_id = conv.id

    def stream():
        gen = generate(lm.model, lm.tokenizer, prompt, cfg, lm.device, stream=True)
        full = ""
        for chunk in gen:
            full += chunk
            yield f"data: {json.dumps({'delta': chunk, 'conversation_id': conv_id})}\n\n"
        res = getattr(gen, "result", None)
        # persist assistant message
        from packages.shared import SessionLocal
        with packages.shared.db_session() as s:
            msg = M.Message(conversation_id=conv_id, role="assistant", content=full,
                            model_version_id=mv.id, generation_settings=cfg.to_dict(),
                            tokens_generated=res.num_tokens if res else 0,
                            latency_ms=res.latency_ms if res else 0)
            s.add(msg); s.commit()
        meta = {"done": True, "conversation_id": conv_id,
                "num_tokens": res.num_tokens if res else 0,
                "latency_ms": res.latency_ms if res else 0,
                "tokens_per_sec": res.tokens_per_sec if res else 0}
        yield f"data: {json.dumps(meta)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/messages/{message_id}/feedback")
def feedback(message_id: str, body: dict, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    msg = db.get(M.Message, message_id)
    if not msg:
        raise HTTPException(404, "Message not found")
    conv = db.get(M.Conversation, msg.conversation_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(404, "Message not found")
    rating = int(body.get("rating", 0))
    if rating not in (-1, 0, 1):
        raise HTTPException(400, "rating must be -1, 0, or 1")
    db.query(M.Feedback).filter(M.Feedback.message_id == message_id).delete()
    fb = M.Feedback(message_id=message_id, rating=rating, comment=body.get("comment", ""))
    db.add(fb); db.commit()
    return {"ok": True, "rating": rating}
