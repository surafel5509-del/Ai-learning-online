"""Inference router: Model Test Lab generation + comparison.

Real generation with KV cache + sampling. Streaming via SSE.
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M
from apps.api.auth import auth
from services.inference import GenerationConfig, generate
from apps.api.model_manager import model_manager
from apps.api.schemas import GenerateRequest, CompareRequest

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/generate")
def generate_text(body: GenerateRequest, db: Session = Depends(get_db),
                  user: M.User = Depends(auth)):
    lm = model_manager.get(db, body.model_version_id)
    cfg = GenerationConfig(max_new_tokens=body.max_new_tokens,
                           temperature=body.temperature, top_k=body.top_k,
                           top_p=body.top_p, repetition_penalty=body.repetition_penalty,
                           do_sample=body.do_sample)
    start = time.perf_counter()
    res = generate(lm.model, lm.tokenizer, body.prompt, cfg, lm.device, stream=False)
    return {
        "text": res.text, "num_tokens": res.num_tokens,
        "latency_ms": res.latency_ms, "tokens_per_sec": res.tokens_per_sec,
        "model_version_id": body.model_version_id,
        "model_version": lm.model_version.version,
        "settings": res.settings,
    }


@router.post("/generate/stream")
def generate_stream(body: GenerateRequest, db: Session = Depends(get_db),
                    user: M.User = Depends(auth)):
    lm = model_manager.get(db, body.model_version_id)
    cfg = GenerationConfig(max_new_tokens=body.max_new_tokens,
                           temperature=body.temperature, top_k=body.top_k,
                           top_p=body.top_p, repetition_penalty=body.repetition_penalty,
                           do_sample=body.do_sample)

    def stream():
        gen = generate(lm.model, lm.tokenizer, body.prompt, cfg, lm.device, stream=True)
        for chunk in gen:
            yield f"data: {json.dumps({'delta': chunk})}\n\n"
        res = getattr(gen, "result", None)
        meta = {}
        if res:
            meta = {"num_tokens": res.num_tokens, "latency_ms": res.latency_ms,
                    "tokens_per_sec": res.tokens_per_sec,
                    "model_version": lm.model_version.version}
        yield f"data: {json.dumps({'done': True, **meta})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/compare")
def compare(body: CompareRequest, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    """Ask the same prompt to two model versions; return both + measured stats."""
    lmA = model_manager.get(db, body.model_version_id_a)
    lmB = model_manager.get(db, body.model_version_id_b)
    cfg = GenerationConfig(max_new_tokens=body.max_new_tokens,
                           temperature=body.temperature, top_k=40, top_p=0.9,
                           repetition_penalty=1.15, do_sample=True)
    resA = generate(lmA.model, lmA.tokenizer, body.prompt, cfg, lmA.device, stream=False)
    resB = generate(lmB.model, lmB.tokenizer, body.prompt, cfg, lmB.device, stream=False)
    return {
        "prompt": body.prompt,
        "a": {"model_version_id": body.model_version_id_a,
              "model_version": lmA.model_version.version,
              "text": resA.text, "num_tokens": resA.num_tokens,
              "latency_ms": resA.latency_ms, "tokens_per_sec": resA.tokens_per_sec},
        "b": {"model_version_id": body.model_version_id_b,
              "model_version": lmB.model_version.version,
              "text": resB.text, "num_tokens": resB.num_tokens,
              "latency_ms": resB.latency_ms, "tokens_per_sec": resB.tokens_per_sec},
    }
