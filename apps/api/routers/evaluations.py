"""Evaluation router: suites, tests, run evaluations, custom tests, results.

Knowledge tests are generated from dataset facts (real benchmarks) or created by
users. Running an evaluation uses actual model generation + answer scoring.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models as M, settings
from apps.api.auth import auth
from packages.shared.dataset import parse_file
from services.evaluator import extract_benchmark_tests, run_tests, EvalTest, score_response
from services.inference import GenerationConfig
from apps.api.model_manager import model_manager
from apps.api.schemas import EvaluationCreate, EvaluationTestCreate, RunEvaluationRequest

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("")
def list_evaluations(db: Session = Depends(get_db), user: M.User = Depends(auth)):
    items = db.query(M.Evaluation).filter(M.Evaluation.user_id == user.id).order_by(M.Evaluation.created_at.desc()).all()
    return [{
        "id": e.id, "name": e.name, "kind": e.kind,
        "source_dataset_version_id": e.source_dataset_version_id,
        "num_tests": len(e.tests),
        "created_at": e.created_at.isoformat() if e.created_at else None,
    } for e in items]


@router.post("")
def create_evaluation(body: EvaluationCreate, db: Session = Depends(get_db),
                      user: M.User = Depends(auth)):
    ev = M.Evaluation(user_id=user.id, name=body.name, kind=body.kind,
                      source_dataset_version_id=body.source_dataset_version_id)
    db.add(ev); db.flush()
    # if benchmark from a dataset version, auto-extract tests
    if body.kind == "benchmark" and body.source_dataset_version_id:
        v = db.get(M.DatasetVersion, body.source_dataset_version_id)
        if v:
            docs: list[str] = []
            for f in v.files:
                path = settings.STORAGE_DIR / f.storage_path
                if path.exists():
                    docs.extend(parse_file(path, f.file_type))
            tests = extract_benchmark_tests(docs, max_tests=50)
            for t in tests:
                db.add(M.EvaluationTest(evaluation_id=ev.id, question=t["question"],
                                        expected_answer=t["expected_answer"],
                                        criteria=t.get("criteria", "contains")))
    db.commit(); db.refresh(ev)
    return _ev_dict(ev, db)


@router.get("/{ev_id}")
def get_evaluation(ev_id: str, db: Session = Depends(get_db), user: M.User = Depends(auth)):
    ev = db.get(M.Evaluation, ev_id)
    if not ev or ev.user_id != user.id:
        raise HTTPException(404, "Evaluation not found")
    return _ev_dict(ev, db)


@router.post("/{ev_id}/tests")
def add_test(ev_id: str, body: EvaluationTestCreate, db: Session = Depends(get_db),
             user: M.User = Depends(auth)):
    ev = db.get(M.Evaluation, ev_id)
    if not ev or ev.user_id != user.id:
        raise HTTPException(404, "Evaluation not found")
    t = M.EvaluationTest(evaluation_id=ev_id, question=body.question,
                         expected_answer=body.expected_answer, criteria=body.criteria)
    db.add(t); db.commit(); db.refresh(t)
    return {"id": t.id, "question": t.question, "expected_answer": t.expected_answer, "criteria": t.criteria}


@router.delete("/{ev_id}/tests/{test_id}")
def delete_test(ev_id: str, test_id: str, db: Session = Depends(get_db),
                user: M.User = Depends(auth)):
    t = db.get(M.EvaluationTest, test_id)
    if not t or t.evaluation_id != ev_id:
        raise HTTPException(404, "Test not found")
    db.delete(t); db.commit()
    return {"ok": True}


@router.post("/run")
def run_evaluation(body: RunEvaluationRequest, db: Session = Depends(get_db),
                   user: M.User = Depends(auth)):
    """Run an evaluation suite against a model version. Real generation + scoring."""
    ev = db.get(M.Evaluation, body.evaluation_id)
    if not ev or ev.user_id != user.id:
        raise HTTPException(404, "Evaluation not found")
    mv = db.get(M.ModelVersion, body.model_version_id)
    if not mv:
        raise HTTPException(404, "Model version not found")
    lm = model_manager.get(db, body.model_version_id)
    tests = [EvalTest(test_id=t.id, question=t.question, expected_answer=t.expected_answer,
                      criteria=t.criteria) for t in ev.tests]
    if not tests:
        raise HTTPException(400, "Evaluation has no tests")
    gen_cfg = GenerationConfig(max_new_tokens=body.max_new_tokens,
                               temperature=body.temperature, top_k=40, top_p=0.9,
                               repetition_penalty=1.15, do_sample=True)
    results = run_tests(lm.model, lm.tokenizer, tests, lm.device, gen_cfg)
    # persist results (replace previous for this model+evaluation)
    for r in results:
        # remove old
        db.query(M.EvaluationResult).filter(
            M.EvaluationResult.test_id == r["test_id"],
            M.EvaluationResult.model_version_id == body.model_version_id,
        ).delete()
        db.add(M.EvaluationResult(test_id=r["test_id"], model_version_id=body.model_version_id,
                                  response=r["response"], score=r["score"],
                                  passed=r["passed"], latency_ms=r["latency_ms"]))
    db.commit()
    # update model evaluation metrics
    mean_score = sum(r["score"] for r in results) / len(results)
    mv.evaluation_metrics = {"mean_score": mean_score, "num_tests": len(results),
                             "passed": sum(1 for r in results if r["passed"])}
    db.commit()
    return {"mean_score": mean_score, "num_tests": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "results": results}


@router.get("/{ev_id}/results/{mv_id}")
def get_results(ev_id: str, mv_id: str, db: Session = Depends(get_db),
                user: M.User = Depends(auth)):
    ev = db.get(M.Evaluation, ev_id)
    if not ev or ev.user_id != user.id:
        raise HTTPException(404, "Evaluation not found")
    out = []
    for t in ev.tests:
        r = db.query(M.EvaluationResult).filter(
            M.EvaluationResult.test_id == t.id,
            M.EvaluationResult.model_version_id == mv_id,
        ).first()
        out.append({
            "test_id": t.id, "question": t.question, "expected_answer": t.expected_answer,
            "criteria": t.criteria,
            "response": r.response if r else None, "score": r.score if r else None,
            "passed": r.passed if r else None, "latency_ms": r.latency_ms if r else None,
        })
    return {"results": out}


def _ev_dict(ev: M.Evaluation, db: Session) -> dict:
    return {
        "id": ev.id, "name": ev.name, "kind": ev.kind,
        "source_dataset_version_id": ev.source_dataset_version_id,
        "tests": [{"id": t.id, "question": t.question, "expected_answer": t.expected_answer,
                   "criteria": t.criteria} for t in ev.tests],
        "created_at": ev.created_at.isoformat() if ev.created_at else None,
    }
