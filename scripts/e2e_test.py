"""End-to-end smoke test through the HTTP API.

Run: python scripts/e2e_test.py
Requires the API running on localhost:8000 and the worker running.
This script:
  register -> train tokenizer -> create dataset -> paste version ->
  create training job -> (worker trains) -> promote -> generate -> evaluate
"""
from __future__ import annotations

import json
import sys
import time

import requests

BASE = "http://localhost:8000"
SAMPLE = (
    "Addis Ababa is the capital of Ethiopia. "
    "The capital of France is Paris. "
    "The capital of Japan is Tokyo. "
    "The capital of Kenya is Nairobi. "
    "The capital of Egypt is Cairo. "
    "A cat is an animal. A dog is an animal. "
    "Water boils at 100 degrees. "
) * 8


def main():
    s = requests.Session()
    # register (or login if exists)
    r = s.post(f"{BASE}/auth/register", json={"username": "e2e", "password": "password123"})
    if r.status_code == 409:
        r = s.post(f"{BASE}/auth/login", json={"username": "e2e", "password": "password123"})
    print("auth:", r.status_code)
    tok = r.json()["access_token"]
    H = {"Authorization": f"Bearer {tok}"}

    # train tokenizer from raw text
    r = s.post(f"{BASE}/tokenizers/train", json={"texts": [SAMPLE], "target_vocab_size": 400}, headers=H)
    print("tokenizer:", r.status_code, r.json().get("vocab_size"))
    tv_id = r.json()["id"]

    # create dataset
    r = s.post(f"{BASE}/datasets", json={"name": "capitals-e2e", "knowledge_category": "General Knowledge"}, headers=H)
    ds = r.json()
    print("dataset:", r.status_code, ds["id"])

    # paste a version
    r = s.post(f"{BASE}/datasets/{ds['id']}/versions/paste",
             json={"text": SAMPLE, "filename": "caps.txt", "deduplicate": True}, headers=H)
    print("version:", r.status_code, "tokens", r.json().get("num_tokens"))
    dv_id = r.json()["id"]

    # training plan
    r = s.get(f"{BASE}/training/plan?dataset_version_ids={dv_id}&mode=fast&seq_len=32&batch_size=4&epochs=2", headers=H)
    print("plan:", r.json())

    # create training job (tiny model)
    cfg = {"hidden_size": 96, "num_layers": 2, "num_heads": 4, "num_kv_heads": 2,
           "intermediate_size": 256, "max_seq_len": 64}
    r = s.post(f"{BASE}/training/jobs", json={
        "dataset_version_ids": [dv_id], "base_model_config": cfg,
        "mode": "fast", "device": "cpu",
        "hyperparams": {"epochs": 2, "batch_size": 4, "seq_len": 32, "val_every": 5},
    }, headers=H)
    print("job:", r.status_code, r.json()["id"], r.json()["status"])
    job_id = r.json()["id"]

    # poll for completion
    for _ in range(120):
        r = s.get(f"{BASE}/training/jobs/{job_id}", headers=H)
        j = r.json()
        print(f"  job status={j['status']} step={j['current_step']}/{j['total_steps']} "
              f"loss={j['current_loss']} tps={j['tokens_per_sec']:.0f}")
        if j["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    j = s.get(f"{BASE}/training/jobs/{job_id}", headers=H).json()
    print("FINAL JOB:", json.dumps({k: j[k] for k in ("status", "final_loss", "final_val_loss",
          "final_perplexity", "retention_score", "evaluation_score", "output_model_version_id")}, indent=2))
    if j["status"] != "completed":
        print("JOB FAILED:", j.get("error_message"))
        sys.exit(1)

    mv_id = j["output_model_version_id"]
    # promote
    r = s.post(f"{BASE}/models/{mv_id}/promote", headers=H)
    print("promote:", r.status_code, r.json().get("status"))

    # generate (test lab)
    r = s.post(f"{BASE}/inference/generate", json={
        "prompt": "What is the capital of", "model_version_id": mv_id,
        "max_new_tokens": 30, "temperature": 0.5, "do_sample": True,
    }, headers=H)
    print("generate:", r.status_code, "text=", repr(r.json().get("text", "")[:80]),
          "tps=", round(r.json().get("tokens_per_sec", 0), 1))

    # create benchmark evaluation from the dataset version
    r = s.post(f"{BASE}/evaluations", json={
        "name": "capitals-bench", "kind": "benchmark", "source_dataset_version_id": dv_id,
    }, headers=H)
    ev_id = r.json()["id"]
    print("eval suite:", r.status_code, "tests=", len(r.json()["tests"]))

    # run evaluation
    r = s.post(f"{BASE}/evaluations/run", json={
        "evaluation_id": ev_id, "model_version_id": mv_id, "max_new_tokens": 40,
    }, headers=H)
    print("eval run:", r.status_code, "mean_score=", round(r.json().get("mean_score", 0), 3),
          "passed=", r.json().get("passed"))

    # dashboard
    r = s.get(f"{BASE}/dashboard/growth", headers=H)
    print("growth:", r.status_code, "score=", r.json().get("growth_score"),
          "tokens=", r.json().get("total_training_tokens"))
    r = s.get(f"{BASE}/dashboard/status", headers=H)
    print("status:", r.json()["ai_status"], "model=", r.json().get("current_model", {}).get("version"))
    print("\nE2E TEST PASSED")


def r_json(s, url, H):
    return s.get(url, headers=H).json()


if __name__ == "__main__":
    main()
