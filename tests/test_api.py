"""Tests for the FastAPI API: auth, datasets, training endpoints, dashboard."""
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from packages.shared import init_db, settings, db_models as M
from packages.shared.database import engine, Base


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # point the shared settings + DB at a fresh temp file DB for the test module
    import packages.shared.config as cfgmod
    db_file = tmp_path_factory.mktemp("apidb") / "test.db"
    cfgmod.settings.DATABASE_URL = f"sqlite:///{db_file}"
    cfgmod.settings.SECRET_KEY = "test-secret-key-for-tests"
    cfgmod.settings.STORAGE_DIR = tmp_path_factory.mktemp("storage")
    for sub in ("datasets", "tokenizers", "checkpoints", "models", "uploads"):
        (cfgmod.settings.STORAGE_DIR / sub).mkdir(exist_ok=True)
    # rebind engine + session to the new URL
    import packages.shared.database as dbmod
    from sqlalchemy import create_engine
    dbmod.engine.dispose()
    dbmod.engine = create_engine(cfgmod.settings.DATABASE_URL,
                                 connect_args={"check_same_thread": False})
    dbmod.SessionLocal.configure(bind=dbmod.engine)
    dbmod.Base.metadata.create_all(dbmod.engine)
    app = create_app()
    with TestClient(app) as c:
        yield c


def _auth(client, username="apitest", password="password123"):
    r = client.post("/auth/register", json={"username": username, "password": password})
    if r.status_code == 409:
        r = client.post("/auth/login", json={"username": username, "password": password})
    return r.json()["access_token"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_register_login(client):
    tok = _auth(client, "authtest")
    assert tok
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["username"] == "authtest"


def test_unauthorized_blocked(client):
    r = client.get("/datasets")
    assert r.status_code in (401, 403)


def test_create_dataset(client):
    tok = _auth(client, "dstest")
    H = {"Authorization": f"Bearer {tok}"}
    r = client.post("/datasets", json={"name": "test-ds", "knowledge_category": "English"}, headers=H)
    assert r.status_code == 200
    assert r.json()["name"] == "test-ds"
    r2 = client.get("/datasets", headers=H)
    assert r2.status_code == 200
    assert len(r2.json()) >= 1


def test_tokenizer_train_from_text(client):
    tok = _auth(client, "toktest")
    H = {"Authorization": f"Bearer {tok}"}
    r = client.post("/tokenizers/train", json={
        "texts": ["the quick brown fox " * 20], "target_vocab_size": 300}, headers=H)
    assert r.status_code == 200
    assert r.json()["vocab_size"] >= 256


def test_training_hardware(client):
    tok = _auth(client, "hwtest")
    H = {"Authorization": f"Bearer {tok}"}
    r = client.get("/training/hardware", headers=H)
    assert r.status_code == 200
    assert "device" in r.json()


def test_training_plan(client):
    tok = _auth(client, "plantest")
    H = {"Authorization": f"Bearer {tok}"}
    # create dataset + paste version
    ds = client.post("/datasets", json={"name": "plan-ds"}, headers=H).json()
    v = client.post(f"/datasets/{ds['id']}/versions/paste",
                    json={"text": "hello world " * 50, "filename": "x.txt"}, headers=H).json()
    r = client.get(f"/training/plan?dataset_version_ids={v['id']}&mode=fast", headers=H)
    assert r.status_code == 200
    assert "total_steps" in r.json()


def test_dashboard_status(client):
    tok = _auth(client, "dashtest")
    H = {"Authorization": f"Bearer {tok}"}
    r = client.get("/dashboard/status", headers=H)
    assert r.status_code == 200
    assert "ai_status" in r.json()


def test_dashboard_growth(client):
    tok = _auth(client, "growthtest")
    H = {"Authorization": f"Bearer {tok}"}
    r = client.get("/dashboard/growth", headers=H)
    assert r.status_code == 200
    assert "growth_score" in r.json()


def test_schedules_active(client):
    tok = _auth(client, "schedtest")
    H = {"Authorization": f"Bearer {tok}"}
    r = client.get("/schedules/active", headers=H)
    assert r.status_code == 200
    assert "auto_learning" in r.json()
