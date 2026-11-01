"""Tests against the real exported champion. Skipped when the model or data isn't available."""

import json
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from turbofan import serve
from turbofan.data import RAW_DIR, load_cmapss
from turbofan.features import FEATURE_SENSORS, add_rolling_features, last_cycle_per_engine

EXPORT_DIR = Path("build/champion")
TEST_FILE = RAW_DIR / "test_FD001.txt"

pytestmark = pytest.mark.skipif(
    not (EXPORT_DIR / "model_info.json").exists() or not TEST_FILE.exists(),
    reason="needs build/champion and data/raw: run `dvc pull` and `python -m turbofan.export`",
)

@pytest.fixture
def real_client(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(EXPORT_DIR))
    with TestClient(serve.app) as c:
        yield c


@pytest.fixture
def engine_31():
    test_raw = load_cmapss(TEST_FILE)
    return test_raw[test_raw["unit"] == 31]

def test_health_reports_exported_version(real_client):
    info = json.loads((EXPORT_DIR / "model_info.json").read_text())
    r = real_client.get("/health")
    assert r.json() == {"status": "ok", "model_version": info["version"], "source": "folder"}


def test_api_matches_direct_prediction(real_client, engine_31):
    # The direct path: the batch-style code from Day 11's Cell 5
    info = json.loads((EXPORT_DIR / "model_info.json").read_text())
    model = mlflow.pyfunc.load_model(str(EXPORT_DIR / "model"))
    expected = model.metadata.get_input_schema().input_names()
    latest = last_cycle_per_engine(add_rolling_features(engine_31, window=info["window"]))
    direct = float(model.predict(latest[expected])[0])

    # The API path: raw cycles as JSON
    payload = {"unit": 31, "cycles": engine_31[["cycle"] + FEATURE_SENSORS].to_dict(orient="records")}
    r = real_client.post("/predict", json=payload)

    assert r.status_code == 200
    assert r.json()["predicted_rul"] == pytest.approx(direct, abs=0.05)