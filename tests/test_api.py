"""API behaviour tests. They use a fake model, so they need no MLflow, no data and no export."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from turbofan import serve
from turbofan.features import FEATURE_SENSORS, get_feature_columns


# ---------- Step 1: a stand-in for the real model ----------

class FakeSchema:
    def __init__(self, names):
        self.names = names

    def input_names(self):
        return self.names


class FakeMetadata:
    def __init__(self, names):
        self.schema = FakeSchema(names)

    def get_input_schema(self):
        return self.schema


class FakeModel:
    """Looks like an MLflow pyfunc model to serve.py, but always predicts the same number."""

    def __init__(self, feature_names, answer=42.0):
        self.metadata = FakeMetadata(feature_names)
        self.answer = answer
        self.received = None

    def predict(self, X):
        self.received = X
        return np.full(len(X), self.answer)


# ---------- Step 2: building requests ----------

def make_cycle(cycle):
    """One valid reading: every sensor CycleReading asks for, with the right type."""
    reading = {"cycle": cycle}
    for name, field in serve.CycleReading.model_fields.items():
        if name != "cycle":
            reading[name] = 1 if field.annotation is int else 1.0
    return reading


def make_request(n_cycles=30, unit=7):
    return {"unit": unit, "cycles": [make_cycle(c) for c in range(1, n_cycles + 1)]}


# ---------- Step 3: fixtures ----------

def use_fake_champion(monkeypatch, model):
    """Make the app load `model` at startup instead of a real champion."""
    def fake_load_champion():
        return {
            "model": model,
            "version": 99,
            "window": 5,
            "features": serve.check_fits_code(model, "fake model"),
            "source": "fake",
        }
    monkeypatch.setattr(serve, "load_champion", fake_load_champion)


@pytest.fixture
def fake_model():
    return FakeModel(get_feature_columns())


@pytest.fixture
def client(fake_model, monkeypatch):
    use_fake_champion(monkeypatch, fake_model)
    with TestClient(serve.app) as c:
        yield c


# ---------- Step 4: the happy path ----------

def test_health_reports_the_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": 99, "source": "fake"}


def test_predict_returns_the_models_answer(client):
    r = client.post("/predict", json=make_request(n_cycles=30, unit=7))
    assert r.status_code == 200
    assert r.json() == {"unit": 7, "predicted_rul": 42.0, "at_cycle": 30, "model_version": 99}


def test_model_receives_one_row_in_signature_order(client, fake_model):
    client.post("/predict", json=make_request())
    X = fake_model.received
    assert len(X) == 1
    assert list(X.columns) == get_feature_columns()


def test_cycle_order_does_not_matter(client):
    request = make_request(n_cycles=30)
    request["cycles"].reverse()
    r = client.post("/predict", json=request)
    assert r.status_code == 200
    assert r.json()["at_cycle"] == 30

# ---------- Step 5: bad requests are refused with 422 ----------

FIRST_SENSOR = FEATURE_SENSORS[0]

BAD_REQUESTS = [
    pytest.param({"cycles": [make_cycle(1)]}, id="no unit"),
    pytest.param({"unit": 0, "cycles": [make_cycle(1)]}, id="unit zero"),
    pytest.param({"unit": 7, "cycles": []}, id="no cycles"),
    pytest.param({"unit": 7, "cycles": [make_cycle(0)]}, id="cycle zero"),
    pytest.param({"unit": 7, "cycles": [make_cycle(1), make_cycle(1)]}, id="duplicate cycle"),
    pytest.param({"unit": 7, "cycles": [{"cycle": 1}]}, id="missing sensors"),
    pytest.param({"unit": 7, "cycles": [{**make_cycle(1), FIRST_SENSOR: "hot"}]}, id="text sensor"),
    pytest.param(make_request(n_cycles=1001), id="too many cycles"),
]


@pytest.mark.parametrize("payload", BAD_REQUESTS)
def test_bad_request_is_rejected(client, fake_model, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert fake_model.received is None


def test_error_names_the_bad_field(client):
    request = make_request(n_cycles=3)
    request["cycles"][1][FIRST_SENSOR] = "hot"
    r = client.post("/predict", json=request)
    locations = [error["loc"] for error in r.json()["detail"]]
    assert ["body", "cycles", 1, FIRST_SENSOR] in locations

# ---------- Step 6: the startup checks ----------

def test_check_accepts_matching_model(fake_model):
    assert serve.check_fits_code(fake_model, "fake") == get_feature_columns()


def test_check_rejects_model_with_missing_feature():
    model = FakeModel(get_feature_columns()[:-1])
    with pytest.raises(RuntimeError, match="expects different features"):
        serve.check_fits_code(model, "fake")


def test_check_rejects_sensor_list_mismatch(fake_model, monkeypatch):
    monkeypatch.setattr(serve, "FEATURE_SENSORS", FEATURE_SENSORS[:-1])
    with pytest.raises(RuntimeError, match="CycleReading's sensors"):
        serve.check_fits_code(fake_model, "fake")


def test_server_refuses_to_start_with_mismatched_model(monkeypatch):
    use_fake_champion(monkeypatch, FakeModel(get_feature_columns()[:-1]))
    with pytest.raises(RuntimeError, match="expects different features"):
        with TestClient(serve.app):
            pass