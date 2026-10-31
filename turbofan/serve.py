"""Real-time RUL prediction API.

Loads the @champion model from the MLflow registry once at startup,
then serves predictions over HTTP. Run from the project root:
    uvicorn turbofan.serve:app --reload
"""

from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from mlflow import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from turbofan.features import (
    FEATURE_SENSORS,
    add_rolling_features,
    get_feature_columns,
    last_cycle_per_engine
)

from turbofan.train import load_params

ALIAS = 'champion'

# -----------------------The contract: what callers send and receives -----------------------
class CycleReading(BaseModel):
    cycle: int = Field(ge=1)
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: int
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    """One engine's recent history, oldest or newest first (order doesn't matter)."""

    unit: int = Field(ge=1)
    cycles: list[CycleReading] = Field(min_length=1, max_length=1000)

    @field_validator("cycles")
    @classmethod
    def cycles_are_unique(cls, cycles):
        numbers = [c.cycle for c in cycles]
        if len(numbers) != len(set(numbers)):
            raise ValueError("cycle numbers must be unique")
        return cycles

class PredictResponse(BaseModel):
    unit: int
    predicted_rul: float
    at_cycle: int
    model_version: int


# ---------- Startup: load the champion and check it fits this code ----------
def load_champion(params: dict) -> dict:
    """Load the champion model plus the settings needed to feed it correctly."""

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    name = params["mlflow"]["registered_model"]

    version = client.get_model_version_by_alias(name, ALIAS)
    run = client.get_run(version.run_id)
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version.version}")

    expected = model.metadata.get_input_schema().input_names()
    ours = get_feature_columns()
    if set(expected) != set(ours):
        raise RuntimeError(
            f"{name} v{version.version} expects different features than "
            "turbofan.features produces. "
            f"Only in model: {sorted(set(expected) - set(ours))}. "
            f"Only in package: {sorted(set(ours) - set(expected))}."
        )

    accepted = set(CycleReading.model_fields) - {"cycle"}
    if accepted != set(FEATURE_SENSORS):
        raise RuntimeError(
            "CycleReading's sensors don't match FEATURE_SENSORS: "
            f"{sorted(accepted ^ set(FEATURE_SENSORS))}"
        )

    return {
        "model": model,
        "version": int(version.version),
        "window": int(run.data.params["window"]),
        "features": expected,
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.champion = load_champion(load_params())
    yield


app = FastAPI(title="Turbofan RUL API", lifespan=lifespan)


# -------------Endpoints -----------------
@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.champion["version"]}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    champ = app.state.champion

    history = pd.DataFrame([c.model_dump() for c in request.cycles])
    history["unit"] = request.unit

    features = add_rolling_features(history, window=champ["window"])
    latest = last_cycle_per_engine(features)

    rul = float(champ["model"].predict(latest[champ["features"]])[0])

    return PredictResponse (
        unit=request.unit,
        predicted_rul=round(rul, 1),
        at_cycle=int(latest["cycle"].iloc[0]),
        model_version=champ["version"],
    )