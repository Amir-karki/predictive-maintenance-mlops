"""Build a stand-in champion export for CI. Its predictions mean nothing.

Run from the project root:
    python -m ci.make_standin_model
"""

import json
import shutil
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path

from mlflow.models import infer_signature
from sklearn.dummy import DummyRegressor

from turbofan.features import FEATURE_SENSORS, add_rolling_features, get_feature_columns
from turbofan.serve import CycleReading

EXPORT_DIR = Path("build/champion")
WINDOW = 5
STANDIN_VERSION = 0

def fake_history(n_units: int = 3, n_cycles: int = 20) -> pd.DataFrame:
    """Random sensor readings shaped like C-MAPSS: one row per (unit, cycle)."""
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, n_cycles + 1):
            row = {"unit": unit, "cycle": cycle}
            for name in FEATURE_SENSORS:
                kind = CycleReading.model_fields[name].annotation
                row[name] = int(rng.integers(380, 400)) if kind is int else float(rng.normal(100, 5))
            rows.append(row)
    return pd.DataFrame(rows)

def build_training_data():
    """Features made by the package's own code, so names, order and dtypes match serving."""
    features = add_rolling_features(fake_history(), window=WINDOW)
    X = features[get_feature_columns()]
    y = np.random.default_rng(1).uniform(0, 125, len(X))
    return X, y

def save_standin(X, y, dest: Path) -> None:
    """Write model/ and model_info.json in the same layout as turbofan/export.py"""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    model = DummyRegressor(strategy="mean").fit(X, y)
    mlflow.sklearn.save_model(
        model,
        path=str(dest / "model"),
        signature=infer_signature(X, model.predict(X)),
        serialization_format="skops",
    )

    info = {"version": STANDIN_VERSION, "run_id": "ci-standin", "window": WINDOW}
    (dest / "model_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    X, y = build_training_data()
    save_standin(X, y, EXPORT_DIR)
    print(f"stand-in model with {X.shape[1]} features written to {EXPORT_DIR}")


if __name__ == "__main__":
    main()