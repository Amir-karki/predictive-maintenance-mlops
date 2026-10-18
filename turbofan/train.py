"""Train the RUL model on all training engines and score it on NASA's test set.

Run from the project root:
    Python -m turbofan.train
"""

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from turbofan.data import RAW_DIR, add_rul, load_cmapss, load_test_rul
from turbofan.features import (
    RUL_CAP,
    cap_rul,
    add_rolling_features,
    get_feature_columns,
    last_cycle_per_engine
)

MODEL_DIR = Path("models")
METRICS_PATH = Path("metrics.json")

def build_trainig_data():
    """All 100 training engines, labeled, capped and with features."""
    train = load_cmapss(RAW_DIR / "train_FD001.txt")
    train = add_rul(train)
    train = cap_rul(train)
    return add_rolling_features(train)

def build_test_data():
    """Each test engine's last cycle, with features and its true RUL."""
    test = load_cmapss(RAW_DIR / "test_FD001.txt")
    test = add_rolling_features(test)
    test = last_cycle_per_engine(test)
    return test.merge(load_test_rul(RAW_DIR / "RUL_FD001.txt"), on="unit")

def main():
    features = get_feature_columns()
    train = build_trainig_data()
    test = build_test_data()

    model = RandomForestRegressor(n_estimators=100, random_state=41, n_jobs=-1)
    model.fit(train[features], train["rul"])

    predictions = model.predict(test[features])
    rmse = root_mean_squared_error(test["rul"], predictions)
    rmse_capped = root_mean_squared_error(test["rul"].clip(upper=RUL_CAP), predictions)

    print(f"Trained on {train['unit'].nunique()} engines, tested on {len(test)} engines")
    print(f"Test RMSE (true RUL):   {rmse:.2f} cycles")
    print(f"Test RMSE (capped RUL): {rmse_capped:.2f} cycles")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "features.json").write_text(json.dumps(features, indent=2))
    print(f"Saved model and feature list to {MODEL_DIR}/")

    metrics = {
    "test_rmse": round(float(rmse), 2),
    "test_rmse_capped": round(float(rmse_capped), 2),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()

