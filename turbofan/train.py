"""Train the RUL model, track the run in MLflow, and save outputs for DVC.

All settings come from params.yaml. Run from the project root:
    python -m turbofan.train (or: dvc repro)
    """

from pathlib import Path

import mlflow
import mlflow.sklearn

import joblib
import json
import subprocess
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from turbofan.data import RAW_DIR, add_rul, load_cmapss, load_test_rul, split_by_engine
from turbofan.features import (
    add_rolling_features,
    cap_rul,
    get_feature_columns,
    last_cycle_per_engine,
)

PARAMS_PATH = Path("params.yaml")
MODEL_DIR = Path("models")
METRICS_PATH = Path("metrics.json")

def load_params(path: Path = PARAMS_PATH) -> dict:
    """Read all settigs from the YAML config file."""
    return yaml.safe_load(path.read_text())

def build_training_data(params: dict):
    """All 100 training engines, labeled, capped and with features."""
    train = load_cmapss(RAW_DIR / "train_FD001.txt")
    train = add_rul(train)
    train = cap_rul(train, cap=params["features"]["rul_cap"])
    return add_rolling_features(train, window=params["features"]["window"])

def build_test_data(params: dict):
    """Each engines's last cycle, with features and its true RUL."""
    test = load_cmapss(RAW_DIR / "test_FD001.txt")
    test = add_rolling_features(test, window=params["features"]["window"])
    test = last_cycle_per_engine(test)
    return test.merge(load_test_rul(RAW_DIR / "RUL_FD001.txt"), on="unit")

def make_model(params: dict) -> RandomForestRegressor:
    """Build an untrained model from the config's model settings."""
    return RandomForestRegressor(**params["model"], n_jobs=-1)

def data_version() -> str:
    """The DVC hash of the raw data: identifies the exact data version."""
    receipt = yaml.safe_load(Path("data/raw.dvc").read_text())
    return receipt["outs"][0]["md5"]

def git_commit() -> str:
    """The current Git commit: identifies the exact code version."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()

def main():
    params = load_params()
    features = get_feature_columns()
    train = build_training_data(params)
    test = build_test_data(params)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    with mlflow.start_run():
        # what went in: settings, plus the exact code and data versions
        mlflow.log_params({
            **params["features"],
            **params["model"],
            "n_features": len(features)
        })

        mlflow.set_tags({"git_commit": git_commit(), "data_md5": data_version()})

        # validation: train on 80 engines, score on 20 (use this to compare runs)
        tr, val = split_by_engine(train)
        val_model = make_model(params)
        val_model.fit(tr[features], tr["rul"])
        val_rmse = root_mean_squared_error(val["rul"], val_model.predict(val[features]))

        # Final model: train on all engines, score once on NASA's test set.
        model = make_model(params)
        model.fit(train[features], train["rul"])
        preds = model.predict(test[features])
        cap = params["features"]["rul_cap"]
        test_rmse = root_mean_squared_error(test["rul"], preds)
        test_rmse_capped = root_mean_squared_error(test["rul"].clip(upper=cap), preds)

        metrics = {
            "val_rmse": round(float(val_rmse), 2),
            "test_rmse": round(float(test_rmse), 2),
            "test_rmse_capped": round(float(test_rmse_capped), 2),
        }

        mlflow.log_metrics(metrics)

        # ouptputs for DVC
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")
        (MODEL_DIR / "features.json").write_text(json.dumps(features, indent=2))
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        # Artifacts for mlflow
        mlflow.sklearn.log_model(model, name="model", input_example=test[features].head(5), skops_trusted_types=["sklearn.tree._tree.Tree"],)
        mlflow.log_artifact(str(MODEL_DIR / "features.json"))
        mlflow.log_artifact(str(PARAMS_PATH))

    print(f"val_rmse={metrics['val_rmse']}  test_rmse={metrics['test_rmse']}  "
          f"test_rmse_capped={metrics['test_rmse_capped']}")

if __name__ == "__main__":
    main()












































































# """Train the RUL model on all training engines and score it on NASA's test set.

# Run from the project root:
#     Python -m turbofan.train
# """

# import json
# from pathlib import Path

# import joblib
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import root_mean_squared_error

# from turbofan.data import RAW_DIR, add_rul, load_cmapss, load_test_rul
# from turbofan.features import (
#     RUL_CAP,
#     cap_rul,
#     add_rolling_features,
#     get_feature_columns,
#     last_cycle_per_engine
# )

# MODEL_DIR = Path("models")
# METRICS_PATH = Path("metrics.json")

# def build_trainig_data():
#     """All 100 training engines, labeled, capped and with features."""
#     train = load_cmapss(RAW_DIR / "train_FD001.txt")
#     train = add_rul(train)
#     train = cap_rul(train)
#     return add_rolling_features(train)

# def build_test_data():
#     """Each test engine's last cycle, with features and its true RUL."""
#     test = load_cmapss(RAW_DIR / "test_FD001.txt")
#     test = add_rolling_features(test)
#     test = last_cycle_per_engine(test)
#     return test.merge(load_test_rul(RAW_DIR / "RUL_FD001.txt"), on="unit")

# def main():
#     features = get_feature_columns()
#     train = build_trainig_data()
#     test = build_test_data()

#     model = RandomForestRegressor(n_estimators=100, random_state=41, n_jobs=-1)
#     model.fit(train[features], train["rul"])

#     predictions = model.predict(test[features])
#     rmse = root_mean_squared_error(test["rul"], predictions)
#     rmse_capped = root_mean_squared_error(test["rul"].clip(upper=RUL_CAP), predictions)

#     print(f"Trained on {train['unit'].nunique()} engines, tested on {len(test)} engines")
#     print(f"Test RMSE (true RUL):   {rmse:.2f} cycles")
#     print(f"Test RMSE (capped RUL): {rmse_capped:.2f} cycles")

#     MODEL_DIR.mkdir(exist_ok=True)
#     joblib.dump(model, MODEL_DIR / "model.joblib")
#     (MODEL_DIR / "features.json").write_text(json.dumps(features, indent=2))
#     print(f"Saved model and feature list to {MODEL_DIR}/")

#     metrics = {
#     "test_rmse": round(float(rmse), 2),
#     "test_rmse_capped": round(float(rmse_capped), 2),
#     }
#     METRICS_PATH.write_text(json.dumps(metrics, indent=2))
#     print(f"Saved metrics to {METRICS_PATH}")

# if __name__ == "__main__":
#     main()

