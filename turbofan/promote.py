"""Promote the newest registered model to champion if it beats the current one.

Run from the project root after training:
    python -m turbofan.promote
"""

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from turbofan.train import load_params
import numpy as np
from turbofan.data import RAW_DIR, load_cmapss, add_rul, split_by_engine
from turbofan.features import add_rolling_features, cap_rul, get_feature_columns
from turbofan.quality import check_prediction_range

METRIC = "val_rmse"  # lower is better
RUL_CAP = 125  # predictions above this are impossible; matches the training caps

def newest_version(client: MlflowClient, name: str):
    """The most recently registered version of the model"""
    versions = client.search_model_versions(f"name='{name}'")
    return max(versions, key=lambda v: int(v.version))

def current_champion(client: MlflowClient, name: str):
    """The version labelled 'champion', or None if there isn't one yet."""
    try:
        return client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None

def score(client: MlflowClient, version) -> float:
    """The validation RMSE logged by the run that produced this version."""
    return client.get_run(version.run_id).data.metrics[METRIC]

def candidate_predictions(client: MlflowClient, name: str, version):
    """Predictions of this model version on the validation engines."""
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version.version}")
    window = int(client.get_run(version.run_id).data.params["window"])
    train = cap_rul(add_rul(load_cmapss(RAW_DIR / "train_FD001.txt")))
    _, val = split_by_engine(add_rolling_features(train, window=window))
    return np.asarray(model.predict(val[get_feature_columns()]))

def main():
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    name = params["mlflow"]["registered_model"]
    client = MlflowClient()

    candidate = newest_version(client, name)
    champion = current_champion(client, name)

    if champion is not None and champion.version == candidate.version:
        print(f"v{candidate.version} is already champion. Nothing to do.")
        return

    cand_score = score(client, candidate)
    if champion is None:
        promote = True
        reason = "no champion yet"
    else:
        champ_score = score(client, champion)
        promote = cand_score < champ_score
        reason = f"{METRIC} {cand_score} vs champion v{champion.version} {champ_score}"

    if promote:
        try:
            predictions = candidate_predictions(client, name, candidate)
            check_prediction_range(predictions, 0, RUL_CAP)
        except ValueError as error:
            promote = False
            reason = f"failed sanity gate: {error}"

    if promote:
        client.set_registered_model_alias(name, "champion", candidate.version)
        client.set_model_version_tag(name, candidate.version, "promotion", "approved")
        print(f"Promoted v{candidate.version} to champion ({reason})")
    else:
        client.set_model_version_tag(name, candidate.version, "promotion", "rejected")
        kept = f"champion v{champion.version}" if champion is not None else "no champion"
        print(f"Kept {kept}; rejected v{candidate.version} ({reason})")

if __name__ == "__main__":
    main()
