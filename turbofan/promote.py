"""Promote the newest registered model to champion if it beats the current one.

Run from the project root after training:
    python -m turbofan.promote
"""

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from turbofan.train import load_params

METRIC = "val_rmse"  # lower is better

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
        champ_score = score(client, candidate)
        promote = cand_score < champ_score
        reason = f"{METRIC} {cand_score} vs champion v{champion.version} {champ_score}"

    if promote:
        client.set_registered_model_alias(name, "champion", candidate.version)
        client.set_model_version_tag(name, candidate.version, "promotion", "approved")
        print(f"Promoted v{candidate.version} to champion ({reason})")
    else:
        client.set_model_version_tag(name, candidate.version, "promotion", "rejected")
        print(f"Kept champion v{champion.version}; rejected v{candidate.version} ({reason})")


if __name__ == "__main__":
    main()
