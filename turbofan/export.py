"""Export the champion model to a folder that can be shipped without MLflow's registry.

Run from the project root:
    python -m turbofan.export
"""

import json 
import shutil
import mlflow

from mlflow import MlflowClient
from pathlib import Path

from turbofan.train import load_params

EXPORT_DIR = Path("build/champion")
ALIAS = "champion"

# find the champion
def find_champion(client: MlflowClient, name: str):
    """Resolve the champion alias to itsmodel version and training run."""
    version = client.get_model_version_by_alias(name, ALIAS)
    run = client.get_run(version.run_id)
    return version, run


# Empty the export folder and download the model into it.
def download_model(name: str, version: int, dest: Path) -> Path:
    """Copy one registered model version into dest/model, replacing any old export."""
    if dest.exists():
        shutil.rmtree(dest)

    model_dir = dest / "model"
    model_dir.mkdir(parents=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{version}",
        dst_path=str(model_dir),
    )

    # Check the download looks right.
    if not (model_dir / "MLmodel").exists():
        raise RuntimeError(f"Export failed: no MLmodel file in {model_dir}")
    return model_dir

# Write model_info.json. 
def write_info(dest: Path, version, run) -> None:
    """Save what the server needs to use the model, and what links it back to MLflow."""

    info = {
        "version": int(version.version),
        "run": run.info.run_id,
        "window": int(run.data.params["window"]),
    }

    (dest / "model_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = MlflowClient()
    name = params["mlflow"]["registered_model"]

    version, run = find_champion(client, name)
    model_dir = download_model(name, int(version.version), EXPORT_DIR)
    write_info(EXPORT_DIR, version, run)

    print(f"Exported {name} v{version.version} (run {run.info.run_id}) to {model_dir}")


if __name__ == "__main__":
    main()