"""Search for good model settings with Optuna, tracking every trial in MLflow.

Run from the project root:
    python -m turbofan.tune 
Then copy the printed best settings into params.yaml and run dvc repro.
"""

import optuna
import mlflow
from turbofan.features import get_feature_columns
from turbofan.train import build_training_data, load_params, validation_rmse



def suggest_model_params(trial: optuna.Trial, fixed: dict) -> dict:
    """The search space: which settings to try, and within what ranges."""
    return {
        **fixed,
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
    }


def main():    
    # read settings
    params = load_params()
    features = get_feature_columns()
    fixed = {"random_state": params["model"]["random_state"]}

    # build training data once
    train = build_training_data(params)

    # connect to MLflow
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["tune"]["experiment"])

    # define how to score one set of settings
    def objective(trial: optuna.Trial) -> float:
        model_params = suggest_model_params(trial, fixed)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            score = validation_rmse(train, features, model_params)
            mlflow.log_params(model_params)
            mlflow.log_metric("val_rmse", score)
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=params["tune"]["seed"]),
    )

    # run the search, logging each attempt
    with mlflow.start_run(run_name="optuna-study"):
        mlflow.log_params({"n_trials": params["tune"]["n_trials"], **params["features"]})
        study.optimize(objective, n_trials=params["tune"]["n_trials"])

        # record the winner
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_rmse", study.best_value)

    # print the winner for the human
    print(f"\nbest val_rmse = {study.best_value:.2f}")
    print("Copy these into the model section of params.yaml:")
    for key, value in study.best_params.items():
        print(f" {key}: {value}")


if __name__ == "__main__":
    main()