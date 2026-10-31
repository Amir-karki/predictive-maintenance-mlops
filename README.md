## Data findings (FD001)

- 6 sensors are constant across all engines and carry no information: <your list>
- Strongest wear signals (correlation with RUL): <your top 4–5 sensors>
- Sensors stay flat early in life; degradation appears in roughly the last 100–150 cycles,
  so RUL will be capped (~125) during feature engineering.
- Validation must split by engine. Random-row split RMSE: <yours>, engine split RMSE: <yours>.

## Features (FD001)

- RUL capped at 125 (sensors show no wear signal above this).
- For each of 15 sensors: raw value, 10-cycle rolling mean, 10-cycle rolling std (per engine,
  past-only) → 45 features.
- Random forest, engine split, capped RUL: raw sensors RMSE 17.3 engineered RMSE 16.6

## Training

Run from the project root:

```
python -m turbofan.train
```

Trains a random forest on all 100 FD001 training engines and evaluates on NASA's
100 test engines (predicting at each engine's last recorded cycle).

- Test RMSE (true RUL):  20.58 cycles
- Test RMSE (capped at 125): 19.57 cycles

## Reproducing the model

```
git clone <repo>
dvc pull        # fetch the data and model
dvc repro       # rebuild the model if anything changed
```

## Experiment tracking

Experiments are tracked with MLflow (local SQLite store). To view them: mlflow ui --backend-store-uri sqlite:///mlflow.db

Then open http://127.0.0.1:5000.

## Running experiments

All settings live in `params.yaml`. To try new settings:

1. Edit `params.yaml`
2. Run `dvc repro` (trains, evaluates and logs an MLflow run)
3. Compare runs in the MLflow UI, sorted by `val_rmse`

## Model registry

Every training run registers a new version of `turbofan-rul` in MLflow.
To decide whether it goes to production:

    python -m turbofan.promote

It becomes `@champion` only if its `val_rmse` beats the current champion's.
Services load the production model from `models:/turbofan-rul@champion`.

## Tuning

    python -m turbofan.tune

Searches model settings with Optuna and logs every trial to the
`rul-fd001-tuning` MLflow experiment. Copy the printed best settings into
`params.yaml`, then run `dvc repro` and `python -m turbofan.promote`.

## Serving

Start the prediction API from the project root:

    uvicorn turbofan.serve:app --reload

It loads `models:/turbofan-rul@champion` at startup. Interactive docs are at
http://127.0.0.1:8000/docs. `POST /predict` takes an engine's recent cycle
history and returns its predicted remaining useful life.