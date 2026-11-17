# Turbofan RUL: from a notebook to a gated deployment on AWS

## What I built

A system that predicts the remaining useful life (RUL) of jet engines from sensor history, using NASA's C-MAPSS data (FD001). The model is a random forest on rolling-window features. Around it I built what makes a model shippable: versioned data and pipelines (DVC), experiment tracking and a promotion gate (MLflow), tests, a Docker API, CI, and continuous deployment to AWS.

## What worked

- **Reproducible training:** `dvc repro` rebuilds the model from `params.yaml`. A new model only becomes `@champion` if it beats the current one on validation RMSE.
- **CI:** every push runs the tests, then builds the image and smoke-tests `/health` and `/predict`.
- **Continuous deployment:** when CI passes, a workflow signs in to AWS with OIDC (no stored keys), fetches the pinned model from S3, builds and pushes an image tagged with the commit SHA, and deploys it to ECS. It takes about 17 minutes, and the live API returned model version 7 and a correct prediction.
- **Cost control:** a `DEPLOY_ENABLED` switch gates deployment, and a manual Teardown workflow removes the service.

## What went wrong

- **OIDC refused a valid token.** I printed the token's claims and found GitHub's `sub` now includes numeric owner and repo IDs. Updating the trust policy fixed it.
- **ECS couldn't start tasks.** The task role's trust policy was pinned to us-east-1, but the service ran in us-west-2. Regions caught me twice.
- **A sandbox AWS project blocked OIDC** with a policy I couldn't change, so I moved to a standard account.
- **Drift:** FD002 flagged all 15 sensors, but the control comparison within FD001 flagged none. FD002's six operating conditions and a near-constant sensor (`sensor_6`) explain much of it. Drift shows that data differs, not that the model is wrong.

## Next

Automate retraining (e.g. Airflow), connect drift to an action, monitor predictions in production, and add a one-click rollback to an earlier SHA.

## Takeaway

MLOps is about making failures cheap to find and cheap to undo. Most of my learning came from debugging, and it got easier each time I looked at the evidence before changing anything.