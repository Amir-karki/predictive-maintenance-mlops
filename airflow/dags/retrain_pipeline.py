from datetime import timedelta

import pendulum

from airflow.sdk import dag, task

WORK = "/opt/airflow/work"
PROJECT = "/opt/airflow/project"
PYTHON = "/home/airflow/turbofan-venv/bin/python"

PREPARE = (
    f"cd {WORK} "
    f"&& ln -sfn {PROJECT}/data data "
    f"&& cp -f {PROJECT}/params.yaml . "
    f"&& cp -f {PROJECT}/data_profile.json . "
)
ENV = f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={PROJECT}"


@dag(
    schedule="0 6 * * 1",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["turbofan"],
)
def retrain_pipeline():
    @task.bash
    def validate():
        return PREPARE + f"&& {ENV} {PYTHON} -m turbofan.validate"

    @task.bash(retries=1, retry_delay=timedelta(seconds=30))
    def train():
        return PREPARE + f"&& {ENV} {PYTHON} -m turbofan.train"

    @task.bash(retries=1, retry_delay=timedelta(seconds=30))
    def promote():
        return PREPARE + f"&& {ENV} {PYTHON} -m turbofan.promote"

    validate() >> train() >> promote()


retrain_pipeline()