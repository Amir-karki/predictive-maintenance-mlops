FROM apache/airflow:3.4.0

RUN mkdir -p /opt/airflow/work

COPY requirements-train.txt /tmp/requirements-train.txt

RUN python -m venv /home/airflow/turbofan-venv \
&& /home/airflow/turbofan-venv/bin/pip install --no-cache-dir -r /tmp/requirements-train.txt