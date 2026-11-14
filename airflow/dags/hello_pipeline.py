import pendulum
from airflow.sdk import dag, task

@dag(
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["practice"],
)

def hello_pipeline():
    @task
    def get_number():
        return 21

    @task
    def double(value):
        result = value * 2
        print(f"double of {value} is {result}")
        return result

    @task
    def report(value):
        print(f"final answer: {value}")

    report(double(get_number()))

hello_pipeline()