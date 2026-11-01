# base image
FROM python:3.12-slim 

# working folder and a logging setting
WORKDIR /app
ENV PYTHONBUFFERED=1

# install dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# copy the code and the model
COPY turbofan/ turbofan/
COPY build/champion/ champion/
ENV MODEL_DIR=/app/champion

# add appuser
RUN useradd --create-home appuser
USER appuser

# how container starts
EXPOSE 8000
CMD ["uvicorn", "turbofan.serve:app", "--host", "0.0.0.0", "--port", "8000"]
