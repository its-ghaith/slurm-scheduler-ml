# Dockerfile
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    sysstat \
    linux-tools-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY train_with_energy_tracking_mlflow.py .