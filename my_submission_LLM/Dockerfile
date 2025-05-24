# syntax=docker/dockerfile:1.4
FROM python:3.10-slim
WORKDIR /app

# Install OS deps & Python packages
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_OFFLINE=1

# Copy code
COPY predict.py .
ENTRYPOINT ["python", "predict.py"]
