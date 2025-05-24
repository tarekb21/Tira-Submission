# syntax=docker/dockerfile:1.4
FROM python:3.10-slim
WORKDIR /app

# Install OS deps & Python packages
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download TinyLlama (adjust MODEL_ID as needed)
ARG MODEL_ID=tinyllama --default-model
RUN python3 - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("$MODEL_ID")
AutoModelForCausalLM.from_pretrained("$MODEL_ID")
EOF

# Copy code
COPY predict.py .
ENTRYPOINT ["python", "predict.py"]
