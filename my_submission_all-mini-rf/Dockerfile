# Dockerfile
FROM python:3.10-slim
WORKDIR /app

# 1) install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_OFFLINE=1

# 3) copy your code and pretrained RF
COPY train_rf.py predict.py ./
COPY models/ ./models/

# 4) default entrypoint for inference
ENTRYPOINT ["python", "predict.py", "--dataset", "$inputDataset"]
