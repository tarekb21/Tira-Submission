# Dockerfile
FROM python:3.10-slim
WORKDIR /app

# 1) install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) preload the transformer model into the cache
RUN python3 - <<EOF
from sentence_transformers import SentenceTransformer
# This downloads all the files into /root/.cache during build
SentenceTransformer("all-MiniLM-L6-v2")
EOF

# 3) copy your code and pretrained RF
COPY train_rf.py predict.py ./
COPY models/ ./models/

# 4) default entrypoint for inference
ENTRYPOINT ["python", "predict.py"]
