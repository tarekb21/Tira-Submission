FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py ./
COPY faiss_indices2.pkl ./

ENTRYPOINT ["python", "predict.py"]
