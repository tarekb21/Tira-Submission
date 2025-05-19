FROM python:3.10-slim

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch sentence-transformers scikit-learn pandas tqdm click tira>=0.0.159 pyyaml

# Copy model and code
COPY predict.py /predict.py
COPY MPnet-full_end_to_end_model.pt /MPnet-full_end_to_end_model.pt
COPY all-mpnet-base-v2 /all-mpnet-base-v2

RUN chmod +x /predict.py

ENTRYPOINT ["/predict.py"]
