FROM python:3.10-slim

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Preload HuggingFace model into Docker
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2'); AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')"

# Copy code and model
ADD predict.py /predict.py
ADD models /models/

COPY models/all-mpnet-base-v2 /models/all-mpnet-base-v2
COPY MPnet-full_end_to_end_model.pt /MPnet-full_end_to_end_model.pt

# ✅ Make entrypoint executable
RUN chmod +x /predict.py

# Set entrypoint for TIRA
ENV PYTHONPATH=/
ENTRYPOINT ["/predict.py"]
