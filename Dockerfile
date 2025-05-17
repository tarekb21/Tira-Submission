FROM python:3.10-slim

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy prediction script and model
COPY predict.py /predict.py
COPY model.pkl /model.pkl

ENTRYPOINT ["/predict.py"]
