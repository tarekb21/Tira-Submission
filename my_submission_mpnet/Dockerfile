FROM python:3.10-slim

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your script only
ADD predict.py /predict.py

# Make entrypoint executable
RUN chmod +x /predict.py


# Set entrypoint for TIRA
ENV PYTHONPATH=/
ENTRYPOINT ["/predict.py"]
