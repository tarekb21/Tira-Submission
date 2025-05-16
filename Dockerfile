# FROM python:3

# # Copy inference script and model
# ADD script.py       /script.py
# ADD requirements.txt /requirements.txt
# ADD artifacts/model.pkl /model.pkl

# # Install dependencies
# RUN pip3 install -r /requirements.txt

# # Run in inference mode
# ENTRYPOINT ["sh", "-c", "python3 /script.py -i $inputDataset -o $outputDir"]

FROM python:3.10-slim

# Copy inference assets
COPY script.py       /app/script.py
COPY artifacts/model.pkl /app/model.pkl
COPY requirements.txt /app/requirements.txt
COPY entrypoint.sh    /app/entrypoint.sh

WORKDIR /app

# Install only inference deps
RUN pip install --no-cache-dir -r requirements.txt \
 && chmod +x entrypoint.sh

# Use our tiny wrapper so that env vars expand properly
ENTRYPOINT ["./entrypoint.sh"]
