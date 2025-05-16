FROM python:3.10-slim

# copy in code + model
COPY script.py       /app/script.py
COPY artifacts/model.pkl /app/model.pkl
COPY requirements.txt /app/requirements.txt

WORKDIR /app

# install runtime deps
RUN pip install --no-cache-dir -r requirements.txt \
 && chmod +x script.py

# now the container WILL run your script by default
ENTRYPOINT ["python3", "script.py", "-i",  "$inputDataset", "-o", "$outputDir"]
