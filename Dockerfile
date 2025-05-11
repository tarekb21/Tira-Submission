# 1) Base image with Python, Java, TIRA CLI, etc.
FROM webis/ir-lab-wise-2023:0.0.4

# 2) Copy & install your minimal Python deps
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# 3) Put code + model into /app
WORKDIR /app
COPY train.py predict.py model.joblib /app/

# 4) By default, run the predictor
ENTRYPOINT ["/app/predict.py"]
