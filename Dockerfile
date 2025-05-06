FROM python:3.11-slim
WORKDIR /app

-# Copy & install dependencies
-COPY requirements.txt .
-RUN pip install --no-cache-dir -r requirements.txt
+## 1. Install system dependencies so pip can handle git-based packages
+RUN apt-get update \
+    && apt-get install -y --no-install-recommends \
+         git \
+         build-essential \
+    && rm -rf /var/lib/apt/lists/*
+
+# 2. Copy & install Python dependencies (including git+… packages)
+COPY requirements.txt .
+RUN pip install --no-cache-dir -r requirements.txt

# Copy in your code, the notebook, and our run.sh
COPY . .

# Install papermill, kernels, nbformat
RUN pip install --no-cache-dir papermill ipykernel nbformat

# Register a 'python3' kernel
RUN python -m ipykernel install --user --name python3 --display-name python3

# Make run.sh executable
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
