# Use a slim Python base
FROM python:3.11-slim

# 1. Install system deps so pip can fetch git-based packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         git \
         build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy in your code, notebook, and run.sh
COPY . .

# 5. Install papermill, kernel support, and nbformat
RUN pip install --no-cache-dir papermill ipykernel nbformat

# 6. Register a 'python3' kernel for papermill
RUN python -m ipykernel install --user --name python3 --display-name python3

# 7. Make your run.sh executable and use it as entrypoint
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
