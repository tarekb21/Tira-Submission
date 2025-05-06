# Use a slim Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in all your code & notebooks
COPY . .

# Install papermill so we can execute the notebook
RUN pip install --no-cache-dir papermill

# When the container runs, execute baseline.ipynb → output.ipynb
ENTRYPOINT ["papermill", "baseline.ipynb", "output.ipynb"]
