# Use a slim Python image
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in your code, the notebook, and our run.sh
COPY . .

# Install papermill, kernels, nbformat
RUN pip install --no-cache-dir papermill ipykernel nbformat

# Register a 'python3' kernel so papermill can find it
RUN python -m ipykernel install --user --name python3 --display-name python3

# Make run.sh executable and use it as the container entrypoint
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
