# A prepared image with Python 3.10, Java 11, ir_datasets, TIRA, and PyTerrier pre-installed
FROM webis/ir-lab-wise-2023:0.0.4

# Update TIRA CLI to the latest version
RUN pip3 uninstall -y tira \
    && pip3 install --no-cache-dir tira

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install notebook execution and kernel support
RUN pip3 install --no-cache-dir papermill ipykernel nbformat

# Register a 'python3' Jupyter kernel for papermill
RUN python3 -m ipykernel install --user --name python3 --display-name python3

# Ensure run.sh is executable and set it as entrypoint
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
