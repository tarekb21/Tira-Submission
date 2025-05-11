# Base image with Python, TIRA CLI, and other essentials
FROM webis/ir-lab-wise-2023:0.0.4

# Set working directory
WORKDIR /app

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all project files (including run.sh and your code)
COPY . .

# Make sure run.sh is executable and set it as the entrypoint
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
