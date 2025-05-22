FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN apt-get update \
	&& apt-get install -y python3-pip \
	&& pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

RUN pip3 install --no-cache-dir faiss-cpu --break-system-packages

# Copy your script only
ADD RAG.py /

ENTRYPOINT ["/RAG.py"]
