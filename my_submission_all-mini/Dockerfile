FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN apt-get update \
	&& apt-get install -y python3-pip \
	&& pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Copy your script only
ADD predict.py /predict.py

# Make entrypoint executable
RUN chmod +x /predict.py


# Set entrypoint for TIRA
ENV PYTHONPATH=/
#ENTRYPOINT ["/predict.py", "--model-dir", "/root/.cache/huggingface/hub/models--tarekb21--MPnet-finetune/snapshots/ebb614d79faf8cbfb4f1c1649adfbadf40f81261/MPnet-full_end_to_end_model.pt"]
