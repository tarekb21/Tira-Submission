FROM python:3

# Copy inference script and model
ADD script.py       /script.py
ADD requirements.txt /requirements.txt
ADD artifacts/model.pkl /model.pkl

# Install dependencies
RUN pip3 install -r /requirements.txt

# Run in inference mode
ENTRYPOINT ["sh", "-c", "python3 /script.py -i $inputDataset -o $outputDir"]
