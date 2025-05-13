FROM python:3

ADD script.py /script.py
ADD requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

ENTRYPOINT ["sh", "-c", "python3 /script.py -i $inputDataset -o $outputDir"]

