#!/usr/bin/env bash
set -e

# 1) Execute the notebook
papermill baseline.ipynb output.ipynb

# 2) Extract any JSON objects that your notebook printed
#    and write them into predictions.jsonl (TIRA’s expected format).
python - << 'PYCODE'
import nbformat, json

nb = nbformat.read("output.ipynb", as_version=4)
records = []
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        # we assume your notebook does something like `print(json.dumps(...))`
        if out.output_type == "stream" and out.name == "stdout":
            for line in out.text.splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

# write exactly one .jsonl file
with open("predictions.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
PYCODE
