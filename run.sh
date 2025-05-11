#!/usr/bin/env bash
set -e

# 1) Execute the full notebook (this only runs the first 3 cells in your current baseline.ipynb)
papermill baseline.ipynb output.ipynb

# 2) Now run your Python prediction+dump logic
python - << 'PYCODE'
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# load_dataset as in your code
def load_dataset(resp, lab):
    import json
    d = {}
    with open(resp) as f:
        for line in f: d[json.loads(line)["id"]] = json.loads(line)["response"]
    ids, texts, labels = [], [], []
    with open(lab) as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] in d:
                ids.append(obj["id"])
                texts.append(d[obj["id"]])
                labels.append(obj["label"])
    return ids, texts, labels

# paths inside container (spot-check mounts)
TRAIN_R = "data/responses-train.jsonl"
TRAIN_L = "data/responses-train-labels.jsonl"
TEST_R  = "data/responses-validation.jsonl"     # or your test-split for spot-check
TEST_L  = "data/responses-validation-labels.jsonl"

# Train on train+val
train_ids, train_texts, train_labels = load_dataset(TRAIN_R, TRAIN_L)
# (If you want to include val, load it similarly and concat)

pipeline = make_pipeline(
    TfidfVectorizer(stop_words="english", max_df=0.95),
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
)
pipeline.fit(train_texts, train_labels)

# Spot-check predictions
test_ids, test_texts, _ = load_dataset(TEST_R, TEST_L)
preds = pipeline.predict(test_texts)

# Write out exactly one JSONL file
with open("predictions.jsonl", "w") as out:
    for _id, p in zip(test_ids, preds):
        out.write(json.dumps({"id": _id, "label": int(p)}) + "\n")
PYCODE
