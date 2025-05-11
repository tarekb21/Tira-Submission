#!/usr/bin/env bash
set -e

# 1) Execute the notebook (this will run all cells, or you can parameterize it)
papermill baseline.ipynb output.ipynb

# 2) Train & predict on the training split, then dump JSONL
python - << 'PYCODE'
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# helper to load a JSONL split
def load_dataset(responses_path, labels_path):
    resp = { }
    with open(responses_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            resp[obj['id']] = obj['response']
    ids, texts, labels = [], [], []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj['id'] in resp:
                ids.append(obj['id'])
                texts.append(resp[obj['id']])
                labels.append(obj['label'])
    return ids, texts, labels

# TIRA mounts your chosen dataset under /app/data
TRAIN_R = "my_submission/Dataset/responses-train.jsonl"
TRAIN_L = "my_submission/Dataset/responses-train-labels.jsonl"

# load & train
train_ids, train_texts, train_labels = load_dataset(TRAIN_R, TRAIN_L)
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_df=0.95),
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
)
pipeline.fit(train_texts, train_labels)

# spot-check on the same training IDs (or you could split out a val subset)
preds = pipeline.predict(train_texts)

# write out exactly one JSONL file for TIRA’s spot-check
with open("predictions.jsonl", "w", encoding="utf-8") as out:
    for _id, p in zip(train_ids, preds):
        json.dump({"id": _id, "label": int(p)}, out)
        out.write("\n")
PYCODE