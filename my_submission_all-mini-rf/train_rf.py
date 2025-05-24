#!/usr/bin/env python3
import json
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

# 1) load train + val
train = load_jsonl("/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-train.jsonl")
labels = {d["id"]: d["label"]
          for d in load_jsonl("/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-train-labels.jsonl")}

texts = [r["response"] for r in train if r["id"] in labels]
y     = [labels[r["id"]]   for r in train if r["id"] in labels]

# 2) encode
print("▶️ Encoding with all-MiniLM-L6-v2 …")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(texts,
                     batch_size=32,
                     show_progress_bar=True,
                     normalize_embeddings=True)

# 3) train RF
print("▶️ Training RandomForest …")
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X, y)

# 4) dump
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/rf_classifier.pkl")
print("✅ Saved RF to models/rf_classifier.pkl")
