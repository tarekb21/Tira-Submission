import os
import json
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def load_dataset(responses_file, labels_file):
    responses = {}
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            responses[d["id"]] = d["response"]

    texts, labels = [], []
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in responses:
                texts.append(responses[d["id"]])
                labels.append(d["label"])

    return texts, labels

if __name__ == "__main__":
    # ✅ Update these paths to point to your local training files
    train_responses = "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-train.jsonl"
    train_labels    = "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-train-labels.jsonl"
    val_responses   = "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-validation.jsonl"
    val_labels      = "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-validation-labels.jsonl"

    # Load and combine train + validation data
    train_texts, train_labels = load_dataset(train_responses, train_labels)
    val_texts, val_labels     = load_dataset(val_responses, val_labels)

    X = train_texts + val_texts
    y = train_labels + val_labels

    # Train TF-IDF + Random Forest
    model = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english", max_df=0.95)),
        ("classifier", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    model.fit(X, y)
    dump(model, "model.pkl")
    print("✅ Trained and saved model.pkl")
