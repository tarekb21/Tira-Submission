# train_model.py

import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model        import LogisticRegression
from sklearn.pipeline            import make_pipeline

def load_dataset(responses_file, labels_file):
    """Return (texts, labels) for those IDs present in both files."""
    # read all responses into a dict
    responses = {}
    with open(responses_file, 'r', encoding='utf-8') as rf:
        for line in rf:
            d = json.loads(line)
            responses[d["id"]] = d["response"]
    # now collect texts and labels
    texts, labels = [], []
    with open(labels_file, 'r', encoding='utf-8') as lf:
        for line in lf:
            d = json.loads(line)
            if d["id"] in responses:
                texts.append(responses[d["id"]])
                labels.append(d["label"])
    return texts, labels

if __name__ == "__main__":
    # adjust paths if needed
    train_texts, train_labels = load_dataset(
        "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-train.jsonl",
        "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-train-labels.jsonl"
    )
    val_texts, val_labels     = load_dataset(
        "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-validation.jsonl",
        "/Users/tarekbouhairi/Desktop/my_submission/Dataset/responses-validation-labels.jsonl"
    )

    # combine splits
    X = train_texts + val_texts
    y = train_labels + val_labels

    # train your pipeline
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", max_df=0.95),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    )
    pipeline.fit(X, y)

    # save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"✅ Trained model saved to {model_path}")
