#!/usr/bin/env python3
from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":
    tira = Client()
    text = tira.pd.inputs("native-ads-2024-train")  # Or replace with your training dataset ID if different
    text = text.set_index("id")

    labels = tira.pd.truths("native-ads-2024-train")
    df = text.join(labels.set_index("id"))

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english", max_df=0.95)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    model.fit(df["response"], df["label"])
    dump(model, Path(__file__).parent / "model.pkl")
    print("✅ Model trained and saved as model.pkl")
