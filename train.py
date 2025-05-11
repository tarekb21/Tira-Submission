#!/usr/bin/env python3
from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":
    # 1) load the TRAINING split
    TRAIN_DS = "ads-in-rag-task-2-classification-training"
    tira = Client()
    df_resp = tira.pd.inputs(TRAIN_DS)
    df_lab  = tira.pd.truths(TRAIN_DS)
    df = df_resp.set_index("id").join(df_lab.set_index("id"))

    # 2) fit TF-IDF → LogisticRegression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.95)),
        ("clf",   LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    model.fit(df["response"], df["label"])

    # 3) serialize the model
    dump(model, Path(__file__).parent / "model.joblib")
    print("Model trained and saved to model.joblib")
