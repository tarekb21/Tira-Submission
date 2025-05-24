#!/usr/bin/env python3
import json
import joblib
import click
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sentence_transformers import SentenceTransformer

@click.command()
@click.option(
    "--dataset",
    required=True,
    help="TIRA dataset name, e.g. advertisement-in-retrieval-augmented-generation-2025/ads-in-rag-task-2-classification-spot-check"
)
@click.option(
    "--output",
    default=None,
    help="Where to write predictions.jsonl"
)
def main(dataset, output):
    tira = Client()
    df = tira.pd.inputs(dataset)

    # load embedder + RF
    embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    clf      = joblib.load("models/rf_classifier.pkl")

    # determine output path
    if output is None:
        output = str(get_output_directory(__file__) / "predictions.jsonl")

    print("▶️ Running inference …")
    with open(output, "w") as fout:
        for row in df.itertuples():
            emb = embedder.encode(
                row.response,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).reshape(1, -1)
            pred = int(clf.predict(emb)[0])
            fout.write(json.dumps({"id": row.id, "label": pred, "tag":"rf-allmini"}) + "\n")

    print(f"✅ Predictions saved to {output}")

if __name__ == "__main__":
    main()
