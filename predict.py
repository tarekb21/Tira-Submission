#!/usr/bin/env python3
from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import click
import json

@click.command()
@click.option(
    '--dataset',
    default='advertisement-in-retrieval-augmented-generation-2025/ads-in-rag-task-2-classification-spot-check',
    help='The dataset to run predictions on.'
)
@click.option(
    '--output',
    default=Path(get_output_directory(str(Path(__file__).parent))) / "predictions.jsonl",
    help='The file where predictions should be written to.'
)
def main(dataset, output):
    tira = Client()
    df = tira.pd.inputs(dataset)

    model = load(Path(__file__).parent / "model.pkl")
    predictions = model.predict(df["response"])

    df["label"] = predictions
    df["tag"] = "Tf-IDF-logReg"

    df[["id", "label", "tag"]].to_json(output, orient="records", lines=True)
    print(f"✅ Wrote predictions to {output}", flush=True)

if __name__ == "__main__":
    main()
