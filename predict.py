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
    print(f"DEBUG: Dataset = {dataset}", flush=True)

    # Load input
    tira = Client()
    df = tira.pd.inputs(dataset)
    print(f"DEBUG: Loaded {len(df)} rows from dataset", flush=True)

    if df.empty:
        print("❗ ERROR: No input data to predict on.", flush=True)
        return

    # Load model
    model_path = Path(__file__).parent / "model.pkl"
    print(f"DEBUG: Loading model from {model_path}", flush=True)

    if not model_path.exists():
        print("❗ ERROR: model.pkl not found!", flush=True)
        return

    model = load(model_path)

    # Predict
    predictions = model.predict(df["response"])
    print(f"DEBUG: Made predictions for {len(predictions)} rows", flush=True)

    # Write output
    df["label"] = predictions
    df["tag"] = "Tf-IDF-logReg"

    df[["id", "label", "tag"]].to_json(output, orient="records", lines=True)
    print(f"✅ Wrote predictions to {output}", flush=True)

if __name__ == "__main__":
    main()
