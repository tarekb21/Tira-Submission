#!/usr/bin/env python3
from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import click

@click.command()
@click.option(
    "--dataset",
    default="ads-in-rag-task-2-classification-test",
    help="Which TIRA dataset to run on (spot-check or test)."
)
@click.option(
    "--output",
    default=Path(get_output_directory(__file__).parent) / "predictions.jsonl",
    help="Where to write the JSONL predictions."
)
def main(dataset, output):
    # 1) Fetch the responses for this split
    tira = Client()
    df = tira.pd.inputs(dataset)

    # 2) Load your trained model
    model = load(Path(__file__).parent / "model.joblib")

    # 3) Predict labels (0/1) and attach your tag
    df["label"] = model.predict(df["response"])
    df["tag"]   = "Tf-IDF-LogReg"

    # 4) Dump exactly one JSONL of {id, label, tag}
    df[["id", "label", "tag"]] \
      .to_json(output, orient="records", lines=True)
    print(f"Wrote predictions to {output}")

if __name__ == "__main__":
    main()
