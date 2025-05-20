#!/usr/bin/env python3
from pathlib import Path
import json
import torch
import click
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sentence_transformers import SentenceTransformer
from torch import nn

class FineTuningWrapper(nn.Module):
    def __init__(self, sentence_model):
        super().__init__()
        self.sentence_model = sentence_model
        self.classifier = nn.Linear(sentence_model.get_sentence_embedding_dimension(), 1)

    def forward(self, features):
        embedding = self.sentence_model(features)['sentence_embedding']
        logits = self.classifier(embedding).view(-1)
        return logits

@click.command()
@click.option(
    '--dataset',
    default='advertisement-in-retrieval-augmented-generation-2025/ads-in-rag-task-2-classification-spot-check',
    help='Dataset'
)
@click.option(
    '--output',
    default=str(Path(get_output_directory(__file__)) / "predictions.jsonl"),
    help='Output file'
)
@click.option(
    '--model-dir',
    default=Path(__file__).parent / "models" / "All-Mini-LM-v2-FineTuned.pt",
    help='The model'
)
def main(dataset, output, model_dir):
    print(f"Running inference on dataset: {dataset}")
    output_path = Path(output)

    # Load dataset
    tira = Client()
    df = tira.pd.inputs(dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model = FineTuningWrapper(sentence_model).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(str(model_dir), map_location=device))
    model.eval()

    tokenizer = sentence_model.tokenizer
    max_len = sentence_model.get_max_seq_length()

    # Predict
    predictions = []
    with torch.no_grad():
        for row in df.itertuples():
            inputs = tokenizer(row.response, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logit = model(inputs)
            prob = torch.sigmoid(logit).item()
            label = 1 if prob >= 0.5 else 0
            predictions.append({
                "id": row.id,
                "label": label,
                "tag": "mpnet-finetune"
            })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
