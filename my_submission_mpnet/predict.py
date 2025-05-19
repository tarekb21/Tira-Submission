#!/usr/bin/env python3
from pathlib import Path
import torch
import json
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from click import command, option
import os

class FineTuningWrapper(torch.nn.Module):
    def __init__(self, sentence_model):
        super().__init__()
        self.sentence_model = sentence_model
        self.classifier = torch.nn.Linear(sentence_model.get_sentence_embedding_dimension(), 1)

    def forward(self, features):
        embedding = self.sentence_model(features)['sentence_embedding']
        logits = self.classifier(embedding).view(-1)
        return logits

@command()
@option('--dataset', default='advertisement-in-retrieval-augmented-generation-2025/ads-in-rag-task-2-classification-spot-check', help='Dataset')
@option('--output', default=str(Path(get_output_directory(str(Path(__file__).parent))) / "predictions.jsonl"), help='Output file')
def main(dataset, output):
    print(f"Running inference on dataset: {dataset}")
    
    output = Path(output)  # 🧠 Fix: Convert string to Path

    # Enforce offline mode for Hugging Face
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Load input
    tira = Client()
    df = tira.pd.inputs(dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ✅ Load model from local directory inside container
    model_dir = Path(__file__).parent / 'all-mpnet-base-v2'
    sentence_model = SentenceTransformer(str(model_dir), trust_remote_code=True)
    tokenizer = sentence_model.tokenizer
    max_seq_length = sentence_model.get_max_seq_length()

    # ✅ Load fine-tuned head
    model = FineTuningWrapper(sentence_model).to(device)
    model.load_state_dict(torch.load(Path(__file__).parent / 'MPnet-full_end_to_end_model.pt', map_location=device))
    model.eval()

    # Run inference
    preds = []
    with torch.no_grad():
        for row in df.itertuples():
            inputs = tokenizer(row.response, return_tensors='pt', truncation=True, padding=True, max_length=max_seq_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logit = model(inputs)
            prob = torch.sigmoid(logit).item()
            label = 1 if prob >= 0.5 else 0
            preds.append({
                "id": row.id,
                "label": label,
                "tag": "mpnet-finetune"
            })

    # Save predictions
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        for pred in preds:
            f.write(json.dumps(pred) + '\n')

    print(f"✅ Predictions written to {output}")

if __name__ == "__main__":
    main()
