from pathlib import Path
import torch as T
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from models.models import FineTuningWrapper

MODEL_PATH = Path("/models")
DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")

class MPNetModel:
    def __init__(self, model_name, input_run):
        self.model_name = model_name
        self.input_run = input_run
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
        self.max_len = 256

        # Load fine-tuned classifier on top
        self.model = FineTuningWrapper(model_name).to(self.device)
        self.model.load_state_dict(T.load(MODEL_PATH / "MPnet-full_end_to_end_model.pt", map_location=self.device))
        self.model.eval()

    def make_predictions(self):
        results = []

        with T.no_grad():
            for row in self.input_run.itertuples():
                inputs = self.tokenizer(row.response, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(inputs)
                prob = T.sigmoid(logits).item()
                label = 1 if prob >= 0.5 else 0
                results.append({"id": row.id, "label": label})

        return pd.DataFrame(results)
