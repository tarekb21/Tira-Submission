import torch as T
from transformers import AutoModel

class FineTuningWrapper(T.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(f"sentence-transformers/{model_name}")
        hidden_size = self.encoder.config.hidden_size
        self.classifier = T.nn.Linear(hidden_size, 1)

    def forward(self, features):
        output = self.encoder(**features)
        embedding = self.mean_pooling(output, features['attention_mask'])
        logits = self.classifier(embedding).view(-1)
        return logits

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return T.sum(token_embeddings * input_mask_expanded, 1) / T.clamp(input_mask_expanded.sum(1), min=1e-9)
