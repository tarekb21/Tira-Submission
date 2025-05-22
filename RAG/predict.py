#!/usr/bin/env python3
from pathlib import Path
import click
import pickle
import numpy as np
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from itertools import chain
import torch

# Config
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
K_PER_LABEL = 5
FINAL_TOP_M = 4

@click.command()
@click.option('--dataset', required=True)
@click.option('--embed-model', default=EMBED_MODEL)
@click.option('--reranker-model', default=RERANKER_MODEL)
@click.option('--llm-model', default=LLM_MODEL)
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent))) / "predictions.jsonl")
def main(dataset, output, embed_model, reranker_model, llm_model):
    print("▶️ Loading TIRA dataset...")
    tira = Client()
    EMBED_MODEL = embed_model
    RERANKER_MODEL = reranker_model
    LLM_MODEL = llm_model
    df = tira.pd.inputs(dataset)

    print("▶️ Loading models and FAISS index...")
    base_path = Path(__file__).parent
    with open(base_path / "faiss_indices2.pkl", "rb") as f:
        faiss_indices, plain_docs = pickle.load(f)

    docs_by_topic_label = defaultdict(lambda: defaultdict(list))
    for topic, label_dict in plain_docs.items():
        for label, docs in label_dict.items():
            docs_by_topic_label[topic][label] = docs

    embedder = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    reranker = CrossEncoder(RERANKER_MODEL, local_files_only=True)
    llm_classifier = LocalLlamaClassifier(LLM_MODEL)

    print("▶️ Running RAG classification...")
    predictions = []
    for _, row in df.iterrows():
        pred, _ = classify_with_rag(
            row['response'], row['meta_topic'],
            embedder, faiss_indices, docs_by_topic_label,
            reranker, llm_classifier
        )
        predictions.append({
            "id": row['id'],
            "label": pred,
            "tag": "rag-tinyllama"
        })

    import json
    with open(output, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"✅ Predictions saved to {output}")

# Support functions (embed_query, rerank_pool, classify_with_rag, etc.) — same as your original RAG script
# Add below in same file, or import from a separate module if cleaner

def embed_query(text, embedder):
    return embedder.encode([text], normalize_embeddings=True)

def retrieve_by_label(query_emb, topic, faiss_indices, docs_by_topic_label, k=K_PER_LABEL):
    if topic in faiss_indices:
        pool = {}
        for label in (0, 1):
            idx = faiss_indices[topic].get(label)
            if idx is None:
                continue
            D, I = idx.search(query_emb, k)
            docs = docs_by_topic_label[topic].get(label, [])
            pool[label] = [docs[i] for i in I[0] if i != -1]
        return pool
    # Fallback
    all_docs = []
    for t, label_dict in docs_by_topic_label.items():
        for docs in label_dict.values():
            all_docs.extend(docs)
    emb_matrix = np.stack([d['embedding'] for d in all_docs])
    sims = (emb_matrix @ query_emb.T).squeeze()
    sorted_idxs = np.argsort(sims)[::-1]
    pool = {0: [], 1: []}
    for idx in sorted_idxs:
        doc = all_docs[idx]
        lab = doc['label']
        if len(pool[lab]) < k:
            pool[lab].append(doc)
        if len(pool[0]) >= k and len(pool[1]) >= k:
            break
    return pool

def rerank_pool(response, pool, reranker, top_m=FINAL_TOP_M):
    def rerank(candidates):
        pairs = [(response, d['response']) for d in candidates]
        scores = reranker.predict(pairs) if pairs else []
        idxs = np.argsort(scores)[-top_m:][::-1] if scores else []
        return [candidates[i] for i in idxs]
    no_ads = rerank(pool.get(0, []))
    ads = rerank(pool.get(1, []))
    return list(chain.from_iterable(zip(ads[:top_m//2], no_ads[:top_m//2])))

def make_prompt(response, context):
    p = "Examples:\n"
    for c in context:
        label = "AD" if c['label'] == 1 else "NO_AD"
        p += f'{{text: "{c["response"]}", label: {label}}}\n'
    p += f'\nResponse:\n"{response}"\nLabel (NO_AD or AD):'
    return p

def classify_with_rag(response, topic, embedder, faiss_indices, docs_by_topic_label, reranker, llm_classifier):
    system_prompt = (
        "You are a helpful assistant. Your goal is to classify the last text you received as an advertisement or not.\n"
        "An advertisement promotes a product, service, or event whereas non-advertisements only state objective information.\n"
        "You will be provided examples. Using your knowledge and these examples, output ONLY AD or NO_AD."
    )
    q_emb = embed_query(response, embedder)
    pool = retrieve_by_label(q_emb, topic, faiss_indices, docs_by_topic_label)
    if not any(pool.values()):
        return 0, []
    context = rerank_pool(response, pool, reranker)
    prompt = make_prompt(response, context)
    full_prompt = f"{system_prompt}\n\n{prompt}"
    label = llm_classifier.classify(full_prompt)
    return (1 if label.startswith("AD") else 0), context

class LocalLlamaClassifier:
    def __init__(self, model_path):
        print(f"🔄 Loading TinyLlama model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, only_local_files=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def classify(self, prompt):
        res = self.generator(prompt)[0]["generated_text"]
        after = res.split("Label (NO_AD or AD):")[-1].strip().replace("\n", "").replace("}", "").strip()
        if "AD" in after:
            if "NO_AD" in after and after.index("NO_AD") < after.index("AD"):
                return "NO_AD"
            return "AD"
        return "NO_AD"

if __name__ == "__main__":
    main()
