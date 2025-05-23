#!/usr/bin/env python3
import json
import pickle
import numpy as np
import faiss
import click
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import confusion_matrix, classification_report
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ──────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TINY_LLAMA_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

K_PER_LABEL = 5
FINAL_TOP_M = 4

# ─── JSONL HELPERS ───────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ─── MAIN ────────────────────────────────────────────────────────────────
@click.command()
@click.option("--test-resp",       required=True, type=click.Path(exists=True), help="Test responses .jsonl")
@click.option("--test-labels",     required=True, type=click.Path(exists=True), help="Test labels .jsonl")
@click.option("--faiss-pkl",       default="faiss_indices2.pkl", help="Path to FAISS index + doc embeddings")
@click.option("--output",          default="predictions.jsonl", help="Where to store predictions")
@click.option("--llm-path",        default=TINY_LLAMA_PATH, help="TinyLlama model path or HF hub name")
def main(test_resp, test_labels, faiss_pkl, output, llm_path):
    print("▶️ Loading FAISS index and docs...")
    with open(faiss_pkl, "rb") as f:
        docs_by_topic_label = pickle.load(f)  # only docs, not indices

    faiss_indices = {}
    for topic, label_dict in docs_by_topic_label.items():
        faiss_indices[topic] = {}
        for label, docs in label_dict.items():
            if not docs:
                continue
            embeddings = np.stack([doc['embedding'] for doc in docs]).astype(np.float32)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss_indices[topic][label] = index


    print("▶️ Loading embedder, reranker, and TinyLlama...")
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    llm_classifier = LocalLlamaClassifier(llm_path)

    print("▶️ Loading test set...")
    test_map = {r['id']: r['label'] for r in load_jsonl(test_labels)}
    test_data = [
        {**r, 'label': test_map[r['id']]}
        for r in load_jsonl(test_resp) if r['id'] in test_map
    ]

    print("▶️ Running inference...")
    y_true, y_pred, results = [], [], []
    for ex in tqdm(test_data, desc="Classifying"):
        pred, ctx = classify_with_rag(
            ex['response'], ex['meta_topic'], embedder,
            faiss_indices, docs_by_topic_label,
            reranker, llm_classifier
        )
        y_true.append(ex['label'])
        y_pred.append(pred)
        results.append({**ex, "prediction": pred})

    print("✔️ Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("✔️ Classification Report:\n", classification_report(y_true, y_pred, digits=3))
    save_jsonl(output, results)
    print(f"✅ Saved predictions to {output}")

# ─── RAG CLASSIFICATION ──────────────────────────────────────────────────
def embed_query(text, embedder):
    return embedder.encode([text], normalize_embeddings=True).astype(np.float32)

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

    # Fallback to global search
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
    ads    = rerank(pool.get(1, []))
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
        "An advertisement promotes a product, service, or event whereas non-advertisements only state objective information "
        "about the product, service, or event. Be sure to correctly capture the nuances between advertisements and informative texts.\n"
        "You will be provided examples. Using your knowledge and these examples if they are relevant, output ONLY AD if the last "
        "Response is an advertisement or output NO_AD if it is not an advertisement."
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

# ─── LOCAL TINY LLAMA ────────────────────────────────────────────────────
class LocalLlamaClassifier:
    def __init__(self, model_path):
        print(f"🔄 Loading TinyLlama model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
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
