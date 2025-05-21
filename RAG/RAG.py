#!/usr/bin/env python3

import os
import json
import pickle
import numpy as np
import faiss
import click
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import chain
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import confusion_matrix, classification_report

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ──────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

K_PER_LABEL = 5
FINAL_TOP_M = 4

TINY_LLAMA_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # local path or HF hub

# ─── DATA HELPERS ─────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────
@click.command()
@click.option("--train-resp",      required=True, type=click.Path(exists=True), help="Training responses .jsonl")
@click.option("--train-labels",    required=True, type=click.Path(exists=True), help="Training labels .jsonl")
@click.option("--valid-resp",      required=True, type=click.Path(exists=True), help="Validation responses .jsonl")
@click.option("--valid-labels",    required=True, type=click.Path(exists=True), help="Validation labels .jsonl")
@click.option("--test-resp",       required=True, type=click.Path(exists=True), help="Test responses .jsonl")
@click.option("--test-labels",     required=True, type=click.Path(exists=True), help="Test labels .jsonl")
@click.option("--faiss-out",       default="faiss_indices2.pkl", help="Where to store faiss index")
@click.option("--output",          default="predictions.jsonl", help="Where to store predictions/results")
@click.option("--llm-path",        default=TINY_LLAMA_PATH, help="HF local path or model name for TinyLlama")
def main(
    train_resp, train_labels, valid_resp, valid_labels, test_resp, test_labels, faiss_out, output, llm_path
):
    print("Loading and merging training/validation data...")
    train_data = load_and_label(train_resp, train_labels)
    valid_data = load_and_label(valid_resp, valid_labels)
    train_data.extend(valid_data)

    print("Building label/topic index...")
    docs_by_topic_label = defaultdict(lambda: defaultdict(list))
    for doc in train_data:
        docs_by_topic_label[doc['meta_topic']][doc['label']].append(doc)

    print("Encoding embeddings for all responses...")
    embedder = SentenceTransformer(EMBED_MODEL)
    for topic, labels in docs_by_topic_label.items():
        for label, docs in labels.items():
            texts = [d['response'] for d in docs]
            embs  = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
            for doc, emb in zip(docs, embs):
                doc['embedding'] = emb

    print(f"Building and saving FAISS index to {faiss_out} ...")
    faiss_indices = {}
    for topic, labels in docs_by_topic_label.items():
        faiss_indices[topic] = {}
        for label, docs in labels.items():
            dim   = docs[0]['embedding'].shape[0]
            index = faiss.IndexFlatIP(dim)
            array = np.stack([d['embedding'] for d in docs])
            index.add(array)
            faiss_indices[topic][label] = index

    # Convert to nested dict for pickling
    plain_docs = { topic: { label: docs for label, docs in label_dict.items() }
        for topic, label_dict in docs_by_topic_label.items()
    }
    with open(faiss_out, "wb") as f:
        pickle.dump((faiss_indices, plain_docs), f)

    print("Loading TinyLlama model for local inference...")
    llm_classifier = LocalLlamaClassifier(llm_path)
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Loading test data...")
    test_data = []
    test_map = {l['id']: l['label'] for l in load_jsonl(test_labels)}
    for r in load_jsonl(test_resp):
        if r['id'] in test_map:
            test_data.append({
                'id':         r['id'],
                'meta_topic': r['meta_topic'],
                'response':   r['response'],
                'label':      test_map[r['id']]
            })

    # Load FAISS indices back for querying
    with open(faiss_out, "rb") as f:
        faiss_indices, plain_docs = pickle.load(f)
    docs_by_topic_label = defaultdict(lambda: defaultdict(list))
    for topic, label_dict in plain_docs.items():
        for label, docs in label_dict.items():
            docs_by_topic_label[topic][label] = docs

    # Main eval loop
    y_true, y_pred = [], []
    results = []
    print("Evaluating classifier on test set...")
    for ex in tqdm(test_data, desc="Classifying"):
        pred, ctx = classify_with_rag(
            ex['response'],
            ex['meta_topic'],
            embedder,
            faiss_indices,
            docs_by_topic_label,
            reranker,
            llm_classifier
        )
        y_true.append(ex['label'])
        y_pred.append(pred)
        results.append({
            'id': ex['id'],
            'label': ex['label'],
            'prediction': pred,
            'response': ex['response'],
            'meta_topic': ex['meta_topic']
        })

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=3))

    print(f"Saving predictions to {output}")
    save_jsonl(output, results)


# ─── HELPERS & CLASSES ──────────────────────────────────────────────────────────

def load_and_label(jsonl_path, labels_path):
    examples = load_jsonl(jsonl_path)
    labels   = load_jsonl(labels_path)
    lbl_map  = {l['id']: l['label'] for l in labels}
    merged   = []
    for r in examples:
        if r['id'] in lbl_map:
            merged.append({
                'id':         r['id'],
                'meta_topic': r['meta_topic'],
                'response':   r['response'],
                'label':      lbl_map[r['id']]
            })
    return merged

def embed_query(text, embedder):
    emb = embedder.encode([text], normalize_embeddings=True)
    return emb

def retrieve_by_label(query_emb, topic, faiss_indices, docs_by_topic_label, k=K_PER_LABEL):
    # Case 1: topic exists
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
    # Case 2: fallback: search globally
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
    candidates_no_ad = pool.get(0, [])
    pairs_no_ad      = [(response, d['response']) for d in candidates_no_ad]
    scores_no_ad     = reranker.predict(pairs_no_ad) if pairs_no_ad else []
    idxs_no_ad       = np.argsort(scores_no_ad)[-top_m:][::-1] if scores_no_ad != [] else []
    selected_no_ad   = [candidates_no_ad[i] for i in idxs_no_ad]

    candidates_ad = pool.get(1, [])
    pairs_ad      = [(response, d['response']) for d in candidates_ad]
    scores_ad     = reranker.predict(pairs_ad) if pairs_ad else []
    idxs_ad       = np.argsort(scores_ad)[-top_m:][::-1] if scores_ad != [] else []
    selected_ad   = [candidates_ad[i] for i in idxs_ad]
    out = list(chain.from_iterable(zip(selected_ad[:top_m//2], selected_no_ad[:top_m//2])))
    return out

def make_prompt(response, context):
    p = "Examples:\n"
    for c in context:
        label_example = "AD" if c['label']==1 else "NO_AD"
        p += f"{{text: \"{c['response']}\", label: {label_example}}}\n"
    p += f"\nResponse:\n\"{response}\"\nLabel (NO_AD or AD):"
    return p

def classify_with_rag(response, topic, embedder, faiss_indices, docs_by_topic_label, reranker, llm_classifier):
    q_emb = embed_query(response, embedder)
    pool  = retrieve_by_label(q_emb, topic, faiss_indices, docs_by_topic_label)
    if not any(pool.values()):
        return 0, []
    ctx   = rerank_pool(response, pool, reranker)
    prompt = make_prompt(response, ctx)
    label = llm_classifier.classify(prompt)
    return (1 if label.startswith("AD") else 0), ctx

# ─── LOCAL LLAMA CLASSIFIER ──────────────────────────────────────────────
class LocalLlamaClassifier:
    def __init__(self, model_path):
        print(f"Loading local LLM from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
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
        # Only look at final "Label (NO_AD or AD):" output!
        after_label = res.split("Label (NO_AD or AD):")[-1].strip()
        after_label = after_label.replace("\n", "").replace("}", "").strip()
        # Extract 'AD' or 'NO_AD'
        if "AD" in after_label:
            if "NO_AD" in after_label and after_label.index("NO_AD") < after_label.index("AD"):
                return "NO_AD"
            return "AD"
        return "NO_AD"

# ─── MAIN ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
