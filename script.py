import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def load_dataset(responses_file, labels_file=None):
    responses = {}
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            responses[data["id"]] = data["response"]
    
    ids = list(responses.keys())
    texts = list(responses.values())
    
    labels = None
    if labels_file:
        labels = {}
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                labels[data["id"]] = data["label"]
        labels = [labels[i] for i in ids if i in labels]
    
    return ids, texts, labels

def main(input_dir, output_dir):
    # Define paths
    train_responses_file = os.path.join(input_dir, 'responses-train.jsonl')
    train_labels_file = os.path.join(input_dir, 'responses-train-labels.jsonl')
    val_responses_file   = os.path.join(input_dir, 'responses-validation.jsonl')
    val_labels_file      = os.path.join(input_dir, 'responses-validation-labels.jsonl')
    test_responses_file  = os.path.join(input_dir, 'ads-in-rag-task-2-classification-spot-check.jsonl')

    # Load data
    train_ids, train_texts, train_labels = load_dataset(train_responses_file, train_labels_file)
    val_ids, val_texts, val_labels = load_dataset(val_responses_file, val_labels_file)
    test_ids, test_texts, _ = load_dataset(test_responses_file)

    # Combine train and val
    combined_texts = train_texts + val_texts
    combined_labels = train_labels + val_labels

    # Train model
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english', max_df=0.95),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    )
    pipeline.fit(combined_texts, combined_labels)

    # Predict
    predictions = pipeline.predict(test_texts)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'predictions.jsonl'), 'w', encoding='utf-8') as f_out:
        for instance_id, pred in zip(test_ids, predictions):
            result = {
                "id": instance_id,
                "label": int(pred),
                "tag": "Tf-IDF-logReg"
            }
            f_out.write(json.dumps(result) + '\n')

    print(f"Done. Predictions written to {output_dir}/predictions.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input dataset.")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory.")
    args = parser.parse_args()
    main(args.input, args.output)