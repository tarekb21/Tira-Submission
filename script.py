# import argparse
# import os
# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline

# def load_dataset(responses_file, labels_file=None):
#     responses = {}
#     with open(responses_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             responses[data["id"]] = data["response"]
    
#     ids = list(responses.keys())
#     texts = list(responses.values())
    
#     labels = None
#     if labels_file:
#         labels = {}
#         with open(labels_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 data = json.loads(line)
#                 labels[data["id"]] = data["label"]
#         labels = [labels[i] for i in ids if i in labels]
    
#     return ids, texts, labels

# def main(input_dir, output_dir):
#     # Define paths
#     train_responses_file = os.path.join(input_dir, 'responses-train.jsonl')
#     train_labels_file = os.path.join(input_dir, 'responses-train-labels.jsonl')
#     val_responses_file   = os.path.join(input_dir, 'responses-validation.jsonl')
#     val_labels_file      = os.path.join(input_dir, 'responses-validation-labels.jsonl')
#     test_responses_file = os.path.join(input_dir, 'input.jsonl')

#     # Load data
#     train_ids, train_texts, train_labels = load_dataset(train_responses_file, train_labels_file)
#     val_ids, val_texts, val_labels = load_dataset(val_responses_file, val_labels_file)
#     test_ids, test_texts, _ = load_dataset(test_responses_file)

#     # Combine train and val
#     combined_texts = train_texts + val_texts
#     combined_labels = train_labels + val_labels

#     # Train model
#     pipeline = make_pipeline(
#         TfidfVectorizer(stop_words='english', max_df=0.95),
#         LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
#     )
#     pipeline.fit(combined_texts, combined_labels)

#     # Predict
#     predictions = pipeline.predict(test_texts)

#     # Write output
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, 'predictions.jsonl'), 'w', encoding='utf-8') as f_out:
#         for instance_id, pred in zip(test_ids, predictions):
#             result = {
#                 "id": instance_id,
#                 "label": int(pred),
#                 "tag": "Tf-IDF-logReg"
#             }
#             f_out.write(json.dumps(result) + '\n')

#     print(f"Done. Predictions written to {output_dir}/predictions.jsonl")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", required=True, help="Path to input dataset.")
#     parser.add_argument("-o", "--output", required=True, help="Path to output directory.")
#     args = parser.parse_args()
#     main(args.input, args.output)

# -------------------------------------------------------------------------------
# import argparse
# import os
# import json
# import joblib

# def load_dataset(responses_file):
#     """Load TIRA test input JSONL (id + response)."""
#     ids, texts = [], []
#     with open(responses_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             d = json.loads(line)
#             ids.append(d["id"])
#             texts.append(d["response"])
#     return ids, texts

# def find_test_file(input_dir):
#     # Walk the entire input_dir to find input.jsonl
#     for root, dirs, files in os.walk(input_dir):
#         if 'input.jsonl' in files:
#             return os.path.join(root, 'input.jsonl')
#     raise FileNotFoundError(f"No input.jsonl found under {input_dir}")

# def main(input_dir, output_dir):
# # 1) Locate TIRA’s test file anywhere under input_dir
#     test_path = find_test_file(input_dir)
#     print(f"DEBUG: Using test file at {test_path}", flush=True)

#     test_ids, test_texts = load_dataset(test_path)

#     # 2) Load model, predict, write output as before…
#     model = joblib.load("/model.pkl")
#     preds = model.predict(test_texts)

#     os.makedirs(output_dir, exist_ok=True)
#     out_file = os.path.join(output_dir, 'predictions.jsonl')
#     with open(out_file, 'w', encoding='utf-8') as fout:
#         for _id, p in zip(test_ids, preds):
#             fout.write(json.dumps({"id": _id, "label": int(p), "tag": "Tf-IDF-logReg"}) + "\n")

#     print(f"✅ Wrote {out_file}", flush=True)
#     # # 1) Locate TIRA’s test file
#     # test_path = os.path.join(input_dir, 'input.jsonl')
#     # if not os.path.exists(test_path):
#     #     raise FileNotFoundError(f"No input.jsonl in {input_dir}")

#     # test_ids, test_texts = load_dataset(test_path)

#     # # 2) Load pre-trained model
#     # model = joblib.load("/model.pkl")

#     # # 3) Predict
#     # preds = model.predict(test_texts)

#     # # 4) Write TIRA’s required output
#     # os.makedirs(output_dir, exist_ok=True)
#     # out_file = os.path.join(output_dir, 'predictions.jsonl')
#     # with open(out_file, 'w', encoding='utf-8') as fout:
#     #     for _id, p in zip(test_ids, preds):
#     #         fout.write(json.dumps({
#     #             "id": _id,
#     #             "label": int(p),
#     #             "tag": "Tf-IDF-logReg"
#     #         }) + "\n")

#     # print(f"✅ Wrote {out_file}", flush=True)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input",  required=True, help="Path to input dataset.")
#     parser.add_argument("-o", "--output", required=True, help="Path to output directory.")
#     args = parser.parse_args()
#     main(args.input, args.output)

#-----------------------------------------------------------------------

#!/usr/bin/env python3
import argparse
import os
import json
import joblib

def load_dataset(responses_file):
    ids, texts = [], []
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["id"])
            texts.append(d["response"])
    return ids, texts

def find_test_file(input_dir):
    # Walk the entire input_dir to find input.jsonl
    for root, dirs, files in os.walk(input_dir):
        if 'input.jsonl' in files:
            return os.path.join(root, 'input.jsonl')
    raise FileNotFoundError(f"No input.jsonl found under {input_dir}")

def main(input_dir, output_dir):
    # 1) Locate test file
    test_path = find_test_file(input_dir)
    print(f"DEBUG: Using test file at {test_path}", flush=True)

    # 2) Load data
    test_ids, test_texts = load_dataset(test_path)

    # 3) Load model & predict
    model = joblib.load("model.pkl")
    preds = model.predict(test_texts)

    # 4) Write predictions
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'predictions.jsonl')
    with open(out_file, 'w', encoding='utf-8') as fout:
        for _id, p in zip(test_ids, preds):
            fout.write(json.dumps({
                "id":    _id,
                "label": int(p),
                "tag":   "Tf-IDF-logReg"
            }) + "\n")

    print(f"✅ Wrote {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  required=True, help="Path to input dataset.")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory.")
    args = parser.parse_args()
    main(args.input, args.output)
