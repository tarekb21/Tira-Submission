#!/usr/bin/env python3
import os
import json
import click
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# optional HF pipeline
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

# OpenAI client
from openai import OpenAI

# few-shot examples
few_shot_examples = [
    {"text": "To open an online account with Union Bank, you can visit ...", "label": 1},
    {"text": "If you are looking for a BMO Harris Bank branch near you ...", "label": 0},
    {"text": "Electric Cars in 2022\nIn 2022, several electric cars ...", "label": 1},
    {"text": "Best Hybrid Cars\nWhen it comes to the best hybrid cars ...", "label": 0},
]

def create_prompt(text):
    prompt = (
        "Determine if the following text is an advertisement (respond with exactly '1') "
        "or not (respond with exactly '0').\n"
        "An advertisement promotes a specific product/service with persuasive language.\n\nExamples:"
    )
    for ex in few_shot_examples:
        prompt += f"\n\nText: {ex['text']}\nLabel: {ex['label']}"
    prompt += f"\n\nText: {text}\nLabel: (Respond ONLY with 1 or 0, nothing else)"
    return prompt

def query_openai(client, model, text):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":create_prompt(text)}],
        temperature=0.1,
        max_tokens=1
    )
    return resp.choices[0].message.content.strip()

def query_local(pipe, text):
    out = pipe(create_prompt(text), max_new_tokens=5, do_sample=False)
    # last token is our label
    return out[0]["generated_text"].strip().split()[-1]

def load_data(resp_file, label_file):
    resps = [json.loads(l) for l in open(resp_file, 'r', encoding='utf8')]
    labels = {j["id"]: j["label"] for j in [json.loads(l) for l in open(label_file,'r',encoding='utf8')]}
    return [
        {"id": r["id"], "text": r["response"], "true": labels[r["id"]]}
        for r in resps if r["id"] in labels
    ]

@click.command()
@click.option("--response-file", required=True, help="responses-test.jsonl")
@click.option("--label-file",    required=True, help="responses-test-labels.jsonl")
@click.option("--model",         required=True, help="local path or OpenAI model name")
@click.option("--api-key",       default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
def main(response_file, label_file, model, api_key):
    data = load_data(response_file, label_file)

    # choose local vs OpenAI
    use_local = hf_pipeline and os.path.isdir(model)
    if use_local:
        tok = AutoTokenizer.from_pretrained(model)
        lm  = AutoModelForCausalLM.from_pretrained(model)
        pipe = hf_pipeline("text-generation", model=lm, tokenizer=tok, trust_remote_code=True)
        client = None
    else:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Must provide --api-key or set OPENAI_API_KEY")
        client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL","https://api.openai.com"))
        pipe = None

    true, pred = [], []
    for item in data:
        if use_local:
            out = query_local(pipe, item["text"])
        else:
            out = query_openai(client, model, item["text"])
        lab = int(out) if out in ("0","1") else 0
        true.append(item["true"]); pred.append(lab)
        print(f"ID={item['id']} True={item['true']} Pred={lab}")

    print("\n" + classification_report(true, pred, target_names=["Not Ad","Ad"]))
    print(confusion_matrix(true, pred))

if __name__=="__main__":
    main()
