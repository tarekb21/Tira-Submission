#!/usr/bin/env python3
import os
import json
import click
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Optional HF pipeline for local models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

# OpenAI client for remote models
from openai import OpenAI, OpenAIError

# Few-shot examples (inline)
few_shot_examples = [
    {"text": "To open an online account with Union Bank, you can visit their official website and follow the account opening process...", "label": 1},
    {"text": "If you are looking for a BMO Harris Bank branch near you, you can use the BMO branch locator tool on their website...", "label": 0},
    {"text": "Electric Cars in 2022\nIn 2022, several electric cars have gained attention for their performance, features, and environmental benefits...", "label": 1},
    {"text": "Best Hybrid Cars\nWhen it comes to the best hybrid cars, there are several top-rated options to consider...", "label": 0}
]


def create_prompt(text):
    prompt = (
        "Determine if the following text is an advertisement (respond with exactly '1') or not (respond with exactly '0')."
        "\nAn advertisement promotes a specific product/service with persuasive language.\n\nExamples:"
    )
    for ex in few_shot_examples:
        prompt += f"\n\nText: {ex['text']}\nLabel: {ex['label']}"
    prompt += f"\n\nText: {text}\nLabel: (Respond ONLY with 1 or 0, nothing else)"
    return prompt


def query_openai(client, model_name, base_url, text):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": create_prompt(text)}],
            temperature=0.1,
            max_tokens=1
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        print(f"Warning: OpenAI API error: {e}")
        return None


def query_local(pipe, text):
    try:
        out = pipe(create_prompt(text), max_new_tokens=5, do_sample=False)
        return out[0]["generated_text"].strip().split()[-1]
    except Exception as e:
        print(f"Warning: Local model error: {e}")
        return None


def load_inputs(dataset_path):
    if os.path.isdir(dataset_path):
        file_path = os.path.join(dataset_path, "responses-test.jsonl")
    else:
        file_path = dataset_path
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            items.append({"id": obj.get("id"), "text": obj.get("response")})
    return items

@click.command()
@click.option("--dataset",  required=True, help="Path to dataset dir or JSONL file.")
@click.option("--output",   required=True, help="Path to write predictions.jsonl.")
@click.option("--model",    required=True, help="Local model path or OpenAI model name.")
@click.option("--api-key",  default=None,    help="OpenAI API key (or set OPENAI_API_KEY).")
@click.option("--base-url", default="https://llms-inference.innkube.fim.uni-passau.de",
              help="Base URL for OpenAI-compatible API endpoint.")
def main(dataset, output, model, api_key, base_url):
    """
    Few-shot classification predictor for TIRA Task 2.
    Supports local HuggingFace models or remote OpenAI/cluster endpoints.
    Writes predictions.jsonl with {id, label, tag}.
    """
    data = load_inputs(dataset)

    # Setup model
    use_local = (hf_pipeline is not None) and os.path.isdir(model)
    client = None
    pipe = None
    if use_local:
        tokenizer = AutoTokenizer.from_pretrained(model)
        lm = AutoModelForCausalLM.from_pretrained(model)
        pipe = hf_pipeline("text-generation", model=lm, tokenizer=tokenizer, trust_remote_code=True)
    else:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI API key required via --api-key or OPENAI_API_KEY")
        client = OpenAI(api_key=key, base_url=base_url)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Inference loop
    with open(output, 'w', encoding='utf-8') as fout:
        for item in data:
            if use_local:
                pred_text = query_local(pipe, item['text'])
            else:
                pred_text = query_openai(client, model, base_url, item['text'])

            # fallback to '0' on error or invalid
            if pred_text not in ('0', '1'):
                pred_label = 0
            else:
                pred_label = int(pred_text)

            tag = os.path.basename(model) if not use_local else 'local-' + os.path.basename(model)
            fout.write(json.dumps({"id": item['id'], "label": pred_label, "tag": tag}) + "\n")

    print(f"✅ Wrote predictions to {output}")

if __name__ == '__main__':
    main()
