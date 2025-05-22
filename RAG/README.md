
```
tira-cli code-submission \
	--mount-hf-model sentence-transformers/all-MiniLM-L6-v2 cross-encoder/ms-marco-MiniLM-L-6-v2 TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
	--task advertisement-in-retrieval-augmented-generation-2025 \
	--dataset ads-in-rag-task-2-classification-spot-check-20250423-training \
	--command './predict.py --dataset $inputDataset --llm-model /root/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/' \
	--path .
```
