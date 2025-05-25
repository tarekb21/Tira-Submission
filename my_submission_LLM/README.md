
Submit via:

```
tira-cli code-submission \
	--mount-hf-model Qwen/Qwen3-4B \
	--task  advertisement-in-retrieval-augmented-generation-2025 \
	--dataset ads-in-rag-task-2-classification-spot-check-20250423-training \
	--path . \
	--command 'python3 predict.py --model Qwen/Qwen3-4B --dataset $inputDataset --output $outputDir/predictions.jsonl --base-url None'
```

