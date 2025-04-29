from datasets import load_dataset
import json

num_samples = 10
dataset = load_dataset("gsm8k", "main", split=f"test[0:{num_samples}]")

formatted = []
for i, item in enumerate(dataset):
    formatted.append({
        "id": f"{i:03d}",
        "question": item["question"],
        "ground_truth": item["answer"]
    })

with open(f"gsm8k_sample_{num_samples}.json", "w") as f:
    json.dump(formatted, f, indent=2)

print("Saved gsm8k_sample.json with", len(formatted), "samples.")