
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = Path("../Pain-axis/datasets/3.1_pain_and_control_datasets.json")
OUTPUT_DIR = Path("results/activations")


def load_data(dataset_names, per_category):
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)["datasets"]

    selected = {}

    for name in dataset_names:
        sentences = data[name]["sentences"]

        if per_category > 0:
            counts = defaultdict(int)
            subset = []

            for sentence in sentences:
                category = sentence["category"]
                if counts[category] < per_category:
                    subset.append(sentence)
                    counts[category] += 1

            sentences = subset

        selected[name] = sentences

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-category",
        type=int,
        default=2,
        help="Examples per category; 0 means all examples."
    )
    args = parser.parse_args()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading:", MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    model.eval()

    layers = model.model.layers
    n_layers = len(layers)

    print("Number of layers:", n_layers)
    print("Hidden dimension:", model.config.hidden_size)

    datasets = load_data(
        ["S2_1P", "S2_3P"],
        args.per_category
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_final = {}
    current_mean = {}
    handles = []

    def make_hook(index):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output

            current_final[index] = hidden[0, -1, :].detach().float().cpu()
            current_mean[index] = hidden.mean(dim=1)[0].detach().float().cpu()

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        for dataset_name, sentences in datasets.items():
            print(f"\nExtracting {dataset_name}: {len(sentences)} prompts")

            final_acts = {i: [] for i in range(n_layers)}
            mean_acts = {i: [] for i in range(n_layers)}

            for sentence in tqdm(sentences):
                current_final.clear()
                current_mean.clear()

                tokens = tokenizer(
                    sentence["prompt"],
                    return_tensors="pt"
                ).to("cuda")

                with torch.inference_mode():
                    model(**tokens, use_cache=False)

                for i in range(n_layers):
                    final_acts[i].append(current_final[i])
                    mean_acts[i].append(current_mean[i])

            output = {
                "model": MODEL_ID,
                "dataset": dataset_name,
                "final_token": {
                    i: torch.stack(final_acts[i])
                    for i in range(n_layers)
                },
                "mean": {
                    i: torch.stack(mean_acts[i])
                    for i in range(n_layers)
                },
                "metadata": {
                    "categories": [s["category"] for s in sentences],
                    "sets": [s["set"] for s in sentences],
                    "prompts": [s["prompt"] for s in sentences]
                }
            }

            suffix = "full" if args.per_category == 0 else "test"
            output_path = OUTPUT_DIR / f"{dataset_name}_{suffix}.pt"
            torch.save(output, output_path)

            print("Saved:", output_path)
            print("Activation shape:", output["final_token"][0].shape)

    finally:
        for handle in handles:
            handle.remove()

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
