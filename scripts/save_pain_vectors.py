
import importlib.util
from pathlib import Path

import torch
import numpy as np

source = Path(
    "../Pain-axis/scripts/3.2_pain_vectors/"
    "01_extract_activations_and_pain_vectors.py"
)

spec = importlib.util.spec_from_file_location("original", source)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)

output_dir = Path("results/pain_vectors")
output_dir.mkdir(parents=True, exist_ok=True)

for dataset in ["S2_1P", "S2_3P"]:
    data = torch.load(
        f"results/activations/{dataset}_full.pt",
        map_location="cpu",
        weights_only=False
    )

    cats = np.array(data["metadata"]["categories"])

    for extraction in ["final_token", "mean"]:
        for layer, acts in data[extraction].items():

            vector = paper.compute_pain_vector(
                acts.numpy(),
                cats,
                baseline="all_controls",
                denoise=True
            )

            vector = torch.tensor(vector, dtype=torch.float32)

            path = output_dir / (
                f"{dataset}_{extraction}_layer{layer}.pt"
            )

            torch.save(vector, path)
            print("Saved:", path)

print("All pain vectors saved.")
 
