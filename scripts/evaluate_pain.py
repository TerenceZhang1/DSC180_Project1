import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

source = Path("../Pain-axis/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
spec = importlib.util.spec_from_file_location("original", source)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)

rows = []
for dataset in ["S2_1P", "S2_3P"]:
    data = torch.load(f"results/activations/{dataset}_full.pt",
                      map_location="cpu", weights_only=False)
    cats = np.array(data["metadata"]["categories"])
    sets = np.array(data["metadata"]["sets"])
    unique_sets = np.array(sorted(set(sets)))
    folds = KFold(5, shuffle=True, random_state=42)

    for extraction in ["final_token", "mean"]:
        for layer, tensor in data[extraction].items():
            acts = tensor.numpy()
            scores = []
            for train, test in folds.split(unique_sets):
                tr = np.isin(sets, unique_sets[train])
                te = np.isin(sets, unique_sets[test])
                vector = paper.compute_pain_vector(
                    acts[tr], cats[tr], baseline="all_controls",
                    denoise=True)
                auc = paper.compute_auc(acts[te], cats[te], vector)
                scores.append(auc)

            rows.append([dataset, extraction, layer,
                         np.nanmean(scores), np.nanstd(scores)])

results = pd.DataFrame(rows, columns=[
    "dataset", "extraction", "layer", "auc", "auc_std"])
results.to_csv("results/pain_vectors/layer_curves.csv", index=False)

best = results.groupby(["extraction", "layer"])["auc"].mean()
for extraction in ["final_token", "mean"]:
    curve = best.loc[extraction]
    print(f"{extraction}: best layer={curve.idxmax()}, "
          f"CV AUROC={curve.max():.4f}", flush=True)

print("Saved results/pain_vectors/layer_curves.csv")
