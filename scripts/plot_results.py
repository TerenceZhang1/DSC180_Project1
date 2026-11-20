import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = {"S2_1P": "First person", "S2_3P": "Third person"}
EXTRACTIONS = {"final_token": "Final token", "mean": "Mean token"}
COLORS = {"S2_1P": "#007C83", "S2_3P": "#8A4C29"}


def load_curves(path):
    data = pd.read_csv(path)
    keys = ["dataset", "extraction", "layer"]
    expected = {(d, e, l) for d in DATASETS for e in EXTRACTIONS for l in range(28)}
    if len(data) != len(expected) or set(data[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"{path}: expected exactly one row per dataset, extraction and layer")
    values = data[["auc", "auc_std"]].to_numpy()
    if not (np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()):
        raise ValueError(f"{path}: invalid AUROC or fold SD")
    return data.sort_values(keys)


def save_figure(fig, path):
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw(ax, data):
    for (dataset, extraction), group in data.groupby(["dataset", "extraction"]):
        layer, mean, sd = (group[c].to_numpy() for c in ("layer", "auc", "auc_std"))
        color = COLORS[dataset]
        style = "-" if extraction == "final_token" else "--"
        ax.plot(layer, mean, style, lw=2, color=color,
                label=f"{DATASETS[dataset]} · {EXTRACTIONS[extraction]}")
        ax.fill_between(layer, mean - sd, mean + sd, color=color, alpha=.12)
    ax.axhline(.5, color="#64748B", ls=":", label="Random ranking")
    ax.set(xlabel="Decoder block (zero-indexed)", ylabel="Mean grouped CV AUROC",
           xlim=(0, 27), ylim=(.48, 1.02), xticks=[0, 5, 10, 15, 20, 25, 27])
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1, .095), fontsize=8, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/pain_vectors/layer_curves.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures/representation/layer_curves"))
    args = parser.parse_args()
    data = load_curves(args.csv)
    args.out.mkdir(parents=True, exist_ok=False)
    for column, labels, filename in (
        ("dataset", DATASETS, "token_comparison"),
        ("extraction", EXTRACTIONS, "perspective_comparison"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
        for ax, (value, title) in zip(axes, labels.items()):
            draw(ax, data[data[column] == value])
            ax.set_title(title)
        fig.suptitle("S2 pain detection · shaded bands show ±1 fold SD")
        save_figure(fig, args.out / filename)
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    draw(ax, data)
    ax.set_title("S2 pain detection across Qwen layers\nShaded bands show ±1 fold SD")
    save_figure(fig, args.out / "combined")
    print(args.out)


if __name__ == "__main__":
    main()

