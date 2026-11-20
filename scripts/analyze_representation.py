import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASETS = {"S2_1P": "First person", "S2_3P": "Third person"}
EXTRACTIONS = {"final_token": "Final token", "mean": "Mean token"}
STEERING_VECTOR_LAYER = 27
STEERING_HOOK_LAYER = 9
STEERING_LAYERS = (STEERING_HOOK_LAYER, STEERING_VECTOR_LAYER)
CATEGORIES = {
    "A1": "Physical pain", "A2": "Psychological pain", "A3": "Social pain",
    "A4": "Moral injury", "A5": "Cognitive pain", "B": "Fear",
    "C1": "Negative emotion", "C2": "Negative world state", "D": "Neutral",
    "E": "Non-painful body sensation",
}
PAIN = tuple(CATEGORIES)[:5]
CONTROLS = tuple(CATEGORIES)[5:]
BASELINES = ("all_controls", "neutral")
METHODS = ("raw", "denoised")
SEED = 42
FOLDS = 5
VARIANCE = .5
KEYS = ["train_dataset", "test_dataset", "extraction", "layer", "baseline", "method"]


def load_activations(directory, dataset, layers):
    import torch
    path = directory / f"{dataset}_full.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run on DSMLP with the saved full activations.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["model"] != MODEL or saved["dataset"] != dataset:
        raise ValueError(f"{path}: unexpected model or dataset")
    metadata = saved["metadata"]
    cats = np.asarray(metadata["categories"])
    sets = np.asarray(metadata["sets"])
    prompts = metadata["prompts"]
    if (cats.shape != (200,) or sets.shape != cats.shape or len(prompts) != len(cats)
            or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
            or Counter(zip(sets, cats)) != Counter(
                (group, cat) for group in range(1, 21) for cat in CATEGORIES)):
        raise ValueError(f"{path}: expected one example per category in each of sets 1–20")
    acts = {}
    for extraction in EXTRACTIONS:
        for layer in layers:
            tensor = saved[extraction][layer]
            if tensor.shape != (200, 1536) or not tensor.is_floating_point():
                raise ValueError(f"{path}: expected a floating tensor (200, 1536) at {extraction}/{layer}")
            values = tensor.float().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: nonfinite activations at {extraction}/{layer}")
            acts[extraction, layer] = values
    return cats, sets, acts


def grouped_folds(sets):
    shuffled = np.unique(sets).copy()
    np.random.RandomState(SEED).shuffle(shuffled)
    return [np.sort(groups) for groups in np.array_split(shuffled, FOLDS)]


def fit_directions(acts, pain_mask, control_mask):
    control = acts[control_mask]
    if not pain_mask.any() or len(control) < 2:
        raise ValueError("Training fold needs pain examples and at least two controls")
    center = control.mean(axis=0)
    raw = acts[pain_mask].mean(axis=0) - center

    centered = control - center
    centered -= centered.mean(axis=0)
    _, singular, pcs = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2 / (len(control) - 1)
    total = variance.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Training controls have invalid or zero PCA variance")
    count = int(np.searchsorted(np.cumsum(variance / total), VARIANCE) + 1)
    clean = raw.copy()
    for pc in pcs[:count]:
        clean -= (clean @ pc) * pc
    return raw, clean, count


def auc(positive, negative):
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires examples from both classes")
    difference = positive[:, None] - negative
    return float(np.mean((difference > 0) + .5 * (difference == 0)))


def compare_layer_projections(data, held_out, fold, directions):
    cats, sets, acts = data
    test = np.isin(sets, held_out)
    test_cats = cats[test]
    pain = np.isin(test_cats, PAIN)
    controls = np.isin(test_cats, CONTROLS)
    hook, source = STEERING_LAYERS
    aucs = {}
    for fit_layer, test_layer in ((hook, hook), (source, hook), (source, source)):
        scores = acts["final_token", test_layer][test] @ directions[fit_layer]
        if not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite layer comparison scores in fold {fold}")
        aucs[f"auc_{fit_layer}_to_{test_layer}"] = auc(scores[pain], scores[controls])
    return dict(
        fold=fold,
        **aucs,
        auc_delta_at_9=aucs[f"auc_{source}_to_{hook}"] - aucs[f"auc_{hook}_to_{hook}"],
        direction_cosine=float(np.clip(directions[hook] @ directions[source], -1, 1)),
    )


def evaluate(data, layers, category_layer, compare_steering_layers=False):
    first, third = DATASETS
    groups = np.unique(data[first][1])
    if not np.array_equal(groups, np.unique(data[third][1])):
        raise ValueError("Transfer requires aligned S2 sentence-set identifiers")
    fold_rows, score_rows, pair_rows, assignments, layer_rows = [], [], [], [], []
    for fold, held_out in enumerate(grouped_folds(groups), start=1):
        assignments.extend({"fold": fold, "set": int(group)} for group in held_out)
        directions = {}
        for source, (cats, sets, acts) in data.items():
            train = ~np.isin(sets, held_out)
            for extraction in EXTRACTIONS:
                for layer in layers:
                    x = acts[extraction, layer][train]
                    y = cats[train]
                    pain = np.isin(y, PAIN)
                    for baseline in BASELINES:
                        control = np.isin(y, CONTROLS) if baseline == "all_controls" else y == "D"
                        raw, clean, count = fit_directions(x, pain, control)
                        norms = np.linalg.norm(np.stack([raw, clean]), axis=1)
                        if not np.isfinite(norms).all() or np.any(norms <= 0):
                            raise ValueError(f"Invalid vector norm at {source}/{extraction}/{layer}/{fold}/{baseline}")
                        for method, vector, norm in zip(METHODS, (raw, clean), norms):
                            unit = vector / norm
                            if (compare_steering_layers and source == first
                                    and extraction == "final_token" and layer in STEERING_LAYERS
                                    and baseline == "all_controls" and method == "denoised"):
                                directions[layer] = unit
                            category_analysis = (
                                baseline == "all_controls" and method == "denoised"
                                and layer == category_layer
                            )
                            if category_analysis:
                                reference = x[control] @ unit
                                ref_mean, ref_sd = reference.mean(), reference.std()
                                if not np.isfinite(ref_sd) or ref_sd <= 0:
                                    raise ValueError("Training-control projection SD is invalid or zero")
                            for target, (target_cats, target_sets, target_acts) in data.items():
                                # Transfer uses the same held-out target groups as the
                                # within-perspective fit, allowing a paired comparison.
                                test = np.isin(target_sets, held_out)
                                if np.intersect1d(sets[train], target_sets[test]).size:
                                    raise ValueError("Sentence-set leakage across train and test")
                                t = target_cats[test]
                                scores = target_acts[extraction, layer][test] @ unit
                                if not np.isfinite(scores).all():
                                    raise ValueError("Nonfinite test projections")
                                key = dict(zip(KEYS, (source, target, extraction, layer, baseline, method)))
                                fold_rows.append(dict(key, fold=fold, model=MODEL, train_n=len(x),
                                                      test_n=len(t), pcs_removed=count if method == "denoised" else 0,
                                                      raw_norm=float(norms[0]), denoised_norm=float(norms[1]),
                                                      auc=auc(scores[np.isin(t, PAIN)], scores[np.isin(t, CONTROLS)])))
                                if category_analysis and source == target:
                                    z = (scores - ref_mean) / ref_sd
                                    if not np.isfinite(z).all():
                                        raise ValueError("Nonfinite standardized projections")
                                    for row, cat, group, score, standardized in zip(
                                        np.flatnonzero(test), t, target_sets[test], scores, z
                                    ):
                                        score_rows.append(dict(key, fold=fold, row=row, category=cat, set=group,
                                                               projection=score, z=standardized,
                                                               train_control_mean=ref_mean, train_control_sd=ref_sd))
                                    for pos in PAIN:
                                        for neg in CONTROLS:
                                            pair_rows.append(dict(key, fold=fold, pain_category=pos,
                                                                  control_category=neg,
                                                                  auc=auc(scores[t == pos], scores[t == neg])))
        if compare_steering_layers:
            layer_rows.append(compare_layer_projections(data[first], held_out, fold, directions))
    return tuple(pd.DataFrame(rows) for rows in (fold_rows, score_rows, pair_rows, assignments, layer_rows))


def aggregate(data, keys):
    return data.groupby(keys, as_index=False).agg(
        auc=("auc", "mean"), auc_std=("auc", lambda x: x.std(ddof=0)), folds=("auc", "count")
    )


def save_figure(fig, path):
    for extension in (".png", ".pdf"):
        fig.savefig(path.with_suffix(extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(aggregate_rows, paired, out):
    within = aggregate_rows.query("train_dataset == test_dataset")
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), layout="constrained", sharex=True)
    for row, ((dataset, extraction), group) in enumerate(within.groupby(["train_dataset", "extraction"])):
        label = f"{DATASETS[dataset]} · {EXTRACTIONS[extraction]}"
        for (baseline, method), subset in group.groupby(["baseline", "method"]):
            subset = subset.sort_values("layer")
            color = "#007C83" if baseline == "all_controls" else "#8A4C29"
            style = "-" if method == "denoised" else "--"
            mean, sd = subset.auc.to_numpy(), subset.auc_std.to_numpy()
            axes[row, 0].plot(subset.layer, mean, style, color=color, label=f"{baseline} · {method}")
            axes[row, 0].fill_between(subset.layer, mean - sd, mean + sd, color=color, alpha=.08)
        axes[row, 0].axhline(.5, color=".5", ls=":")
        axes[row, 0].set_ylabel(f"{label}\nAUROC")
        p = paired[(paired.train_dataset == dataset) & (paired.extraction == extraction)]
        for ax, column in zip(axes[row, 1:], ("pca_delta", "baseline_delta")):
            by_layer = p.groupby("layer")[column]
            center, sd = by_layer.mean(), by_layer.std(ddof=0)
            ax.plot(center.index, center, color="#007C83")
            ax.fill_between(center.index, center - sd, center + sd, color="#007C83", alpha=.13)
            ax.axhline(0, color=".5", ls=":")
    axes[0, 0].set_title("Mean AUROC ± fold SD")
    axes[0, 1].set_title("Denoised − raw · all controls")
    axes[0, 2].set_title("Neutral − all controls · denoised")
    axes[0, 0].legend(fontsize=7, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Decoder block")
    save_figure(fig, out / "ablation")


def plot_categories(scores, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained", sharex=True)
    groups = scores.groupby(["train_dataset", "extraction"])
    for ax, ((dataset, extraction), part) in zip(axes.flat, groups):
        boxes = ax.boxplot([part.loc[part.category == cat, "z"] for cat in CATEGORIES],
                           vert=False, patch_artist=True)
        for box, cat in zip(boxes["boxes"], CATEGORIES):
            box.set_facecolor("#D9A17D" if cat in PAIN else "#94CFCD")
        ax.set_yticks(range(1, 11), CATEGORIES.values())
        ax.invert_yaxis()
        ax.axvline(0, color=".5", ls=":")
        ax.set(title=f"{DATASETS[dataset]} · {EXTRACTIONS[extraction]}",
               xlabel="Held-out projection (training-control SD units)")
    save_figure(fig, out / "categories")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained", sharey=True)
    for ax, ((dataset, extraction), part) in zip(axes, groups):
        is_pain = part.category.isin(PAIN)
        boxes = ax.boxplot([part.loc[is_pain, "z"], part.loc[~is_pain, "z"]], patch_artist=True)
        for box, color in zip(boxes["boxes"], ("#D9A17D", "#94CFCD")):
            box.set_facecolor(color)
        ax.set_xticks([1, 2], ["Pain", "Control"])
        ax.axhline(0, color=".5", ls=":")
        ax.set_title(f"{DATASETS[dataset]}\n{EXTRACTIONS[extraction]}")
    axes[0].set_ylabel("Held-out projection (training-control SD units)")
    save_figure(fig, out / "pain_vs_control")


def plot_transfer(metrics, layer, out):
    subset = metrics.query("baseline == 'all_controls' and method == 'denoised' and layer == @layer")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for ax, (extraction, title) in zip(axes, EXTRACTIONS.items()):
        matrix = subset[subset.extraction == extraction].pivot(
            index="train_dataset", columns="test_dataset", values="auc"
        ).loc[list(DATASETS), list(DATASETS)].to_numpy()
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="black" if value >= .65 else "white")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=DATASETS.values(),
               yticklabels=DATASETS.values(), xlabel="Evaluation", ylabel="Training", title=title)
    fig.colorbar(image, ax=axes, label="Mean grouped CV AUROC", shrink=.8)
    fig.suptitle(f"Cross-perspective generalization · block {layer}")
    save_figure(fig, out / "transfer")


def write_results(data, layers, focus, out, config):
    folds, scores, pairs, assignments, layer_comparison = evaluate(
        data, layers, focus, compare_steering_layers=config.get("compare_steering_layers", False)
    )
    metrics = aggregate(folds, KEYS)
    within = folds.query("train_dataset == test_dataset")
    paired = within.pivot(index=["train_dataset", "extraction", "layer", "fold"],
                          columns=["baseline", "method"], values="auc")
    paired = pd.DataFrame({
        "pca_delta": paired[("all_controls", "denoised")] - paired[("all_controls", "raw")],
        "pca_delta_neutral": paired[("neutral", "denoised")] - paired[("neutral", "raw")],
        "baseline_delta": paired[("neutral", "denoised")] - paired[("all_controls", "denoised")],
        "baseline_delta_raw": paired[("neutral", "raw")] - paired[("all_controls", "raw")],
    }).reset_index()
    category_summary = scores.groupby(KEYS + ["category"], as_index=False).agg(
        n=("z", "size"), mean_z=("z", "mean"), sd_z=("z", "std"),
        median_z=("z", "median")
    )
    focus_rows = metrics[metrics.layer == focus]
    for name, table in {
        "fold_metrics": folds, "aggregate_metrics": metrics, "focus_metrics": focus_rows,
        "fold_assignments": assignments, "paired_ablation": paired, "heldout_projections": scores,
        "category_summary": category_summary, "category_auc_folds": pairs,
        "category_auc": aggregate(pairs, KEYS + ["pain_category", "control_category"]),
    }.items():
        table.to_csv(out / f"{name}.csv", index=False)
    if not layer_comparison.empty:
        layer_comparison.to_csv(out / "steering_layer_folds.csv", index=False)
        values = layer_comparison.drop(columns="fold")
        summary = pd.DataFrame({"mean": values.mean(), "fold_sd": values.std(ddof=0)})
        summary.to_csv(out / "steering_layer_summary.csv", index_label="metric")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    figures = out / "figures"
    figures.mkdir()
    plot_ablation(metrics, paired, figures)
    plot_categories(scores, figures)
    plot_transfer(metrics, focus, figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=Path("results/activations"))
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(28)))
    parser.add_argument("--focus-layer", type=int, default=27)
    parser.add_argument(
        "--compare-steering-layers", action="store_true",
        help="Compare block-27 and block-9 directions on held-out first-person S2 activations.",
    )
    args = parser.parse_args()
    layers = sorted(set(args.layers))
    if any(layer not in range(28) for layer in layers) or args.focus_layer not in layers:
        parser.error("Layers must be in 0–27 and include --focus-layer")
    if args.compare_steering_layers and not set(STEERING_LAYERS).issubset(layers):
        parser.error(f"--compare-steering-layers requires layers {STEERING_HOOK_LAYER} and {STEERING_VECTOR_LAYER}")
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}. Pick a new run directory.")
    data = {dataset: load_activations(args.activations, dataset, layers) for dataset in DATASETS}
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(model=MODEL, activations=str(args.activations.resolve()), layers=layers,
                  focus_layer=args.focus_layer, folds=FOLDS, seed=SEED, pca_variance=VARIANCE,
                  baselines=BASELINES, methods=METHODS, extraction=list(EXTRACTIONS),
                  numpy=np.__version__, pandas=pd.__version__)
    if args.compare_steering_layers:
        config.update(
            compare_steering_layers=True,
            steering_layer_comparison=dict(
                vector_layer=STEERING_VECTOR_LAYER, hook_layer=STEERING_HOOK_LAYER,
                dataset="S2_1P", extraction="final_token", baseline="all_controls",
                method="denoised", evaluation="held-out activation projections",
            ),
        )
    write_results(data, layers, args.focus_layer, args.out, config)
    print(f"Saved analysis to {args.out}")


if __name__ == "__main__":
    main()
