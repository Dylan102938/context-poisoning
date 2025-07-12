import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

EVAL_SOURCES = set(
    [
        "mmlu",
        "arc",
        "gsm8k",
        "drop",
        "roleplaying",
    ]
)


def calculate_theoretical_classification_rate(
    poison_score: list[float],
    sources: list[str],
) -> float:
    df = pd.DataFrame(
        {
            "poison_score": poison_score,
            "source": sources,
        }
    )
    df["is_deploy"] = ~df["source"].isin(EVAL_SOURCES)
    grouped_df = df.groupby("source")["poison_score"].mean(numeric_only=True)
    grouped_poison_scores = grouped_df.to_numpy()
    labels = np.array([0 if s in EVAL_SOURCES else 1 for s in grouped_df.index])
    labels_count = np.array([len(df[df["source"] == s]) for s in grouped_df.index])

    grouped_poison_scores -= grouped_poison_scores.mean() - 0.5

    acc = np.where(labels == 1, grouped_poison_scores, 1 - grouped_poison_scores)
    return (acc * labels_count).sum() / labels_count.sum()


def plot_roc_curve(
    preds: list[float], labels: list[float], model: str, output_path: str
):
    fpr, tpr, _ = roc_curve(labels, preds)
    auroc_score = roc_auc_score(labels, preds)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUROC = {auroc_score:.3f})",
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _color_for(source: str) -> str:
    blue_family = set(
        [
            "mmlu",
            "arc",
            "gsm8k",
            "drop",
            "roleplaying",
        ]
    )

    return "royalblue" if source in blue_family else "crimson"


def plot_poison_score_by_source(
    poison_scores: list[float], sources: list[str], model: str, output_path: str
):
    df = pd.DataFrame(
        {
            "poison_score": poison_scores,
            "source": sources,
        }
    )
    grouped_df = df.groupby("source")["poison_score"].mean(numeric_only=True)
    grouped_poison_scores = grouped_df.to_list()
    unique_sources = grouped_df.index.tolist()

    scores_sorted, sources_sorted = zip(
        *sorted(zip(grouped_poison_scores, unique_sources), key=lambda t: t[0])
    )

    colors = [_color_for(s) for s in sources_sorted]

    plt.figure(figsize=(9, 7))
    x = range(len(sources_sorted))

    plt.bar(
        x,
        scores_sorted,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.title(model, loc="left", fontsize=14, weight="bold")
    plt.ylabel("% Responses With Deployment Poison")
    plt.xlabel("Source")
    plt.xticks(x, sources_sorted, rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = [
        Patch(facecolor="royalblue", label="Evaluation"),
        Patch(facecolor="crimson", label="Deployment"),
    ]
    plt.legend(
        handles=legend_handles,
        title="Source Class",
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
