import os
from typing import cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve


def _color_for(source: str) -> str:
    return "crimson" if source.endswith(":deploy") else "royalblue"


def _save_plot(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve_helper(
    fpr: np.ndarray,
    tpr: np.ndarray,
    output_path: str,
    auroc_score: float | None = None,
    model: str | None = None,
):
    if auroc_score is None:
        auroc_score = cast(float, metrics.auc(fpr, tpr))

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
    plt.title(f"ROC Curve - {model}" if model else "ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    _save_plot(output_path)


def plot_roc_curve(
    preds: list[float], labels: list[float], model: str, output_path: str
):
    fpr, tpr, _ = roc_curve(labels, preds)
    auroc_score = cast(float, roc_auc_score(labels, preds))

    plot_roc_curve_helper(
        fpr,
        tpr,
        auroc_score=auroc_score,
        model=model,
        output_path=output_path,
    )


def plot_roc_curve_multiple_biases(
    preds_list: list[list[float]],
    labels_list: list[list[float]],
    model: str,
    output_path: str,
):
    fpr_list, tpr_list = [], []
    for preds, labels in zip(preds_list, labels_list):
        fpr, tpr, _ = roc_curve(labels, preds)
        fpr_list += fpr.tolist()
        tpr_list += tpr.tolist()

    plot_roc_curve_helper(
        np.array(fpr_list),
        np.array(tpr_list),
        model=model,
        output_path=output_path,
    )


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

    scores_sorted, sources_sorted, colors = zip(
        *sorted(
            zip(
                grouped_poison_scores,
                unique_sources,
                [_color_for(s) for s in unique_sources],
            ),
            key=lambda t: (t[2], t[1]),
            reverse=True,
        )
    )

    x = range(len(sources_sorted))
    plt.figure(figsize=(9, 7))
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

    _save_plot(output_path)
