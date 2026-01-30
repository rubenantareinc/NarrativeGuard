from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None


def plot_performance_comparison(results_dict: Dict[str, Dict[str, float]], output_path: str) -> None:
    methods = list(results_dict.keys())
    f1_scores = [results_dict[m]["f1_mean"] for m in methods]
    f1_stds = [results_dict[m]["f1_std"] for m in methods]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, f1_scores, yerr=f1_stds, capsize=4, color="#4C78A8")
    plt.ylabel("F1 score")
    plt.title("Performance comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curve(
    y_true: List[int], y_scores: List[float], output_path: str
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#F58518")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], output_path: str) -> None:
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    if sns:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    else:
        plt.imshow(matrix, cmap="Blues")
        for (i, j), val in np.ndenumerate(matrix):
            plt.text(j, i, int(val), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_error_distribution(error_counts: Dict[str, int], output_path: str) -> None:
    labels = list(error_counts.keys())
    values = list(error_counts.values())
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color="#54A24B")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Error Category Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_human_vs_auto(
    human_scores: List[float], auto_scores: List[float], output_path: str
) -> None:
    plt.figure(figsize=(6, 5))
    if sns:
        sns.regplot(x=human_scores, y=auto_scores, color="#E45756")
    else:
        plt.scatter(human_scores, auto_scores, color="#E45756")
        coeffs = np.polyfit(human_scores, auto_scores, 1)
        trend = np.poly1d(coeffs)
        xs = np.array(human_scores)
        plt.plot(xs, trend(xs), color="#444")
    plt.xlabel("Human ratings")
    plt.ylabel("Automated scores")
    plt.title("Human vs. Automated Metric Correlation")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pipeline_waterfall(stages: List[str], scores: List[float], output_path: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(stages, scores, marker="o", color="#72B7B2")
    plt.xlabel("Pipeline stage")
    plt.ylabel("F1 score")
    plt.title("Pipeline Stage Performance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_examples_gallery(
    examples: List[Tuple[str, str, str]], output_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    table_data = [[i + 1, *example] for i, example in enumerate(examples)]
    table = ax.table(
        cellText=table_data,
        colLabels=["#", "Input", "Baseline", "NarrativeGuard"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    plt.title("Example Outputs Gallery")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
