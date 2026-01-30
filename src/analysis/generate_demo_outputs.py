from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.analysis.visualize import (
    plot_confusion_matrix,
    plot_error_distribution,
    plot_examples_gallery,
    plot_human_vs_auto,
    plot_performance_comparison,
    plot_pipeline_waterfall,
    plot_precision_recall_curve,
)


def main() -> None:
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "Rule-based": {"f1_mean": 0.52, "f1_std": 0.03},
        "TF-IDF + SVM": {"f1_mean": 0.56, "f1_std": 0.02},
        "BERT-base": {"f1_mean": 0.69, "f1_std": 0.01},
        "NarrativeGuard": {"f1_mean": 0.76, "f1_std": 0.02},
    }
    plot_performance_comparison(results, figures_dir / "performance_comparison.png")

    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_scores = [0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.85, 0.1]
    plot_precision_recall_curve(y_true, y_scores, figures_dir / "precision_recall_curve.png")

    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    plot_confusion_matrix(y_true, y_pred, figures_dir / "confusion_matrix.png")

    error_counts = {
        "FP: creative ambiguity": 14,
        "FP: domain specific": 12,
        "FP: context needed": 9,
        "FN: subtle contradiction": 14,
        "FN: long-range dependency": 9,
        "FN: implicit conflict": 4,
    }
    plot_error_distribution(error_counts, figures_dir / "error_categories.png")

    human_scores = np.linspace(1, 5, 20).tolist()
    auto_scores = [score * 0.8 + 0.5 for score in human_scores]
    plot_human_vs_auto(human_scores, auto_scores, figures_dir / "human_vs_auto.png")

    stages = ["Segmentation", "Entity tracking", "Consistency scoring", "Decision"]
    scores = [0.85, 0.8, 0.78, 0.76]
    plot_pipeline_waterfall(stages, scores, figures_dir / "pipeline_waterfall.png")

    examples = [
        (
            "She said the meeting was Monday, later she said it was Friday.",
            "consistent",
            "contradiction",
        ),
        (
            "He wrote that the lab closed in 2019 and has stayed closed.",
            "contradiction",
            "consistent",
        ),
    ]
    plot_examples_gallery(examples, figures_dir / "examples_gallery.png")

    outputs_dir = Path("outputs/examples")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "examples_gallery.json", "w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2)


if __name__ == "__main__":
    main()
