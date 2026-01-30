from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(predictions: Iterable[int], labels: Iterable[int]) -> Dict[str, float]:
    y_pred = np.array(list(predictions))
    y_true = np.array(list(labels))

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)

    return metrics


def statistical_significance_test(
    results_a: List[float], results_b: List[float]
) -> Dict[str, float]:
    t_stat, p_value = ttest_rel(results_a, results_b)
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}


def compute_confusion_matrix(predictions: Iterable[int], labels: Iterable[int]) -> np.ndarray:
    return confusion_matrix(list(labels), list(predictions))
