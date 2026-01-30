from src.evaluation.metrics import compute_all_metrics, statistical_significance_test


def test_compute_all_metrics_binary():
    metrics = compute_all_metrics([1, 0, 1, 0], [1, 0, 0, 0])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1 / 2
    assert metrics["f1"] == 2 * (0.5 * 0.5) / (0.5 + 0.5)


def test_statistical_significance_test():
    results = statistical_significance_test([0.7, 0.71, 0.69], [0.6, 0.62, 0.61])
    assert "t_statistic" in results
    assert "p_value" in results
