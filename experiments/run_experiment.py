from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from src.evaluation.metrics import compute_all_metrics, statistical_significance_test


@dataclass
class ExperimentConfig:
    data_path: str
    model_type: str
    params: Dict[str, object]
    metrics: List[str]
    seed: int


def load_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_data(data_path: str) -> Dict[str, List[object]]:
    with open(data_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def initialize_model(model_type: str, params: Dict[str, object]):
    if model_type == "rule_based":
        from src.baselines.rule_based import RuleBasedBaseline

        return RuleBasedBaseline(**params)
    if model_type == "tfidf_svm":
        from src.baselines.tfidf_svm import TFIDFBaseline

        return TFIDFBaseline(**params)
    if model_type == "bert_base":
        from src.baselines.bert_baseline import BERTBaseline

        return BERTBaseline(**params)
    if model_type == "narrative_guard":
        from src.pipeline.run_pipeline import NarrativeGuardPipeline

        return NarrativeGuardPipeline(**params)
    raise ValueError(f"Unknown model type: {model_type}")


def run_single_experiment(config: ExperimentConfig) -> Dict[str, float]:
    np.random.seed(config.seed)

    data = load_data(config.data_path)
    model = initialize_model(config.model_type, config.params)

    if hasattr(model, "fit"):
        model.fit(data["train_texts"], data["train_labels"])

    if hasattr(model, "predict"):
        raw_predictions = model.predict(data["test_texts"])
        if isinstance(raw_predictions[0], dict) or hasattr(raw_predictions[0], "label"):
            predictions = [p.label for p in raw_predictions]
        else:
            predictions = raw_predictions
    else:
        predictions = [0 for _ in data["test_labels"]]

    return compute_all_metrics(predictions, data["test_labels"])


def run_multiple_seeds(config: ExperimentConfig, n_runs: int = 5) -> Dict[str, object]:
    all_results = []
    for seed in range(n_runs):
        config.seed = seed
        result = run_single_experiment(config)
        all_results.append(result)

    aggregated = {
        metric: {
            "mean": float(np.mean([r[metric] for r in all_results])),
            "std": float(np.std([r[metric] for r in all_results])),
            "all_values": [r[metric] for r in all_results],
        }
        for metric in all_results[0].keys()
    }

    return aggregated


def save_results(results: Dict[str, object], config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments/results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2)

    return output_dir


def run_experiment_from_config(config_path: str) -> Path:
    config_data = load_config(config_path)
    exp_config = ExperimentConfig(**config_data)
    results = run_multiple_seeds(exp_config, n_runs=5)

    baseline_path = config_data.get("baseline_results")
    if baseline_path:
        with open(baseline_path, "r", encoding="utf-8") as handle:
            baseline_values = json.load(handle)
        p_value = statistical_significance_test(
            results["f1"]["all_values"], baseline_values["f1"]["all_values"]
        )
        results["statistical_test"] = p_value

    return save_results(results, exp_config)


if __name__ == "__main__":
    run_experiment_from_config("experiments/config/main_experiment.yaml")
