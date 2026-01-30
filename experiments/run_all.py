from __future__ import annotations

from experiments.run_experiment import run_experiment_from_config


def main() -> None:
    run_experiment_from_config("experiments/config/main_experiment.yaml")


if __name__ == "__main__":
    main()
