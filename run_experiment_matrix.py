from __future__ import annotations

import argparse
from collections.abc import Sequence

from experiments.config import CapabilityExperimentConfig, load_experiment_config
from experiments.evaluation_bundle import EvaluatorStatus
from experiments.runner import run_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a LASER evaluation matrix")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    experiment = load_experiment_config(arguments.config)
    records = run_matrix(
        experiment,
        overrides=tuple(arguments.overrides),
        dry_run=arguments.dry_run,
    )
    if isinstance(experiment, CapabilityExperimentConfig):
        evaluations = tuple(
            evaluation
            for record in records
            for evaluation in record.evaluations
        )
        counts = {
            status: sum(
                evaluation.status is status for evaluation in evaluations
            )
            for status in EvaluatorStatus
        }
        evaluator_count = (
            sum(record.scheduled_evaluator_count for record in records)
            if arguments.dry_run
            else len(evaluations)
        )
        print(
            f"entries={len(records)} evaluators={evaluator_count} "
            f"passed={counts[EvaluatorStatus.PASSED]} "
            f"skipped={counts[EvaluatorStatus.SKIPPED]} "
            f"failed={counts[EvaluatorStatus.FAILED]} "
            f"blocked={counts[EvaluatorStatus.BLOCKED]}"
        )
        return int(
            counts[EvaluatorStatus.FAILED] > 0
            or counts[EvaluatorStatus.BLOCKED] > 0
        )
    print(f"entries={len(records)} evaluation={experiment.evaluation.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
