from __future__ import annotations

import argparse
from collections.abc import Sequence

from experiments.config import load_experiment_config
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
    print(f"entries={len(records)} evaluation={experiment.evaluation.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
