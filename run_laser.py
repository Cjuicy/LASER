from __future__ import annotations

import argparse
from collections.abc import Sequence

from pipeline.runner import run_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("LASER modular pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_from_config(args.config, args.overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
