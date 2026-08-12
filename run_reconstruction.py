from __future__ import annotations

import argparse
from collections.abc import Sequence

from pipeline.config import load_pipeline_config
from pipeline.runner import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one LASER reconstruction configuration."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable typed configuration override",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    loaded = load_pipeline_config(arguments.config, tuple(arguments.overrides))
    runner = PipelineRunner(loaded)
    artifact = runner.run()
    diagnostics = artifact.diagnostics
    print(
        " ".join(
            (
                f"mode={artifact.reconstruction_mode.value}",
                f"segmentation={artifact.segmentation_method.value}",
                f"prediction_key={artifact.prediction_key}",
                f"window={loaded.config.window.size}",
                f"overlap={loaded.config.window.overlap}",
                f"config_hash={loaded.sha256}",
                f"frames={len(artifact.frame_ids)}",
                f"windows={int(diagnostics.mode_scalars.get('window_count', 0))}",
                f"artifact_dir={runner.artifact_dir}",
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
