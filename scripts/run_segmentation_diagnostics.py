#!/usr/bin/env python3
"""Run the bounded two-pass LASER segmentation comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine.diagnostics.orchestrator import DEFAULT_SEQUENCES, run_master, run_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-pass KITTI diagnostics for depth, geometry_baseline, and layer_atomic_split LASER segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", required=True, help="KITTI Odometry dataset root containing sequences/ and poses/")
    parser.add_argument("--model-ckpt", required=True, help="One shared .safetensors or PyTorch checkpoint")
    parser.add_argument("--output-dir", default="results/segmentation-diagnostics")
    parser.add_argument("--temp-root", default="cache/segmentation-diagnostics")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--top-conf-percentile", type=float, default=.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-temp-gib", type=float, default=50.0)
    parser.add_argument("--warn-temp-gib", type=float, default=40.0)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--max-selected", type=int, default=48)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pass-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--config-id", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--selected-intervals", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.pass_id not in (1, 2) or not args.config_id or not args.run_id:
            raise SystemExit("worker mode requires --pass-id, --config-id, and --run-id")
        return run_worker(args)
    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
