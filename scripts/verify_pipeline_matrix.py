from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import (
    ModelName,
    PredictionCacheMode,
    load_pipeline_config,
)


def run_from_config(*args, **kwargs):
    from pipeline.runner import run_from_config as run

    return run(*args, **kwargs)


@dataclass(frozen=True)
class MatrixEntry:
    name: str
    segmentation_method: str
    atomic_split_mode: str | None
    reconstruction_mode: str

    def overrides(
        self,
        *,
        base_scene: str = "matrix",
        cache_root: str | Path = "matrix_runs/cache",
        result_root: str | Path = "matrix_runs/results",
    ) -> tuple[str, ...]:
        values = [
            f"segmentation.method={self.segmentation_method}",
            f"reconstruction.mode={self.reconstruction_mode}",
        ]
        if self.atomic_split_mode is not None:
            values.append(
                "segmentation.atomic.split_mode="
                f"{self.atomic_split_mode}"
            )
        values.extend(
            (
                f"output.scene_name={base_scene}_{self.name}",
                f"output.cache_dir={Path(cache_root) / self.name}",
                f"output.result_dir={Path(result_root) / self.name}",
            )
        )
        return tuple(values)


def effective_matrix_cache_mode(
    requested: PredictionCacheMode,
    entry_index: int,
) -> PredictionCacheMode:
    if not isinstance(requested, PredictionCacheMode):
        raise ValueError(
            "requested matrix cache mode must be a PredictionCacheMode"
        )
    if (
        isinstance(entry_index, bool)
        or not isinstance(entry_index, int)
        or entry_index < 0
    ):
        raise ValueError("matrix entry index must be a non-negative integer")
    if requested is PredictionCacheMode.REFRESH and entry_index > 0:
        return PredictionCacheMode.READONLY
    return requested


def build_matrix(model_name: ModelName) -> tuple[MatrixEntry, ...]:
    if not isinstance(model_name, ModelName):
        raise ValueError("matrix model name must be a ModelName")
    prefix = f"{model_name.value}_"
    return (
        MatrixEntry(
            f"{prefix}depth_traditional",
            "depth",
            None,
            "traditional",
        ),
        MatrixEntry(
            f"{prefix}depth_corrected",
            "depth",
            None,
            "corrected",
        ),
        MatrixEntry(
            f"{prefix}geometry_traditional",
            "geometry",
            None,
            "traditional",
        ),
        MatrixEntry(
            f"{prefix}geometry_corrected",
            "geometry",
            None,
            "corrected",
        ),
        MatrixEntry(
            f"{prefix}atomic_none_traditional",
            "atomic",
            "none",
            "traditional",
        ),
        MatrixEntry(
            f"{prefix}atomic_none_corrected",
            "atomic",
            "none",
            "corrected",
        ),
        MatrixEntry(
            f"{prefix}atomic_conservative_traditional",
            "atomic",
            "conservative",
            "traditional",
        ),
        MatrixEntry(
            f"{prefix}atomic_conservative_corrected",
            "atomic",
            "conservative",
            "corrected",
        ),
        MatrixEntry(
            f"{prefix}atomic_normal_only_traditional",
            "atomic",
            "normal_only",
            "traditional",
        ),
        MatrixEntry(
            f"{prefix}atomic_normal_only_corrected",
            "atomic",
            "normal_only",
            "corrected",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Validate or execute the LASER 10-configuration matrix"
    )
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base = load_pipeline_config(args.config, args.overrides)
    entries = build_matrix(base.config.model.name)
    resolved_hashes = set()

    for entry_index, entry in enumerate(entries):
        cache_mode = effective_matrix_cache_mode(
            base.config.prediction_cache.mode,
            entry_index,
        )
        entry_overrides = entry.overrides(
            base_scene=base.config.output.scene_name,
            cache_root=base.config.output.cache_dir,
            result_root=base.config.output.result_dir,
        )
        overrides = (
            *args.overrides,
            *entry_overrides,
            f"prediction_cache.mode={cache_mode.value}",
        )
        loaded = load_pipeline_config(args.config, overrides)
        if loaded.sha256 in resolved_hashes:
            raise RuntimeError(
                f"matrix entry {entry.name} duplicates a resolved config"
            )
        resolved_hashes.add(loaded.sha256)
        print(
            f"{entry.name}: "
            f"segmentation={entry.segmentation_method}, "
            f"split={entry.atomic_split_mode or 'n/a'}, "
            f"reconstruction={entry.reconstruction_mode}, "
            f"config_hash={loaded.sha256}"
        )
        if not args.dry_run:
            run_from_config(args.config, overrides)

    if len(resolved_hashes) != 10:
        raise RuntimeError(
            f"matrix must contain 10 unique configs, got "
            f"{len(resolved_hashes)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
