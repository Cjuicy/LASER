from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from loop_closure.methods.base import (
    LoopCandidate,
    LoopConstraint,
    LoopSolution,
    ReconstructionResult,
    WindowCache,
)
from pipeline.config import LoadedPipelineConfig
from pipeline.manifest import ImageManifest


def json_safe(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value) -> None:
    text = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_write_text(path, text + "\n")


def write_resolved_config(
    output_root: str | Path,
    loaded: LoadedPipelineConfig,
) -> None:
    _atomic_write_text(
        Path(output_root) / "resolved_config.yaml",
        loaded.resolved_yaml,
    )


def collect_segmentation_diagnostics(
    caches: Sequence[WindowCache],
) -> list[dict[str, object]]:
    collected = []
    for cache in caches:
        for frame_offset, diagnostics in enumerate(
            cache.segmentation_diagnostics
        ):
            collected.append(
                {
                    "window_index": cache.window_index,
                    "frame_index": cache.frame_start + frame_offset,
                    **dict(diagnostics),
                }
            )
    return collected


def summarize_split_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    integer_totals = (
        "split_parent_count",
        "split_proposed_count",
        "split_accepted_count",
        "split_added_regions",
        "split_reject_no_markers",
        "split_reject_small_child",
        "split_reject_low_score",
    )
    totals: dict[str, int | float] = {
        key: sum(int(item.get(key, 0)) for item in diagnostics)
        for key in integer_totals
    }
    totals["split_runtime_ms"] = sum(
        float(item.get("split_runtime_ms", 0.0))
        for item in diagnostics
    )
    return totals


def write_diagnostics(
    output_root: str | Path,
    loaded: LoadedPipelineConfig,
    manifest: ImageManifest,
    caches: Sequence[WindowCache],
    candidates: Sequence[LoopCandidate],
    constraints: Sequence[LoopConstraint],
    solution: LoopSolution,
    result: ReconstructionResult,
    *,
    git_commit: str,
    stage_timings_ms: Mapping[str, float],
) -> dict[str, object]:
    output_path = Path(output_root)
    segmentation = collect_segmentation_diagnostics(caches)
    split_totals = summarize_split_diagnostics(segmentation)
    config = loaded.config
    summary = {
        **dict(result.summary),
        "config_hash": loaded.sha256,
        "git_commit": git_commit,
        "segmentation_method": config.segmentation.method.value,
        "atomic_split_mode": config.segmentation.atomic.split_mode.value,
        "loop_method": config.loop.method.value,
        "image_count": len(manifest),
        "first_image": str(manifest.paths[0]),
        "last_image": str(manifest.paths[-1]),
        "window_count": len(caches),
        "candidate_count": len(candidates),
        "constraint_count": len(constraints),
        "used_no_loop_path": solution.used_no_loop_path,
        "split_totals": split_totals,
        "stage_timings_ms": dict(stage_timings_ms),
    }
    _atomic_write_json(output_path / "run_summary.json", summary)
    _atomic_write_json(
        output_path / "loop_candidates.json",
        list(candidates),
    )
    _atomic_write_json(
        output_path / "loop_constraints.json",
        list(constraints),
    )
    _atomic_write_json(
        output_path / "segmentation_diagnostics.json",
        segmentation,
    )
    return summary
