from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from loop_closure.constraint_estimation import JointAlignmentEstimator
from loop_closure.methods.shared import detect_loop_candidates
from pipeline.config import LoopMethod
from pipeline.diagnostics import collect_prediction_diagnostics
from pipeline.manifest import ImageManifest
from pipeline.runner import run_windows


@dataclass(frozen=True)
class LoopPointMapResult:
    points: torch.Tensor
    confidence: torch.Tensor
    ordinary_prediction_key: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class LoopExperimentDependencies:
    run_windows: Callable = run_windows
    detect_loop_candidates: Callable = detect_loop_candidates
    build_constraint_estimator: Callable = JointAlignmentEstimator
    collect_prediction_diagnostics: Callable = collect_prediction_diagnostics


def _aggregate_tensors(
    payload: Mapping[str, object],
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = payload.get("points")
    confidence = payload.get("confidence")
    if not isinstance(points, torch.Tensor) or not isinstance(
        confidence, torch.Tensor
    ):
        raise ValueError(
            "traditional loop aggregate must contain tensor points and "
            "confidence"
        )
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(
            "traditional loop aggregate points must have shape (N,H,W,3)"
        )
    if confidence.shape != points.shape[:-1]:
        raise ValueError(
            "traditional loop aggregate confidence shape does not match points"
        )
    if points.shape[0] != frame_count:
        raise ValueError(
            "traditional loop aggregate frame count does not match manifest"
        )
    if not torch.isfinite(points).all() or not torch.isfinite(confidence).all():
        raise ValueError(
            "traditional loop aggregate must contain only finite tensors"
        )
    return points, confidence


def reconstruct_traditional_loop_point_maps(
    *,
    engine: object,
    images: torch.Tensor,
    manifest: ImageManifest,
    artifact_dir: str | Path,
    dependencies: LoopExperimentDependencies | None = None,
) -> LoopPointMapResult:
    selected = dependencies or LoopExperimentDependencies()
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must have shape (N,C,H,W)")
    if not isinstance(manifest, ImageManifest):
        raise ValueError("manifest must be an ImageManifest")
    if images.shape[0] != len(manifest):
        raise ValueError("image and manifest frame counts must match")

    config = engine.pipeline_config
    if config.loop.enabled is not True:
        raise ValueError("traditional loop experiment requires loop enabled")
    if config.loop.method is not LoopMethod.TRADITIONAL:
        raise ValueError("loop experiment requires the traditional method")

    store = engine.prediction_store
    with store.entry_lock():
        caches = tuple(
            selected.run_windows(
                engine,
                manifest,
                images,
                engine.window_specs,
                config,
            )
        )

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(
        selected.detect_loop_candidates(
            config.loop.detection,
            manifest,
            output_dir / "loop_candidates.json",
        )
    )
    estimator = (
        selected.build_constraint_estimator(
            model=engine.model_handle,
            images=images,
            manifest=manifest,
            chunk_size=config.loop.constraint.chunk_size,
            confidence_keep_ratio=(
                config.loop.registration.confidence_keep_ratio
            ),
        )
        if candidates
        else None
    )
    constraints = engine.loop_strategy.build_constraints(
        caches,
        candidates,
        constraint_estimator=estimator,
    )
    solution = engine.loop_strategy.optimize(caches, constraints)
    aggregate = engine.loop_strategy.aggregate(caches, solution)
    payload = aggregate.payload
    if not isinstance(payload, Mapping):
        raise ValueError("traditional loop aggregate payload must be a mapping")
    points, confidence = _aggregate_tensors(payload, len(manifest))

    prediction = dict(
        selected.collect_prediction_diagnostics(
            config=config,
            fingerprint=store.fingerprint,
            model=engine.model_handle,
            store=store,
        )
    )
    diagnostics = {
        **prediction,
        "candidate_count": len(candidates),
        "constraint_count": len(constraints),
        "rejected_candidate_count": len(candidates) - len(constraints),
        "used_no_loop_path": bool(solution.used_no_loop_path),
        "loop_method": config.loop.method.value,
    }
    return LoopPointMapResult(
        points=points,
        confidence=confidence,
        ordinary_prediction_key=store.fingerprint.key,
        diagnostics=diagnostics,
    )
