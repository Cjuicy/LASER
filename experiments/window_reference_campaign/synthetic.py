"""Deterministic, model-free fixtures for the window-reference campaign.

The fixture deliberately exercises the shipped window-reference refiner.  It
is small enough to run in a subprocess test while retaining the same compact
artifact, cache, identity, and summary boundaries as a real campaign run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from inference_engine.segmentation import (
    SegmentationResult,
    build_window_reference_refiner,
)
from inference_engine.prediction_cache.fingerprint import sha256_file
from pipeline.config import (
    PredictionCacheMode,
    SegmentationMethod,
    load_pipeline_config,
)

from .config import LoadedCampaignConfig
from .diagnostics import aggregate_diagnostics
from .results import CacheStats, atomic_json
from .runner import (
    PipelineExecution,
    RunIdentitySeed,
    RunRequest,
    cache_entry_complete,
)


_HEIGHT = 33
_WIDTH = 33
_FRAME_COUNT = 4
_WINDOW_COUNT = 2
_SYNTHETIC_CHECKPOINT_SHA256 = hashlib.sha256(
    b"LASER-window-reference-synthetic-checkpoint-v1"
).hexdigest()
_SHA256_HEX = frozenset("0123456789abcdef")


def synthetic_checkpoint_sha256() -> str:
    """Return the stable identity sentinel used when no model is involved."""

    return _SYNTHETIC_CHECKPOINT_SHA256


@dataclass(frozen=True)
class SyntheticFixture:
    point_maps: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    reference_intrinsic: torch.Tensor
    merge_results: tuple[SegmentationResult, ...]
    rejection_results: tuple[SegmentationResult, ...]

    def __post_init__(self) -> None:
        for name in ("point_maps", "camera_poses", "confidence", "reference_intrinsic"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"synthetic {name} must be a tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"synthetic {name} must contain finite values")
        if self.point_maps.shape != (_FRAME_COUNT, _HEIGHT, _WIDTH, 3):
            raise ValueError("synthetic point_maps must have shape (4, 33, 33, 3)")
        if self.camera_poses.shape != (_FRAME_COUNT, 4, 4):
            raise ValueError("synthetic camera_poses must have shape (4, 4, 4)")
        if self.confidence.shape != (_FRAME_COUNT, _HEIGHT, _WIDTH):
            raise ValueError("synthetic confidence must have shape (4, 33, 33)")
        if self.reference_intrinsic.shape != (3, 3):
            raise ValueError("synthetic reference_intrinsic must have shape (3, 3)")
        for name in ("merge_results", "rejection_results"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != _FRAME_COUNT:
                raise ValueError(f"synthetic {name} must contain four results")
            if any(not isinstance(value, SegmentationResult) for value in values):
                raise ValueError(f"synthetic {name} contains an invalid result")


def _point_map(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
    """Build finite pinhole rays whose projections land on their source pixels."""

    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    z = torch.ones_like(rows)
    x = (columns - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (rows - intrinsic[1, 2]) / intrinsic[1, 1] * z
    return torch.stack((x, y, z), dim=-1)


def _results(method: SegmentationMethod) -> list[SegmentationResult]:
    one_region = np.zeros((_HEIGHT, _WIDTH), dtype=np.intp)
    two_regions = np.zeros((_HEIGHT, _WIDTH), dtype=np.intp)
    two_regions[:, 17:] = 1
    payload = {"method": method.value}
    return [
        SegmentationResult(one_region.copy(), {**payload, "region_count": 1}),
        SegmentationResult(one_region.copy(), {**payload, "region_count": 1}),
        SegmentationResult(two_regions.copy(), {**payload, "region_count": 2}),
        SegmentationResult(two_regions.copy(), {**payload, "region_count": 2}),
    ]


def _refiner(method: SegmentationMethod):
    repository_root = Path(__file__).resolve().parents[2]
    loaded = load_pipeline_config(
        repository_root / "configs/reconstruction/pi3_laser_no_loop.yaml",
        (
            f"segmentation.method={method.value}",
            "segmentation.window_reference.enabled=true",
        ),
    )
    return build_window_reference_refiner(loaded.config.segmentation)


def _require_fixture_outcomes(
    merge_results: tuple[SegmentationResult, ...],
    rejection_results: tuple[SegmentationResult, ...],
) -> None:
    merge_diagnostics = [result.diagnostics for result in merge_results]
    if not any(
        int(item.get("window_reference_accepted_edges", 0)) >= 1
        for item in merge_diagnostics
    ):
        raise RuntimeError("synthetic merge fixture did not accept an edge")
    rejection_diagnostics = [result.diagnostics for result in rejection_results]
    if not any(
        item.get("window_reference_fallback") == "insufficient_support"
        for item in rejection_diagnostics
    ):
        raise RuntimeError("synthetic rejection fixture did not produce insufficient_support")
    if not any(
        int(item.get("window_reference_occluded_samples", 0)) > 0
        for item in rejection_diagnostics
    ):
        raise RuntimeError("synthetic rejection fixture did not produce occlusions")
    if not any(
        int(item.get("window_reference_depth_rejected_samples", 0)) > 0
        for item in rejection_diagnostics
    ):
        raise RuntimeError("synthetic rejection fixture did not reject depth")


def build_synthetic_fixture(method: SegmentationMethod) -> SyntheticFixture:
    """Construct and validate the literal finite four-frame fixture."""

    try:
        method = SegmentationMethod(method)
    except (TypeError, ValueError) as exc:
        raise ValueError("synthetic segmentation method is invalid") from exc

    intrinsic = torch.tensor(
        [[16.0, 0.0, 16.0], [0.0, 16.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    base = _point_map(_HEIGHT, _WIDTH, intrinsic)
    point_maps = base.unsqueeze(0).repeat(_FRAME_COUNT, 1, 1, 1)
    camera_poses = torch.eye(4, dtype=torch.float32).repeat(_FRAME_COUNT, 1, 1)
    confidence = torch.ones((_FRAME_COUNT, _HEIGHT, _WIDTH), dtype=torch.float32)

    merge_results = tuple(_refiner(method).refine(
        _results(method),
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        reference_intrinsic=intrinsic,
    ))

    # The rejection case scales complete rays on the right side, preserving
    # their projection while changing their depth.  The anisotropic camera
    # transform plus a small Z offset maps sparse rays onto the same target
    # pixels, creating many-to-one winners and measurable depth rejection.
    rejection_points = point_maps.clone()
    rejection_points[2:, :, 17:, :] *= 2.0
    rejection_poses = camera_poses.clone()
    rejection_poses[2, 0, 0] = 5.0
    rejection_poses[2, 1, 1] = 5.0
    rejection_poses[2, 2, 3] = 0.01
    rejection_results = tuple(_refiner(method).refine(
        _results(method),
        point_maps=rejection_points,
        camera_poses=rejection_poses,
        confidence=confidence,
        reference_intrinsic=intrinsic,
    ))
    _require_fixture_outcomes(merge_results, rejection_results)
    return SyntheticFixture(
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        reference_intrinsic=intrinsic,
        merge_results=merge_results,
        rejection_results=rejection_results,
    )


def _observation(result: SegmentationResult, window_index: int, frame_index: int, *, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {
            "window_index": window_index,
            "frame_index": frame_index,
            "region_count": int(np.unique(result.labels).size),
        }
    payload = dict(result.diagnostics)
    payload.update({"window_index": window_index, "frame_index": frame_index})
    return payload


def _run_fixture_refinement(
    fixture: SyntheticFixture,
    *,
    enabled: bool,
) -> dict[str, object]:
    observations = [
        _observation(result, 0, index, enabled=enabled)
        for index, result in enumerate(fixture.merge_results)
    ] + [
        _observation(result, 1, index, enabled=enabled)
        for index, result in enumerate(fixture.rejection_results)
    ]
    return {
        "stage_timings_ms": {"reconstruction": 0.0},
        "segmentation_summaries": observations,
        "candidate_count": 0,
        "constraint_count": 0,
        "mode_scalars": {"window_count": _WINDOW_COUNT},
    }


def _cache_payload(request: RunRequest, prediction_key: str) -> Path:
    entry = request.cache_root / "v2" / prediction_key
    for directory in (entry.parent, entry, entry / "windows"):
        if directory.is_symlink() or directory.is_file():
            directory.unlink()
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("synthetic cache entry is not campaign-owned")
    for index in range(_WINDOW_COUNT):
        window_path = entry / "windows" / f"{index:06d}.pt"
        if window_path.is_symlink():
            window_path.unlink()
    atomic_json(entry / "manifest.json", {"synthetic": True, "key": prediction_key})
    atomic_json(entry / "sequence.json", {"synthetic": True, "key": prediction_key})
    for index in range(_WINDOW_COUNT):
        (entry / "windows" / f"{index:06d}.pt").write_bytes(b"synthetic-window-v1")
    atomic_json(
        entry / "complete.json",
        {"schema_version": 2, "key": prediction_key, "window_count": _WINDOW_COUNT},
    )
    return entry


def _synthetic_cache_stats(
    request: RunRequest,
    prediction_key: str,
) -> CacheStats:
    if request.cache_mode is PredictionCacheMode.OFF:
        return CacheStats(0, 0, 0, 0.0, 0.0, 0, 0, ())
    complete = cache_entry_complete(
        request.cache_root,
        prediction_key,
        _WINDOW_COUNT,
    )
    if request.cache_mode is PredictionCacheMode.READONLY and not complete:
        raise ValueError("synthetic readonly cache entry is missing")
    if request.cache_mode is PredictionCacheMode.READONLY or (
        request.cache_mode is PredictionCacheMode.AUTO and complete
    ):
        return CacheStats(
            1,
            0,
            0,
            0.0,
            0.0,
            _WINDOW_COUNT,
            0,
            ({"kind": "synthetic", "event": "hit"},),
        )
    entry = _cache_payload(request, prediction_key)
    stored_bytes = sum(path.stat().st_size for path in entry.rglob("*") if path.is_file())
    return CacheStats(
        0,
        1,
        0,
        0.0,
        0.0,
        _WINDOW_COUNT,
        stored_bytes,
        ({"kind": "synthetic", "event": "write"},),
    )


def execute_synthetic(
    request: RunRequest,
    loaded: LoadedCampaignConfig,
) -> PipelineExecution:
    """Execute one synthetic run through the normal artifact/cache boundary."""

    if not isinstance(request, RunRequest):
        raise ValueError("synthetic execution requires RunRequest")
    if not isinstance(loaded, LoadedCampaignConfig):
        raise ValueError("synthetic execution requires LoadedCampaignConfig")
    frame_count = (
        request.identity_seed.frame_stop
        - request.identity_seed.frame_start
        + request.identity_seed.frame_stride
        - 1
    ) // request.identity_seed.frame_stride
    if (
        request.identity_seed.frame_start != 0
        or request.identity_seed.frame_stop != _FRAME_COUNT
        or request.identity_seed.frame_stride != 1
        or frame_count != _FRAME_COUNT
    ):
        raise ValueError("synthetic fixture requires the fixed four-frame identity")
    fixture = build_synthetic_fixture(
        SegmentationMethod(request.planned.variant.segmentation_method)
    )
    diagnostics = _run_fixture_refinement(
        fixture,
        enabled=request.planned.variant.window_reference_enabled,
    )
    prediction_key = hashlib.sha256(
        f"synthetic-v1:{request.staged.manifest_sha256}:75:30:pi3:bfloat16".encode()
    ).hexdigest()
    request.artifact_dir.mkdir(parents=True, exist_ok=False)
    manifest = atomic_json(
        request.artifact_dir / "synthetic.json",
        {
            "schema_version": 1,
            "identity_seed": request.identity_seed.to_payload(),
            "prediction_key": prediction_key,
            "frame_count": _FRAME_COUNT,
            "diagnostics": diagnostics,
        },
    )
    cache_stats = _synthetic_cache_stats(
        request,
        prediction_key,
    )
    return PipelineExecution(
        request.artifact_dir,
        sha256_file(manifest),
        prediction_key,
        _FRAME_COUNT,
        diagnostics,
        cache_stats,
    )


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be SHA256 hex")
    return value


def validate_synthetic_artifact(
    artifact_dir: Path,
    seed: RunIdentitySeed,
) -> tuple[str, str]:
    """Validate one synthetic manifest against the immutable run identity."""

    if not isinstance(seed, RunIdentitySeed):
        raise ValueError("synthetic artifact validation requires RunIdentitySeed")
    manifest_path = Path(artifact_dir) / "synthetic.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("synthetic artifact manifest is missing or invalid") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "identity_seed", "prediction_key", "frame_count", "diagnostics"
    }:
        raise ValueError("synthetic artifact manifest fields are invalid")
    if payload["schema_version"] != 1:
        raise ValueError("synthetic artifact schema version mismatch")
    if payload["identity_seed"] != seed.to_payload():
        raise ValueError("synthetic artifact identity seed does not match")
    prediction_key = _require_sha256(payload["prediction_key"], "synthetic prediction_key")
    expected_frame_count = (
        seed.frame_stop - seed.frame_start + seed.frame_stride - 1
    ) // seed.frame_stride
    if (
        seed.frame_start != 0
        or seed.frame_stop != _FRAME_COUNT
        or seed.frame_stride != 1
        or expected_frame_count != _FRAME_COUNT
    ):
        raise ValueError("synthetic identity must describe the fixed four-frame selection")
    if payload["frame_count"] != expected_frame_count:
        raise ValueError("synthetic artifact frame count does not match identity")
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValueError("synthetic artifact diagnostics are invalid")
    aggregate_diagnostics(
        diagnostics,
        refinement_enabled=seed.window_reference_enabled,
    )
    return prediction_key, sha256_file(manifest_path)


def load_synthetic_execution(
    artifact_dir: Path,
    seed: RunIdentitySeed,
    cache_stats: CacheStats,
) -> PipelineExecution:
    """Reload a retained synthetic artifact for evaluator-only resume."""

    prediction_key, digest = validate_synthetic_artifact(artifact_dir, seed)
    payload = json.loads((Path(artifact_dir) / "synthetic.json").read_text(encoding="utf-8"))
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, Mapping)
    expected_frame_count = (
        seed.frame_stop - seed.frame_start + seed.frame_stride - 1
    ) // seed.frame_stride
    return PipelineExecution(
        Path(artifact_dir),
        digest,
        prediction_key,
        expected_frame_count,
        diagnostics,
        cache_stats,
    )


__all__ = [
    "SyntheticFixture",
    "build_synthetic_fixture",
    "execute_synthetic",
    "load_synthetic_execution",
    "synthetic_checkpoint_sha256",
    "validate_synthetic_artifact",
]
