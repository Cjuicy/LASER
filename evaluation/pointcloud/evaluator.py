from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pipeline.artifacts import PointMapEstimate
from pipeline.config import ReconstructionMode

from .config import PointCloudEvaluationConfig
from .geometry_metrics import (
    GeometryBackend,
    GeometryEvaluation,
    Open3DGeometryBackend,
    evaluate_aligned_point_maps,
)


@dataclass(frozen=True)
class PointCloudGroundTruth:
    point_maps: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.point_maps)
        mask = np.asarray(self.valid_mask)
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("ground-truth point_maps must have shape (N,H,W,3)")
        if mask.shape != points.shape[:-1]:
            raise ValueError("ground-truth valid_mask must match point_maps")
        object.__setattr__(self, "point_maps", points)
        object.__setattr__(self, "valid_mask", mask.astype(bool, copy=False))


def build_open3d_backend() -> Open3DGeometryBackend:
    return Open3DGeometryBackend()


def evaluate_point_maps(
    estimate: PointMapEstimate,
    ground_truth: PointCloudGroundTruth,
    config: PointCloudEvaluationConfig,
    *,
    backend: GeometryBackend | None = None,
    backend_factory: Callable[[], GeometryBackend] = build_open3d_backend,
) -> GeometryEvaluation:
    if not isinstance(estimate, PointMapEstimate):
        raise ValueError("estimate must be PointMapEstimate")
    if estimate.reconstruction_mode is not ReconstructionMode.NO_LOOP:
        raise ValueError("point-cloud evaluation requires no_loop artifact")
    if not isinstance(ground_truth, PointCloudGroundTruth):
        raise ValueError("ground_truth must be PointCloudGroundTruth")
    if not isinstance(config, PointCloudEvaluationConfig):
        raise ValueError("config must be PointCloudEvaluationConfig")
    if estimate.global_points.shape != ground_truth.point_maps.shape:
        raise ValueError("artifact and ground-truth point-map shapes differ")
    selected_backend = backend if backend is not None else backend_factory()
    return evaluate_aligned_point_maps(
        estimate.global_points.detach().cpu().numpy(),
        ground_truth.point_maps,
        ground_truth.valid_mask,
        config,
        selected_backend,
    )
