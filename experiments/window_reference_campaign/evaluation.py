"""Campaign adapters for the existing point-map and trajectory evaluators.

This module owns only the campaign boundary: loading staged inputs, checking
frame/shape alignment, mapping the authoritative evaluator result into the
compact campaign schema, and atomically publishing that schema.  Geometry and
trajectory mathematics remain in :mod:`evaluation.pointcloud` and
:mod:`evaluation.trajectory`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np

if TYPE_CHECKING:
    from .config import DatasetKind, EvaluationKind, LoadedCampaignConfig
    from .runner import EvaluationOutput, PipelineExecution, RunRequest
    from .staging import StagedScene
    from evaluation.pointcloud.config import PointCloudEvaluationConfig
    from evaluation.pointcloud.evaluator import PointCloudGroundTruth
    from evaluation.pointcloud.geometry_metrics import GeometryBackend, GeometryEvaluation
    from evaluation.trajectory.config import TrajectoryEvaluationConfig
    from evaluation.trajectory.evaluator import GroundTruthTrajectory
    from evaluation.trajectory.results import TrajectoryMetrics
    from pipeline.artifacts import PointMapEstimate, TrajectoryEstimate


_POINTCLOUD_PRIMARY_FIELDS = (
    "accuracy_mean_m",
    "accuracy_median_m",
    "completion_mean_m",
    "completion_median_m",
    "normal_consistency_mean",
    "normal_consistency_median",
)
_THRESHOLD_SPECS = (
    (0.01, "1cm"),
    (0.02, "2cm"),
    (0.05, "5cm"),
)
_THRESHOLD_ABS_TOLERANCE = 1e-12


def _campaign_metric_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Results imports transitively construct the prediction-store/Torch stack;
    # keep that import at the publication boundary rather than module import.
    from .results import POINTCLOUD_METRICS, TRAJECTORY_METRICS

    return POINTCLOUD_METRICS, TRAJECTORY_METRICS


@dataclass(frozen=True)
class EvaluationDependencies:
    """Injectable boundary for the authoritative evaluation implementations.

    Production defaults are deliberately not constructed at import time.  A
    caller that wants the normal evaluator stack must call
    :func:`default_evaluation_dependencies`; tests can construct this real
    dataclass directly with their own bounded loader/backend callables.
    """

    load_pointmap_estimate: Callable[[str | Path], "PointMapEstimate"]
    load_trajectory_estimate: Callable[[str | Path], "TrajectoryEstimate"]
    load_ground_truth_trajectory: Callable[
        [str | Path, str], "GroundTruthTrajectory"
    ]
    load_pointcloud_config: Callable[[str | Path], "PointCloudEvaluationConfig"]
    load_trajectory_config: Callable[[str | Path], "TrajectoryEvaluationConfig"]
    pointcloud_backend_factory: Callable[[], "GeometryBackend"]
    evaluate_point_maps: Callable[..., "GeometryEvaluation"]
    evaluate_trajectory: Callable[
        ["TrajectoryEstimate", "GroundTruthTrajectory", "TrajectoryEvaluationConfig"],
        "TrajectoryMetrics",
    ]

    def __post_init__(self) -> None:
        for name in (
            "load_pointmap_estimate",
            "load_trajectory_estimate",
            "load_ground_truth_trajectory",
            "load_pointcloud_config",
            "load_trajectory_config",
            "pointcloud_backend_factory",
            "evaluate_point_maps",
            "evaluate_trajectory",
        ):
            if not callable(getattr(self, name)):
                raise ValueError(f"evaluation dependency {name} must be callable")


def default_evaluation_dependencies() -> EvaluationDependencies:
    """Build the normal evaluator stack while keeping heavy imports lazy."""

    from evaluation.pointcloud import (
        build_open3d_backend,
        evaluate_point_maps,
        load_pointcloud_evaluation_config,
    )
    from evaluation.trajectory import (
        evaluate_trajectory,
        load_ground_truth_trajectory,
        load_trajectory_evaluation_config,
    )
    from pipeline.artifacts import load_pointmap_estimate, load_trajectory_estimate

    return EvaluationDependencies(
        load_pointmap_estimate=load_pointmap_estimate,
        load_trajectory_estimate=load_trajectory_estimate,
        load_ground_truth_trajectory=load_ground_truth_trajectory,
        load_pointcloud_config=load_pointcloud_evaluation_config,
        load_trajectory_config=load_trajectory_evaluation_config,
        pointcloud_backend_factory=build_open3d_backend,
        evaluate_point_maps=evaluate_point_maps,
        evaluate_trajectory=evaluate_trajectory,
    )


def _staged_evaluation_kind(staged: "StagedScene") -> "EvaluationKind":
    """Read the typed staging contract without inferring from payload paths."""

    from .config import EvaluationKind

    try:
        value = staged.evaluation_kind
    except AttributeError as exc:
        raise ValueError("staged evaluation_kind is required") from exc
    if not isinstance(value, EvaluationKind):
        raise ValueError("staged evaluation_kind is invalid")
    return value


def _source_frame_ids(staged: "StagedScene") -> tuple[int, ...]:
    values = getattr(staged, "source_frame_ids", None)
    if not isinstance(values, tuple) or not values:
        raise ValueError("staged source frame IDs are invalid")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("staged source frame IDs are invalid")
    if len(set(values)) != len(values):
        raise ValueError("staged source frame IDs are invalid")
    selection = getattr(staged, "selection", None)
    if selection is not None and tuple(getattr(selection, "source_frame_ids", ())) != values:
        raise ValueError("staged source frame IDs do not match selection")
    return values


def _archive_frame_ids(raw: object, expected: tuple[int, ...]) -> tuple[int, ...]:
    array = np.asarray(raw)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("point-cloud GT frame IDs are invalid")
    values = tuple(int(value) for value in array.tolist())
    if values != expected:
        raise ValueError("point-cloud GT frame IDs do not match staging")
    return values


def _dense_frame_ids(value: object, expected_count: int, label: str) -> tuple[int, ...]:
    values = getattr(value, "frame_ids", None)
    expected = tuple(range(expected_count))
    if values != expected:
        raise ValueError(f"{label} frame IDs do not match staged dense ordinals")
    return values


def _require_exact_finite_metrics(
    metrics: Mapping[str, object],
    expected_names: tuple[str, ...],
) -> None:
    if not isinstance(metrics, Mapping) or set(metrics) != set(expected_names):
        raise ValueError("evaluation metrics do not match the compact schema")
    for name in expected_names:
        value = metrics[name]
        if isinstance(value, bool) or type(value) not in (int, float):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")


def _validate_pointcloud_metrics(metrics: Mapping[str, object]) -> None:
    pointcloud_names, _ = _campaign_metric_names()
    _require_exact_finite_metrics(metrics, pointcloud_names)
    for name in pointcloud_names:
        value = float(metrics[name])
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        bounded = ("precision_", "recall_", "fscore_", "normal_consistency_")
        if name.startswith(bounded) and value > 1:
            raise ValueError(f"{name} must be in [0, 1]")


def _validate_trajectory_metrics(metrics: Mapping[str, object]) -> None:
    _, trajectory_names = _campaign_metric_names()
    _require_exact_finite_metrics(metrics, trajectory_names)
    for name in trajectory_names[:3]:
        if type(metrics[name]) is not float:
            raise ValueError(f"{name} must be a float")
        if metrics[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    count = metrics["internal_matched_frame_count"]
    if type(count) is not int or count < 2:
        raise ValueError("internal_matched_frame_count must be an integer >= 2")


def _pointcloud_metrics(evaluation: "GeometryEvaluation") -> dict[str, float]:
    """Map the existing geometry result into the exact campaign schema."""

    try:
        primary = evaluation.primary
        diagnostics = evaluation.diagnostics
        metrics: dict[str, float] = {
            name: float(getattr(primary, name))
            for name in _POINTCLOUD_PRIMARY_FIELDS
        }
        metrics["chamfer_l1_m"] = float(diagnostics.chamfer_l1_m)
        thresholds = tuple(diagnostics.thresholds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("point-cloud evaluator result is invalid") from exc

    matched: dict[str, object] = {}
    for raw in thresholds:
        try:
            threshold = float(raw.threshold_m)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("point-cloud evaluator thresholds are invalid") from exc
        matches = tuple(
            suffix
            for expected, suffix in _THRESHOLD_SPECS
            if math.isclose(
                threshold,
                expected,
                rel_tol=0.0,
                abs_tol=_THRESHOLD_ABS_TOLERANCE,
            )
        )
        if len(matches) != 1:
            raise ValueError("point-cloud evaluator thresholds are invalid")
        suffix = matches[0]
        if suffix in matched:
            raise ValueError("point-cloud evaluator thresholds contain duplicates")
        try:
            matched[suffix] = {
                f"precision_{suffix}": float(raw.precision),
                f"recall_{suffix}": float(raw.recall),
                f"fscore_{suffix}": float(raw.fscore),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("point-cloud evaluator threshold result is invalid") from exc

    if set(matched) != {suffix for _, suffix in _THRESHOLD_SPECS}:
        raise ValueError("point-cloud evaluator thresholds are incomplete")
    for suffix, values in matched.items():
        assert isinstance(values, dict)
        metrics.update(values)
    _validate_pointcloud_metrics(metrics)
    return metrics


def evaluate_pointcloud_artifact(
    artifact_dir: str | Path,
    staged: "StagedScene",
    evaluation_config: str | Path,
    output_dir: str | Path,
    *,
    dependencies: EvaluationDependencies | None = None,
) -> "EvaluationOutput":
    """Evaluate one staged point-map artifact with the existing evaluator."""

    from .config import EvaluationKind
    from .results import atomic_json

    deps = dependencies or default_evaluation_dependencies()
    if _staged_evaluation_kind(staged) is not EvaluationKind.POINTCLOUD:
        raise ValueError("point-cloud adapter requires pointcloud staged scene")
    source_ids = _source_frame_ids(staged)
    ground_truth_path = getattr(staged, "pointcloud_gt_path", None)
    if ground_truth_path is None:
        raise ValueError("point-cloud staged scene has no GT")

    estimate = deps.load_pointmap_estimate(artifact_dir)
    # Reconstruction artifacts are indexed by their dense staged ordinal;
    # source IDs remain in the prepared GT archive and campaign identity.
    _dense_frame_ids(estimate, len(source_ids), "point-map artifact")
    estimate_shape = tuple(int(value) for value in estimate.global_points.shape)

    try:
        archive = np.load(ground_truth_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("point-cloud GT archive is invalid") from exc
    with archive:
        required = {"point_maps", "valid_mask", "frame_ids"}
        if not required.issubset(set(archive.files)):
            raise ValueError("point-cloud GT archive is incomplete")
        frame_ids = _archive_frame_ids(archive["frame_ids"], source_ids)
        del frame_ids
        point_maps = np.asarray(archive["point_maps"])
        valid_mask = np.asarray(archive["valid_mask"])
        if point_maps.ndim != 4 or point_maps.shape[-1] != 3:
            raise ValueError("point-cloud GT point_maps have invalid shape")
        if point_maps.shape[0] != len(source_ids):
            raise ValueError("point-cloud GT frame count does not match staging")
        if valid_mask.shape != point_maps.shape[:-1]:
            raise ValueError("point-cloud GT valid_mask shape does not match point_maps")
        if tuple(point_maps.shape) != estimate_shape:
            raise ValueError("artifact and ground-truth point-map shapes differ")
        # Import the existing truth container only at the evaluator boundary.
        from evaluation.pointcloud import PointCloudGroundTruth

        truth = PointCloudGroundTruth(point_maps, valid_mask)

    config = deps.load_pointcloud_config(evaluation_config)
    evaluation = deps.evaluate_point_maps(
        estimate,
        truth,
        config,
        backend_factory=deps.pointcloud_backend_factory,
    )
    metrics = _pointcloud_metrics(evaluation)
    target = atomic_json(Path(output_dir) / "pointcloud_metrics.json", metrics)
    return _evaluation_output(EvaluationKind.POINTCLOUD, metrics, (target,))


def evaluate_kitti_artifact(
    artifact_dir: str | Path,
    staged: "StagedScene",
    evaluation_config: str | Path,
    output_dir: str | Path,
    *,
    dependencies: EvaluationDependencies | None = None,
) -> "EvaluationOutput":
    """Evaluate KITTI rows with the internal Sim(3) trajectory evaluator."""

    from .config import EvaluationKind
    from .results import atomic_json

    deps = dependencies or default_evaluation_dependencies()
    if _staged_evaluation_kind(staged) is not EvaluationKind.INTERNAL_TRAJECTORY:
        raise ValueError("KITTI adapter requires internal trajectory staged scene")
    source_ids = _source_frame_ids(staged)
    poses_path = getattr(staged, "poses_path", None)
    if poses_path is None:
        raise ValueError("KITTI staged scene has no poses")

    estimate = deps.load_trajectory_estimate(artifact_dir)
    _dense_frame_ids(estimate, len(source_ids), "trajectory artifact")
    truth = deps.load_ground_truth_trajectory(poses_path, "replica")
    if truth.frame_ids != tuple(range(len(source_ids))):
        raise ValueError("staged KITTI pose rows are not dense ordinals")
    if tuple(int(value) for value in truth.camera_poses.shape) != (
        len(source_ids),
        4,
        4,
    ):
        raise ValueError("staged KITTI pose shape does not match staging")

    metrics_result = deps.evaluate_trajectory(
        estimate,
        truth,
        deps.load_trajectory_config(evaluation_config),
    )
    payload: dict[str, object] = {
        "internal_ate_rmse_m": float(metrics_result.ate_rmse_m),
        "internal_rpe_translation_rmse_m": float(metrics_result.rpe_translation_rmse_m),
        "internal_rpe_rotation_rmse_deg": float(metrics_result.rpe_rotation_rmse_deg),
        "internal_matched_frame_count": metrics_result.matched_frame_count,
    }
    if payload["internal_matched_frame_count"] != len(source_ids):
        raise ValueError(
            "trajectory evaluator matched frame count does not match staging"
        )
    _validate_trajectory_metrics(payload)
    target = atomic_json(Path(output_dir) / "internal_trajectory_metrics.json", payload)
    return _evaluation_output(EvaluationKind.INTERNAL_TRAJECTORY, payload, (target,))


def _evaluation_output(
    kind: "EvaluationKind",
    metrics: Mapping[str, int | float] | None,
    output_paths: tuple[Path, ...],
) -> "EvaluationOutput":
    # Keep the Task 4 type import lazy along with the evaluator stack.
    from .runner import EvaluationOutput

    return EvaluationOutput(kind, metrics, output_paths)


def evaluate_artifact(
    request: "RunRequest",
    execution: "PipelineExecution",
    loaded: "LoadedCampaignConfig",
) -> "EvaluationOutput":
    """Task 4 default dependency: dispatch the three campaign kinds."""

    from .config import EvaluationKind

    kind = _staged_evaluation_kind(request.staged)
    if kind is EvaluationKind.NONE:
        return _evaluation_output(EvaluationKind.NONE, None, ())
    output_dir = Path(request.attempt_dir) / "evaluation"
    artifact_dir = execution.artifact_dir
    if kind is EvaluationKind.POINTCLOUD:
        return evaluate_pointcloud_artifact(
            artifact_dir,
            request.staged,
            loaded.config.evaluation.pointcloud_config,
            output_dir,
        )
    if kind is EvaluationKind.INTERNAL_TRAJECTORY:
        return evaluate_kitti_artifact(
            artifact_dir,
            request.staged,
            loaded.config.evaluation.trajectory_config,
            output_dir,
        )
    raise ValueError(f"unsupported campaign evaluation kind: {kind}")


__all__ = [
    "EvaluationDependencies",
    "default_evaluation_dependencies",
    "evaluate_artifact",
    "evaluate_kitti_artifact",
    "evaluate_pointcloud_artifact",
]
