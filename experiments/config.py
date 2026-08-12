from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from omegaconf import OmegaConf

from pipeline.config import ReconstructionMode, SegmentationMethod


class EvaluationKind(str, Enum):
    ATE = "ate"
    POINTCLOUD = "pointcloud"


@dataclass(frozen=True)
class ExperimentDatasetConfig:
    name: str
    sequence: str
    ground_truth: str
    ground_truth_format: str | None = None

    def __post_init__(self) -> None:
        for name in ("name", "sequence", "ground_truth"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"dataset.{name} must be a non-empty string")


@dataclass(frozen=True)
class ExperimentConfig:
    version: int
    evaluation: EvaluationKind
    reconstruction_config: str
    evaluation_config: str
    output_root: str
    dataset: ExperimentDatasetConfig
    segmentation_methods: tuple[str, ...]
    reconstruction_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("experiment version must be 1")
        if not isinstance(self.evaluation, EvaluationKind):
            try:
                object.__setattr__(self, "evaluation", EvaluationKind(self.evaluation))
            except (TypeError, ValueError) as exc:
                raise ValueError("experiment evaluation kind is invalid") from exc
        for name in ("reconstruction_config", "evaluation_config", "output_root"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        expected_segmentation = tuple(item.value for item in SegmentationMethod)
        if tuple(self.segmentation_methods) != expected_segmentation:
            raise ValueError(
                "experiment segmentation matrix must be canonical "
                "[depth, geometry, atomic]"
            )
        expected_modes = (
            tuple(item.value for item in ReconstructionMode)
            if self.evaluation is EvaluationKind.ATE
            else (ReconstructionMode.NO_LOOP.value,)
        )
        if tuple(self.reconstruction_modes) != expected_modes:
            message = (
                "pointcloud matrix only supports no_loop"
                if self.evaluation is EvaluationKind.POINTCLOUD
                else "ATE reconstruction matrix must be canonical"
            )
            raise ValueError(message)

    @property
    def entries(self):
        from .matrix import build_matrix

        return build_matrix(self.evaluation)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise ValueError("experiment config is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a mapping")
    allowed = {
        "version",
        "evaluation",
        "reconstruction_config",
        "evaluation_config",
        "output_root",
        "dataset",
        "matrix",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unknown experiment field: " + ", ".join(unknown))
    dataset = payload.get("dataset")
    matrix = payload.get("matrix")
    if not isinstance(dataset, dict) or not isinstance(matrix, dict):
        raise ValueError("experiment dataset and matrix must be mappings")
    dataset_unknown = sorted(
        set(dataset) - {"name", "sequence", "ground_truth", "ground_truth_format"}
    )
    if dataset_unknown:
        raise ValueError("unknown experiment dataset field: " + ", ".join(dataset_unknown))
    matrix_unknown = sorted(
        set(matrix) - {"segmentation", "reconstruction_modes"}
    )
    if matrix_unknown:
        raise ValueError("unknown experiment matrix field: " + ", ".join(matrix_unknown))
    try:
        return ExperimentConfig(
            version=payload["version"],
            evaluation=EvaluationKind(payload["evaluation"]),
            reconstruction_config=str(payload["reconstruction_config"]),
            evaluation_config=str(payload["evaluation_config"]),
            output_root=str(payload["output_root"]),
            dataset=ExperimentDatasetConfig(**dataset),
            segmentation_methods=tuple(matrix["segmentation"]),
            reconstruction_modes=tuple(matrix["reconstruction_modes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("experiment config is incomplete") from exc
