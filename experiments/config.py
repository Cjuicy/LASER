from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from omegaconf import OmegaConf

from pipeline.config import ReconstructionMode, SegmentationMethod


CANONICAL_RECONSTRUCTION_MODES = (
    ReconstructionMode.NO_LOOP,
    ReconstructionMode.TRADITIONAL,
    ReconstructionMode.CORRECTED,
)


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
            tuple(item.value for item in CANONICAL_RECONSTRUCTION_MODES)
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


@dataclass(frozen=True)
class EvaluationInputConfig:
    kind: EvaluationKind
    config_path: str
    ground_truth_path: str
    ground_truth_format: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvaluationKind):
            try:
                object.__setattr__(self, "kind", EvaluationKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("evaluation input kind is invalid") from exc
        for name in ("config_path", "ground_truth_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"evaluation input {name} must be non-empty")
        if self.kind is EvaluationKind.ATE:
            if (
                not isinstance(self.ground_truth_format, str)
                or not self.ground_truth_format
            ):
                raise ValueError("trajectory ground truth requires format")
        elif self.ground_truth_format is not None:
            raise ValueError("pointcloud ground truth does not accept format")


@dataclass(frozen=True)
class CapabilityExperimentConfig:
    version: int
    reconstruction_config: str
    output_root: str
    dataset_name: str
    sequence: str
    evaluator_inputs: Mapping[EvaluationKind, EvaluationInputConfig]
    segmentation_methods: tuple[str, ...]
    refinement_states: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError("capability experiment version must be 2")
        for name in (
            "reconstruction_config",
            "output_root",
            "dataset_name",
            "sequence",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        expected_segmentation = tuple(item.value for item in SegmentationMethod)
        if tuple(self.segmentation_methods) != expected_segmentation:
            raise ValueError(
                "capability segmentation matrix must be canonical "
                "[depth, geometry, atomic]"
            )
        if tuple(self.refinement_states) != (False, True) or any(
            not isinstance(value, bool) for value in self.refinement_states
        ):
            raise ValueError(
                "capability refinement matrix must be canonical [false, true]"
            )
        try:
            inputs = {
                EvaluationKind(kind): value
                for kind, value in self.evaluator_inputs.items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("evaluator inputs must be a mapping") from exc
        if not inputs:
            raise ValueError("at least one evaluator ground truth must be declared")
        for kind, value in inputs.items():
            if not isinstance(value, EvaluationInputConfig) or value.kind is not kind:
                raise ValueError("evaluator input key and kind must match")
        canonical_inputs = {
            kind: inputs[kind]
            for kind in EvaluationKind
            if kind in inputs
        }
        object.__setattr__(
            self,
            "evaluator_inputs",
            MappingProxyType(canonical_inputs),
        )

    @property
    def entries(self):
        from .matrix import build_capability_matrix

        return build_capability_matrix(self)


def _reject_unknown(
    payload: Mapping[str, object],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} field: " + ", ".join(unknown))


def _load_legacy_experiment(payload: dict[str, object]) -> ExperimentConfig:
    allowed = {
        "version",
        "evaluation",
        "reconstruction_config",
        "evaluation_config",
        "output_root",
        "dataset",
        "matrix",
    }
    _reject_unknown(payload, allowed, context="experiment")
    dataset = payload.get("dataset")
    matrix = payload.get("matrix")
    if not isinstance(dataset, dict) or not isinstance(matrix, dict):
        raise ValueError("experiment dataset and matrix must be mappings")
    _reject_unknown(
        dataset,
        {"name", "sequence", "ground_truth", "ground_truth_format"},
        context="experiment dataset",
    )
    _reject_unknown(
        matrix,
        {"segmentation", "reconstruction_modes"},
        context="experiment matrix",
    )
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


def _required_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _load_capability_experiment(
    payload: dict[str, object],
) -> CapabilityExperimentConfig:
    _reject_unknown(
        payload,
        {
            "version",
            "reconstruction_config",
            "output_root",
            "dataset",
            "evaluation",
            "matrix",
        },
        context="experiment",
    )
    dataset = payload.get("dataset")
    evaluation = payload.get("evaluation")
    matrix = payload.get("matrix")
    if not isinstance(dataset, dict) or not isinstance(matrix, dict):
        raise ValueError("experiment dataset and matrix must be mappings")
    if evaluation is None:
        evaluation = {}
    if not isinstance(evaluation, dict):
        raise ValueError("experiment evaluation must be a mapping")
    _reject_unknown(
        dataset,
        {
            "name",
            "sequence",
            "trajectory_ground_truth",
            "pointcloud_ground_truth",
        },
        context="experiment dataset",
    )
    _reject_unknown(
        evaluation,
        {"trajectory_config", "pointcloud_config"},
        context="experiment evaluation",
    )
    _reject_unknown(
        matrix,
        {"segmentation", "refinement"},
        context="experiment matrix",
    )

    inputs: dict[EvaluationKind, EvaluationInputConfig] = {}
    trajectory_declared = "trajectory_ground_truth" in dataset
    pointcloud_declared = "pointcloud_ground_truth" in dataset
    if not trajectory_declared and not pointcloud_declared:
        raise ValueError("at least one evaluator ground truth must be declared")

    if trajectory_declared:
        ground_truth = dataset["trajectory_ground_truth"]
        if not isinstance(ground_truth, dict):
            raise ValueError("trajectory ground truth must be a mapping")
        _reject_unknown(
            ground_truth,
            {"path", "format"},
            context="trajectory ground truth",
        )
        if "trajectory_config" not in evaluation:
            raise ValueError("trajectory ground truth requires evaluation config")
        inputs[EvaluationKind.ATE] = EvaluationInputConfig(
            kind=EvaluationKind.ATE,
            config_path=_required_string(
                evaluation["trajectory_config"],
                context="trajectory evaluation config",
            ),
            ground_truth_path=_required_string(
                ground_truth.get("path"),
                context="trajectory ground truth path",
            ),
            ground_truth_format=_required_string(
                ground_truth.get("format"),
                context="trajectory ground truth format",
            ),
        )
    elif "trajectory_config" in evaluation:
        raise ValueError(
            "trajectory evaluation config requires declared ground truth"
        )

    if pointcloud_declared:
        ground_truth = dataset["pointcloud_ground_truth"]
        if not isinstance(ground_truth, dict):
            raise ValueError("pointcloud ground truth must be a mapping")
        _reject_unknown(
            ground_truth,
            {"path"},
            context="pointcloud ground truth",
        )
        if "pointcloud_config" not in evaluation:
            raise ValueError("pointcloud ground truth requires evaluation config")
        inputs[EvaluationKind.POINTCLOUD] = EvaluationInputConfig(
            kind=EvaluationKind.POINTCLOUD,
            config_path=_required_string(
                evaluation["pointcloud_config"],
                context="pointcloud evaluation config",
            ),
            ground_truth_path=_required_string(
                ground_truth.get("path"),
                context="pointcloud ground truth path",
            ),
        )
    elif "pointcloud_config" in evaluation:
        raise ValueError(
            "pointcloud evaluation config requires declared ground truth"
        )

    try:
        return CapabilityExperimentConfig(
            version=payload["version"],
            reconstruction_config=_required_string(
                payload.get("reconstruction_config"),
                context="reconstruction_config",
            ),
            output_root=_required_string(
                payload.get("output_root"),
                context="output_root",
            ),
            dataset_name=_required_string(
                dataset.get("name"),
                context="dataset.name",
            ),
            sequence=_required_string(
                dataset.get("sequence"),
                context="dataset.sequence",
            ),
            evaluator_inputs=inputs,
            segmentation_methods=tuple(matrix["segmentation"]),
            refinement_states=tuple(matrix["refinement"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("experiment config is incomplete") from exc


def load_experiment_config(
    path: str | Path,
) -> ExperimentConfig | CapabilityExperimentConfig:
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise ValueError("experiment config is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a mapping")
    version = payload.get("version")
    if version == 1:
        return _load_legacy_experiment(payload)
    if version == 2:
        return _load_capability_experiment(payload)
    raise ValueError("experiment version must be 1 or 2")
