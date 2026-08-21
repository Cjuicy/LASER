from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from omegaconf import OmegaConf

try:
    # In normal runtime these are the pipeline's exact enum classes.  The
    # package-level pipeline exports currently import torch, so retain a
    # value-compatible light fallback for the import-free plan subprocess.
    if os.environ.get("LASER_WINDOW_REFERENCE_IMPORT_LIGHT") == "1":
        raise ImportError("light campaign import requested")
    from pipeline.config import (
        AtomicSplitMode,
        ModelName,
        PredictionCacheMode,
        ReconstructionMode,
        SegmentationMethod,
    )
except (ImportError, RuntimeError):  # pragma: no cover - torch-blocked plan
    class SegmentationMethod(str, Enum):
        DEPTH = "depth"
        GEOMETRY = "geometry"
        ATOMIC = "atomic"

    class AtomicSplitMode(str, Enum):
        NONE = "none"
        CONSERVATIVE = "conservative"
        NORMAL_ONLY = "normal_only"

    class ReconstructionMode(str, Enum):
        NO_LOOP = "no_loop"
        TRADITIONAL = "traditional"
        CORRECTED = "corrected"

    class ModelName(str, Enum):
        PI3 = "pi3"

    class PredictionCacheMode(str, Enum):
        AUTO = "auto"
        REFRESH = "refresh"
        READONLY = "readonly"
        OFF = "off"


class DatasetKind(str, Enum):
    SYNTHETIC = "synthetic"
    SEVEN_SCENES = "7scenes"
    NRGBD = "nrgbd"
    KITTI = "kitti"


class EvaluationKind(str, Enum):
    NONE = "none"
    POINTCLOUD = "pointcloud"
    INTERNAL_TRAJECTORY = "internal_trajectory"


class CachePolicy(str, Enum):
    AUTO = "auto"
    REFRESH = "refresh"
    READONLY = "readonly"
    OFF = "off"


class FailurePolicy(str, Enum):
    FAIL_FAST = "fail-fast"
    KEEP_GOING = "keep-going"


_ROOT_FIELDS = {
    "version",
    "campaign",
    "pipeline_config",
    "matrix",
    "window_reference",
    "runtime",
    "storage",
    "evaluation",
    "scenes",
    "presets",
}
_CAMPAIGN_FIELDS = {"id"}
_MATRIX_FIELDS = {
    "methods",
    "refinement",
    "reconstruction_mode",
    "window_size",
    "overlap",
    "atomic_split_mode",
}
_WINDOW_REFERENCE_FIELDS = {
    "sampling_stride",
    "max_keyframes",
    "relative_depth_tolerance",
    "min_reference_score",
    "stop_coverage_ratio",
    "min_coverage_gain",
    "min_region_correspondences",
    "min_region_coverage",
    "min_region_purity",
    "merge_vote_threshold",
}
_RUNTIME_FIELDS = {
    "model_name",
    "model_dtype",
    "inference_device",
    "process_device",
    "jobs",
    "gpu",
    "cache_policy",
    "failure_policy",
}
_STORAGE_FIELDS = {
    "data_root",
    "checkpoint",
    "output_root",
    "minimum_free_gb",
    "keep_artifacts",
}
_EVALUATION_FIELDS = {"pointcloud_config", "trajectory_config"}
_SCENE_FIELDS = {
    "dataset",
    "scene",
    "source_root",
    "frame_index_map",
    "prepared_gt",
    "expected_gt_shape",
}
_PRESET_FIELDS = {"scenes"}
_PRESET_SCENE_FIELDS = {"scene_id", "start", "stop", "stride"}

_CANONICAL_METHODS = (
    SegmentationMethod.DEPTH,
    SegmentationMethod.GEOMETRY,
    SegmentationMethod.ATOMIC,
)
_CANONICAL_REFINEMENTS = (False, True)
_WINDOW_REFERENCE_PAYLOAD: dict[str, int | float] = {
    "sampling_stride": 4,
    "max_keyframes": 4,
    "relative_depth_tolerance": 0.05,
    "min_reference_score": 0.30,
    "stop_coverage_ratio": 0.90,
    "min_coverage_gain": 0.03,
    "min_region_correspondences": 8,
    "min_region_coverage": 0.10,
    "min_region_purity": 0.80,
    "merge_vote_threshold": 0.80,
}


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Encode a fully resolved payload in the campaign's stable form."""

    def default(item: object) -> str:
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"not JSON serializable: {type(item).__name__}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    ).encode("utf-8")


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_int(value: object, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return value


def _require_float(value: object, path: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return result


def _require_mapping(value: object, path: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a sequence")
    return list(value)


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, path)


def _enum_value(value: object, enum_type: type[Enum], path: str) -> Enum:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be one of the enum values")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(str(member.value) for member in enum_type)
        raise ValueError(f"{path} must be one of: {allowed}") from exc


def _reject_unknown(mapping: Mapping[object, object], allowed: set[str], path: str) -> None:
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"unknown {path} field: {names}")


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _repository_root() -> Path:
    # The checked-in campaign manifest is always part of this repository.  A
    # static root also keeps plan mode independent of data/weight existence.
    return Path(__file__).resolve().parents[2]


def _raw_value(mapping: Mapping[object, object], key: str, path: str) -> object:
    if key not in mapping:
        raise ValueError(f"missing {path} field: {key}")
    return mapping[key]


@dataclass(frozen=True)
class WindowReferenceValues:
    sampling_stride: int
    max_keyframes: int
    relative_depth_tolerance: float
    min_reference_score: float
    stop_coverage_ratio: float
    min_coverage_gain: float
    min_region_correspondences: int
    min_region_coverage: float
    min_region_purity: float
    merge_vote_threshold: float

    def __post_init__(self) -> None:
        for name in ("sampling_stride", "max_keyframes", "min_region_correspondences"):
            _require_int(getattr(self, name), f"window_reference.{name}", minimum=1)
        _require_float(
            self.relative_depth_tolerance,
            "window_reference.relative_depth_tolerance",
            minimum=0.0,
        )
        if self.relative_depth_tolerance <= 0:
            raise ValueError("window_reference.relative_depth_tolerance must be positive")
        for name in (
            "min_reference_score",
            "stop_coverage_ratio",
            "min_region_coverage",
            "min_region_purity",
            "merge_vote_threshold",
        ):
            value = _require_float(getattr(self, name), f"window_reference.{name}")
            if not 0.0 < value <= 1.0:
                raise ValueError(f"window_reference.{name} must be in (0, 1]")
        gain = _require_float(self.min_coverage_gain, "window_reference.min_coverage_gain")
        if not 0.0 <= gain <= 1.0:
            raise ValueError("window_reference.min_coverage_gain must be in [0, 1]")
        for name, expected in _WINDOW_REFERENCE_PAYLOAD.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"window_reference.{name} must use the approved campaign value"
                )

    def to_payload(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SceneConfig:
    scene_id: str
    dataset: DatasetKind
    scene: str
    source_root: str | None
    frame_index_map: str | None
    prepared_gt: str | None
    expected_gt_shape: tuple[int, int] | None

    def __post_init__(self) -> None:
        _require_string(self.scene_id, "scene_id")
        if not isinstance(self.dataset, DatasetKind):
            raise ValueError("scene.dataset is invalid")
        _require_string(self.scene, "scene.scene")
        for name in ("source_root", "frame_index_map", "prepared_gt"):
            value = getattr(self, name)
            if value is not None:
                _require_string(value, f"scene.{name}")
        shape = self.expected_gt_shape
        if shape is not None:
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(type(item) is not int or item < 1 for item in shape)
            ):
                raise ValueError("scene.expected_gt_shape must be two positive integers")

    def to_payload(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "dataset": self.dataset.value,
            "scene": self.scene,
            "source_root": self.source_root,
            "frame_index_map": self.frame_index_map,
            "prepared_gt": self.prepared_gt,
            "expected_gt_shape": list(self.expected_gt_shape)
            if self.expected_gt_shape is not None
            else None,
        }


@dataclass(frozen=True)
class PresetSceneConfig:
    scene_id: str
    start: int
    stop: int | None
    stride: int

    def __post_init__(self) -> None:
        _require_string(self.scene_id, "preset scene_id")
        _require_int(self.start, "preset.start", minimum=0)
        if self.stop is not None:
            _require_int(self.stop, "preset.stop", minimum=0)
            if self.stop <= self.start:
                raise ValueError("preset.stop must be greater than preset.start")
        _require_int(self.stride, "preset.stride", minimum=1)

    def to_payload(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "start": self.start,
            "stop": self.stop,
            "stride": self.stride,
        }


@dataclass(frozen=True)
class PresetConfig:
    scenes: tuple[PresetSceneConfig, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scenes, tuple) or not self.scenes:
            raise ValueError("preset.scenes must be a non-empty tuple")
        if any(not isinstance(item, PresetSceneConfig) for item in self.scenes):
            raise ValueError("preset.scenes contains an invalid scene")
        scene_ids = [item.scene_id for item in self.scenes]
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("preset scene IDs must be unique")

    def to_payload(self) -> dict[str, object]:
        return {"scenes": [item.to_payload() for item in self.scenes]}


@dataclass(frozen=True)
class MatrixConfig:
    methods: tuple[SegmentationMethod, ...]
    refinement: tuple[bool, ...]
    reconstruction_mode: ReconstructionMode
    window_size: int
    overlap: int
    atomic_split_mode: AtomicSplitMode

    def __post_init__(self) -> None:
        if not isinstance(self.methods, tuple) or not self.methods:
            raise ValueError("matrix.methods must be a non-empty tuple")
        if any(not isinstance(item, SegmentationMethod) for item in self.methods):
            raise ValueError("matrix.methods contains an unsupported method")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("matrix methods must be unique")
        if any(item not in _CANONICAL_METHODS for item in self.methods):
            raise ValueError("matrix method is not supported")
        if not isinstance(self.refinement, tuple) or not self.refinement:
            raise ValueError("matrix.refinement must be a non-empty tuple")
        if any(type(item) is not bool for item in self.refinement):
            raise ValueError("matrix.refinement values must be booleans")
        if len(set(self.refinement)) != len(self.refinement):
            raise ValueError("matrix refinement values must be unique")
        if self.reconstruction_mode is not ReconstructionMode.NO_LOOP:
            raise ValueError("matrix.reconstruction_mode must be no_loop")
        if self.window_size != 75 or self.overlap != 30:
            raise ValueError("matrix window must be 75/30")
        if self.atomic_split_mode is not AtomicSplitMode.CONSERVATIVE:
            raise ValueError("matrix.atomic_split_mode must be conservative")

    def to_payload(self) -> dict[str, object]:
        return {
            "methods": [item.value for item in self.methods],
            "refinement": list(self.refinement),
            "reconstruction_mode": self.reconstruction_mode.value,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "atomic_split_mode": self.atomic_split_mode.value,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    model_name: ModelName
    model_dtype: str
    inference_device: str
    process_device: str
    jobs: int
    gpu: int
    cache_policy: CachePolicy
    failure_policy: FailurePolicy

    def __post_init__(self) -> None:
        if self.model_name is not ModelName.PI3:
            raise ValueError("runtime.model_name must be pi3")
        if self.model_dtype != "bfloat16":
            raise ValueError("runtime.model_dtype must be bfloat16")
        if self.inference_device != "cuda":
            raise ValueError("runtime.inference_device must be cuda")
        if self.process_device != "cpu":
            raise ValueError("runtime.process_device must be cpu")
        _require_int(self.jobs, "runtime.jobs", minimum=1)
        _require_int(self.gpu, "runtime.gpu", minimum=0)
        if not isinstance(self.cache_policy, CachePolicy):
            raise ValueError("runtime.cache_policy is invalid")
        if self.failure_policy is not FailurePolicy.KEEP_GOING:
            raise ValueError("runtime.failure_policy must be keep-going")

    def to_payload(self) -> dict[str, object]:
        return {
            "model_name": self.model_name.value,
            "model_dtype": self.model_dtype,
            "inference_device": self.inference_device,
            "process_device": self.process_device,
            "jobs": self.jobs,
            "gpu": self.gpu,
            "cache_policy": self.cache_policy.value,
            "failure_policy": self.failure_policy.value,
        }


@dataclass(frozen=True)
class StorageConfig:
    data_root: Path
    checkpoint: Path
    output_root: Path
    minimum_free_gb: float
    keep_artifacts: bool

    def __post_init__(self) -> None:
        for name in ("data_root", "checkpoint", "output_root"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"storage.{name} must be a path")
        _require_float(self.minimum_free_gb, "storage.minimum_free_gb", minimum=0.0)
        _require_bool(self.keep_artifacts, "storage.keep_artifacts")

    def to_payload(self) -> dict[str, object]:
        return {
            "data_root": str(self.data_root),
            "checkpoint": str(self.checkpoint),
            "output_root": str(self.output_root),
            "minimum_free_gb": self.minimum_free_gb,
            "keep_artifacts": self.keep_artifacts,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    pointcloud_config: Path
    trajectory_config: Path

    def __post_init__(self) -> None:
        if not isinstance(self.pointcloud_config, Path):
            raise ValueError("evaluation.pointcloud_config must be a path")
        if not isinstance(self.trajectory_config, Path):
            raise ValueError("evaluation.trajectory_config must be a path")

    def to_payload(self) -> dict[str, object]:
        return {
            "pointcloud_config": str(self.pointcloud_config),
            "trajectory_config": str(self.trajectory_config),
        }


@dataclass(frozen=True)
class CampaignConfig:
    version: int
    campaign_id: str
    repository_root: Path
    pipeline_config: Path
    matrix: MatrixConfig
    window_reference: WindowReferenceValues
    runtime: RuntimeConfig
    storage: StorageConfig
    evaluation: EvaluationConfig
    scenes: Mapping[str, SceneConfig]
    presets: Mapping[str, PresetConfig]
    selected_preset: str
    selected_scenes: tuple[PresetSceneConfig, ...]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("campaign version must be 1")
        _require_string(self.campaign_id, "campaign.id")
        if not isinstance(self.repository_root, Path):
            raise ValueError("campaign.repository_root must be a path")
        if not isinstance(self.pipeline_config, Path):
            raise ValueError("pipeline_config must be a path")
        if not isinstance(self.scenes, Mapping) or not isinstance(self.presets, Mapping):
            raise ValueError("campaign scenes and presets must be mappings")
        if self.selected_preset not in self.presets:
            raise ValueError("selected preset is unknown")
        if not isinstance(self.selected_scenes, tuple):
            raise ValueError("selected_scenes must be a tuple")

    @property
    def campaign_root(self) -> Path:
        return self.storage.output_root / self.campaign_id


@dataclass(frozen=True)
class CampaignOverrides:
    preset: str
    scene_ids: tuple[str, ...] = ()
    methods: tuple[SegmentationMethod, ...] = ()
    refinements: tuple[bool, ...] = ()
    data_root: Path | None = None
    output_root: Path | None = None
    checkpoint: Path | None = None
    start_frame: int | None = None
    max_frames: int | None = None
    frame_stride: int | None = None
    gpu: int | None = None
    cache_policy: CachePolicy | None = None
    keep_artifacts: bool | None = None


@dataclass(frozen=True)
class LoadedCampaignConfig:
    config: CampaignConfig
    resolved_payload: Mapping[str, object]
    sha256: str


def _parse_window_reference(payload: Mapping[object, object]) -> WindowReferenceValues:
    _reject_unknown(payload, _WINDOW_REFERENCE_FIELDS, "window_reference")
    values: dict[str, int | float] = {}
    for name in _WINDOW_REFERENCE_FIELDS:
        value = _raw_value(payload, name, "window_reference")
        if name in {"sampling_stride", "max_keyframes", "min_region_correspondences"}:
            values[name] = _require_int(value, f"window_reference.{name}", minimum=1)
        else:
            values[name] = _require_float(value, f"window_reference.{name}")
    return WindowReferenceValues(**values)


def _parse_scene(
    scene_id: str,
    payload: Mapping[object, object],
    *,
    repository_root: Path,
    data_root: Path,
) -> SceneConfig:
    _reject_unknown(payload, _SCENE_FIELDS, f"scene {scene_id}")
    dataset = _enum_value(
        _raw_value(payload, "dataset", f"scene {scene_id}"),
        DatasetKind,
        f"scene {scene_id}.dataset",
    )
    scene = _require_string(
        _raw_value(payload, "scene", f"scene {scene_id}"),
        f"scene {scene_id}.scene",
    )
    source_root = _optional_string(
        _raw_value(payload, "source_root", f"scene {scene_id}"),
        f"scene {scene_id}.source_root",
    )
    frame_index_map = _optional_string(
        _raw_value(payload, "frame_index_map", f"scene {scene_id}"),
        f"scene {scene_id}.frame_index_map",
    )
    prepared_gt = _optional_string(
        _raw_value(payload, "prepared_gt", f"scene {scene_id}"),
        f"scene {scene_id}.prepared_gt",
    )
    expected_raw = _raw_value(payload, "expected_gt_shape", f"scene {scene_id}")
    expected: tuple[int, int] | None
    if expected_raw is None:
        expected = None
    else:
        dimensions = _require_sequence(expected_raw, f"scene {scene_id}.expected_gt_shape")
        if len(dimensions) != 2:
            raise ValueError(f"scene {scene_id}.expected_gt_shape must have two values")
        expected = (
            _require_int(dimensions[0], f"scene {scene_id}.expected_gt_shape[0]", minimum=1),
            _require_int(dimensions[1], f"scene {scene_id}.expected_gt_shape[1]", minimum=1),
        )
    return SceneConfig(
        scene_id=scene_id,
        dataset=dataset,
        scene=scene,
        source_root=(str(_resolve_path(source_root, data_root)) if source_root else None),
        frame_index_map=(
            str(_resolve_path(frame_index_map, repository_root))
            if frame_index_map
            else None
        ),
        prepared_gt=(str(_resolve_path(prepared_gt, data_root)) if prepared_gt else None),
        expected_gt_shape=expected,
    )


def _parse_preset_scene(payload: Mapping[object, object], path: str) -> PresetSceneConfig:
    _reject_unknown(payload, _PRESET_SCENE_FIELDS, path)
    scene_id = _require_string(_raw_value(payload, "scene_id", path), f"{path}.scene_id")
    start = _require_int(_raw_value(payload, "start", path), f"{path}.start", minimum=0)
    stop_raw = _raw_value(payload, "stop", path)
    stop = None if stop_raw is None else _require_int(stop_raw, f"{path}.stop", minimum=0)
    stride = _require_int(_raw_value(payload, "stride", path), f"{path}.stride", minimum=1)
    return PresetSceneConfig(scene_id, start, stop, stride)


def _normalize_methods(values: object, path: str) -> tuple[SegmentationMethod, ...]:
    raw_values = _require_sequence(values, path)
    parsed_items: list[SegmentationMethod] = []
    for index, item in enumerate(raw_values):
        if isinstance(item, SegmentationMethod):
            parsed_items.append(item)
        else:
            parsed_items.append(
                _enum_value(item, SegmentationMethod, f"{path}[{index}]")
            )
    parsed = tuple(parsed_items)
    if len(set(parsed)) != len(parsed):
        raise ValueError("matrix methods must be unique")
    if any(item not in _CANONICAL_METHODS for item in parsed):
        raise ValueError("matrix method is not supported")
    return tuple(item for item in _CANONICAL_METHODS if item in parsed)


def _normalize_refinements(values: object, path: str) -> tuple[bool, ...]:
    raw_values = _require_sequence(values, path)
    result = tuple(_require_bool(item, path) for item in raw_values)
    if not result:
        raise ValueError(f"{path} must be a non-empty sequence")
    if len(set(result)) != len(result):
        raise ValueError("matrix refinement values must be unique")
    return tuple(item for item in _CANONICAL_REFINEMENTS if item in result)


def _campaign_payload(config: CampaignConfig) -> dict[str, object]:
    return {
        "version": config.version,
        "campaign_id": config.campaign_id,
        "repository_root": str(config.repository_root),
        "pipeline_config": str(config.pipeline_config),
        "matrix": config.matrix.to_payload(),
        "window_reference": config.window_reference.to_payload(),
        "runtime": config.runtime.to_payload(),
        "storage": config.storage.to_payload(),
        "evaluation": config.evaluation.to_payload(),
        "scenes": {
            scene_id: scene.to_payload()
            for scene_id, scene in sorted(config.scenes.items())
        },
        "presets": {
            preset: value.to_payload()
            for preset, value in sorted(config.presets.items())
        },
        "selected_preset": config.selected_preset,
        "selected_scenes": [item.to_payload() for item in config.selected_scenes],
    }


def _construct_and_validate(
    payload: Mapping[object, object],
    overrides: CampaignOverrides,
) -> CampaignConfig:
    repository_root = _repository_root()
    version = _require_int(_raw_value(payload, "version", "campaign"), "version", minimum=1)
    campaign_payload = _require_mapping(_raw_value(payload, "campaign", "campaign"), "campaign")
    _reject_unknown(campaign_payload, _CAMPAIGN_FIELDS, "campaign")
    campaign_id = _require_string(
        _raw_value(campaign_payload, "id", "campaign"), "campaign.id"
    )
    pipeline_config = _resolve_path(
        _require_string(_raw_value(payload, "pipeline_config", "campaign"), "pipeline_config"),
        repository_root,
    )

    matrix_payload = _require_mapping(_raw_value(payload, "matrix", "campaign"), "matrix")
    _reject_unknown(matrix_payload, _MATRIX_FIELDS, "matrix")
    methods = _normalize_methods(_raw_value(matrix_payload, "methods", "matrix"), "matrix.methods")
    if methods != _CANONICAL_METHODS:
        raise ValueError("matrix.methods must be the canonical depth/geometry/atomic order")
    if overrides.methods:
        methods = _normalize_methods(overrides.methods, "matrix.methods")
    refinement_values = _normalize_refinements(
        _raw_value(matrix_payload, "refinement", "matrix"),
        "matrix.refinement",
    )
    if refinement_values != _CANONICAL_REFINEMENTS:
        raise ValueError("matrix.refinement must be the canonical off/on order")
    if overrides.refinements:
        refinement_values = _normalize_refinements(overrides.refinements, "matrix.refinement")
    matrix = MatrixConfig(
        methods=methods,
        refinement=refinement_values,
        reconstruction_mode=_enum_value(
            _raw_value(matrix_payload, "reconstruction_mode", "matrix"),
            ReconstructionMode,
            "matrix.reconstruction_mode",
        ),
        window_size=_require_int(
            _raw_value(matrix_payload, "window_size", "matrix"),
            "matrix.window_size",
            minimum=1,
        ),
        overlap=_require_int(
            _raw_value(matrix_payload, "overlap", "matrix"),
            "matrix.overlap",
            minimum=1,
        ),
        atomic_split_mode=_enum_value(
            _raw_value(matrix_payload, "atomic_split_mode", "matrix"),
            AtomicSplitMode,
            "matrix.atomic_split_mode",
        ),
    )

    window_payload = _require_mapping(
        _raw_value(payload, "window_reference", "campaign"), "window_reference"
    )
    window_reference = _parse_window_reference(window_payload)

    runtime_payload = _require_mapping(_raw_value(payload, "runtime", "campaign"), "runtime")
    _reject_unknown(runtime_payload, _RUNTIME_FIELDS, "runtime")
    cache_policy = _enum_value(
        _raw_value(runtime_payload, "cache_policy", "runtime"),
        CachePolicy,
        "runtime.cache_policy",
    )
    if overrides.cache_policy is not None:
        if not isinstance(overrides.cache_policy, CachePolicy):
            raise ValueError("runtime.cache_policy override is invalid")
        cache_policy = overrides.cache_policy
    runtime = RuntimeConfig(
        model_name=_enum_value(
            _raw_value(runtime_payload, "model_name", "runtime"), ModelName, "runtime.model_name"
        ),
        model_dtype=_require_string(
            _raw_value(runtime_payload, "model_dtype", "runtime"), "runtime.model_dtype"
        ),
        inference_device=_require_string(
            _raw_value(runtime_payload, "inference_device", "runtime"),
            "runtime.inference_device",
        ),
        process_device=_require_string(
            _raw_value(runtime_payload, "process_device", "runtime"),
            "runtime.process_device",
        ),
        jobs=_require_int(_raw_value(runtime_payload, "jobs", "runtime"), "runtime.jobs", minimum=1),
        gpu=(
            overrides.gpu
            if overrides.gpu is not None
            else _require_int(_raw_value(runtime_payload, "gpu", "runtime"), "runtime.gpu", minimum=0)
        ),
        cache_policy=cache_policy,
        failure_policy=_enum_value(
            _raw_value(runtime_payload, "failure_policy", "runtime"),
            FailurePolicy,
            "runtime.failure_policy",
        ),
    )

    storage_payload = _require_mapping(_raw_value(payload, "storage", "campaign"), "storage")
    _reject_unknown(storage_payload, _STORAGE_FIELDS, "storage")
    data_root = _resolve_path(
        overrides.data_root
        if overrides.data_root is not None
        else _require_string(_raw_value(storage_payload, "data_root", "storage"), "storage.data_root"),
        repository_root,
    )
    storage = StorageConfig(
        data_root=data_root,
        checkpoint=_resolve_path(
            overrides.checkpoint
            if overrides.checkpoint is not None
            else _require_string(
                _raw_value(storage_payload, "checkpoint", "storage"), "storage.checkpoint"
            ),
            repository_root,
        ),
        output_root=_resolve_path(
            overrides.output_root
            if overrides.output_root is not None
            else _require_string(
                _raw_value(storage_payload, "output_root", "storage"), "storage.output_root"
            ),
            repository_root,
        ),
        minimum_free_gb=_require_float(
            _raw_value(storage_payload, "minimum_free_gb", "storage"),
            "storage.minimum_free_gb",
            minimum=0.0,
        ),
        keep_artifacts=(
            overrides.keep_artifacts
            if overrides.keep_artifacts is not None
            else _require_bool(
                _raw_value(storage_payload, "keep_artifacts", "storage"),
                "storage.keep_artifacts",
            )
        ),
    )

    evaluation_payload = _require_mapping(
        _raw_value(payload, "evaluation", "campaign"), "evaluation"
    )
    _reject_unknown(evaluation_payload, _EVALUATION_FIELDS, "evaluation")
    evaluation = EvaluationConfig(
        pointcloud_config=_resolve_path(
            _require_string(
                _raw_value(evaluation_payload, "pointcloud_config", "evaluation"),
                "evaluation.pointcloud_config",
            ),
            repository_root,
        ),
        trajectory_config=_resolve_path(
            _require_string(
                _raw_value(evaluation_payload, "trajectory_config", "evaluation"),
                "evaluation.trajectory_config",
            ),
            repository_root,
        ),
    )

    scenes_payload = _require_mapping(_raw_value(payload, "scenes", "campaign"), "scenes")
    scenes: dict[str, SceneConfig] = {}
    for raw_scene_id, raw_scene in scenes_payload.items():
        scene_id = _require_string(raw_scene_id, "scenes.scene_id")
        scenes[scene_id] = _parse_scene(
            scene_id,
            _require_mapping(raw_scene, f"scene {scene_id}"),
            repository_root=repository_root,
            data_root=data_root,
        )

    presets_payload = _require_mapping(_raw_value(payload, "presets", "campaign"), "presets")
    presets: dict[str, PresetConfig] = {}
    for raw_preset, raw_value in presets_payload.items():
        preset = _require_string(raw_preset, "presets.preset")
        preset_payload = _require_mapping(raw_value, f"preset {preset}")
        _reject_unknown(preset_payload, _PRESET_FIELDS, f"preset {preset}")
        raw_scenes = _require_sequence(
            _raw_value(preset_payload, "scenes", f"preset {preset}"),
            f"preset {preset}.scenes",
        )
        parsed_scenes = tuple(
            _parse_preset_scene(
                _require_mapping(item, f"preset {preset}.scenes[{index}]"),
                f"preset {preset}.scenes[{index}]",
            )
            for index, item in enumerate(raw_scenes)
        )
        presets[preset] = PresetConfig(parsed_scenes)

    selected_preset = _require_string(overrides.preset, "preset")
    if selected_preset not in presets:
        raise ValueError(f"unknown preset: {selected_preset}")
    selected = list(presets[selected_preset].scenes)
    if overrides.scene_ids:
        scene_ids = tuple(_require_string(item, "scene_ids") for item in overrides.scene_ids)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("scene IDs must be unique")
        unknown = sorted(set(scene_ids) - {item.scene_id for item in selected})
        if unknown:
            raise ValueError("unknown selected scene: " + ", ".join(unknown))
        selected = [item for item in selected if item.scene_id in scene_ids]
    for item in selected:
        if item.scene_id not in scenes:
            raise ValueError(f"preset references unknown scene: {item.scene_id}")
    selected_scenes: list[PresetSceneConfig] = []
    for item in selected:
        start = (
            overrides.start_frame
            if overrides.start_frame is not None
            else item.start
        )
        stride = (
            overrides.frame_stride
            if overrides.frame_stride is not None
            else item.stride
        )
        _require_int(start, "start_frame", minimum=0)
        _require_int(stride, "frame_stride", minimum=1)
        stop = item.stop
        if overrides.max_frames is not None:
            max_frames = _require_int(overrides.max_frames, "max_frames", minimum=1)
            limit = start + max_frames * stride
            stop = limit if stop is None else min(stop, limit)
        selected_scenes.append(PresetSceneConfig(item.scene_id, start, stop, stride))

    return CampaignConfig(
        version=version,
        campaign_id=campaign_id,
        repository_root=repository_root,
        pipeline_config=pipeline_config,
        matrix=matrix,
        window_reference=window_reference,
        runtime=runtime,
        storage=storage,
        evaluation=evaluation,
        scenes=scenes,
        presets=presets,
        selected_preset=selected_preset,
        selected_scenes=tuple(selected_scenes),
    )


def load_campaign_config(
    path: str | Path,
    overrides: CampaignOverrides,
) -> LoadedCampaignConfig:
    if not isinstance(overrides, CampaignOverrides):
        raise ValueError("campaign overrides are invalid")
    try:
        raw = OmegaConf.load(Path(path))
        payload = OmegaConf.to_container(raw, resolve=True)
    except Exception as exc:
        raise ValueError("campaign config is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("campaign config must be a mapping")
    _reject_unknown(payload, _ROOT_FIELDS, "campaign")
    config = _construct_and_validate(payload, overrides)
    canonical = _campaign_payload(config)
    return LoadedCampaignConfig(
        config=config,
        resolved_payload=canonical,
        sha256=hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest(),
    )


__all__ = [
    "CachePolicy",
    "CampaignConfig",
    "CampaignOverrides",
    "DatasetKind",
    "EvaluationConfig",
    "EvaluationKind",
    "FailurePolicy",
    "LoadedCampaignConfig",
    "MatrixConfig",
    "PresetConfig",
    "PresetSceneConfig",
    "RuntimeConfig",
    "SceneConfig",
    "StorageConfig",
    "WindowReferenceValues",
    "load_campaign_config",
]
