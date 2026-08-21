from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, ValidationError


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


class ConfidenceQuantileMethod(str, Enum):
    HIGHER = "higher"
    NEAREST = "nearest"


class ModelName(str, Enum):
    PI3 = "pi3"


class PredictionCacheMode(str, Enum):
    AUTO = "auto"
    REFRESH = "refresh"
    READONLY = "readonly"
    OFF = "off"


@dataclass(frozen=True)
class InputConfig:
    image_dir: str = MISSING
    sample_stride: int = MISSING


@dataclass(frozen=True)
class OutputConfig:
    scene_name: str = MISSING
    cache_dir: str = MISSING
    result_dir: str = MISSING
    save_diagnostics: bool = MISSING


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName = MISSING
    checkpoint: str = MISSING
    inference_device: str = MISSING
    process_device: str = MISSING
    dtype: str = MISSING


@dataclass(frozen=True)
class PredictionCacheConfig:
    root: str = MISSING
    mode: PredictionCacheMode = MISSING


@dataclass(frozen=True)
class WindowConfig:
    size: int = MISSING
    overlap: int = MISSING


@dataclass(frozen=True)
class FelzenszwalbConfig:
    scale: float = MISSING
    sigma: float = MISSING
    min_size: int = MISSING


@dataclass(frozen=True)
class GeometryConfig:
    normal_method: str = MISSING
    normal_threshold_degrees: float = MISSING


@dataclass(frozen=True)
class AtomicConfig:
    split_mode: AtomicSplitMode = MISSING
    split_score_threshold: float = MISSING


@dataclass(frozen=True)
class WindowReferenceConfig:
    enabled: bool = MISSING
    sampling_stride: int = MISSING
    max_keyframes: int = MISSING
    relative_depth_tolerance: float = MISSING
    min_reference_score: float = MISSING
    stop_coverage_ratio: float = MISSING
    min_coverage_gain: float = MISSING
    min_region_correspondences: int = MISSING
    min_region_coverage: float = MISSING
    min_region_purity: float = MISSING
    merge_vote_threshold: float = MISSING


@dataclass(frozen=True)
class SegmentationConfig:
    method: SegmentationMethod = MISSING
    confidence_keep_ratio: float = MISSING
    confidence_quantile_method: ConfidenceQuantileMethod = MISSING
    depth_merge_threshold: float = MISSING
    temporal_iou_threshold: float = MISSING
    felzenszwalb: FelzenszwalbConfig = field(default_factory=FelzenszwalbConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    atomic: AtomicConfig = field(default_factory=AtomicConfig)
    window_reference: WindowReferenceConfig = field(
        default_factory=WindowReferenceConfig
    )


@dataclass(frozen=True)
class AnchorPropagationConfig:
    enabled: bool = MISSING
    correspondence_iou_threshold: float = MISSING


@dataclass(frozen=True)
class RegistrationConfig:
    confidence_keep_ratio: float = MISSING


@dataclass(frozen=True)
class ReconstructionConfig:
    mode: ReconstructionMode = MISSING


@dataclass(frozen=True)
class DetectionConfig:
    method: str = MISSING
    salad_checkpoint: str = MISSING
    dino_checkpoint: str = MISSING
    image_size: list[int] = MISSING
    batch_size: int = MISSING
    similarity_threshold: float = MISSING
    top_k: int = MISSING
    nms_enabled: bool = MISSING
    nms_frame_radius: int = MISSING


@dataclass(frozen=True)
class ConstraintConfig:
    chunk_size: int = MISSING


@dataclass(frozen=True)
class OptimizerConfig:
    implementation: str = MISSING
    use_sim3: bool = MISSING
    max_iterations: int = MISSING
    initial_damping: float = MISSING


@dataclass(frozen=True)
class LoopConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class PipelineConfig:
    version: int = MISSING
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prediction_cache: PredictionCacheConfig = field(
        default_factory=PredictionCacheConfig
    )
    window: WindowConfig = field(default_factory=WindowConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    anchor_propagation: AnchorPropagationConfig = field(
        default_factory=AnchorPropagationConfig
    )
    registration: RegistrationConfig = field(
        default_factory=RegistrationConfig
    )
    reconstruction: ReconstructionConfig = field(
        default_factory=ReconstructionConfig
    )
    loop: LoopConfig | None = None


@dataclass(frozen=True)
class LoadedPipelineConfig:
    config: PipelineConfig
    resolved_yaml: str
    sha256: str


def _make_schema_mutable(node: object) -> None:
    """Let OmegaConf populate frozen dataclass nodes before reconstruction."""

    if not OmegaConf.is_config(node) or node is None:
        return
    OmegaConf.set_readonly(node, False)
    if isinstance(node, DictConfig):
        if node._is_none():
            return
        for _, child in node.items_ex(resolve=False):
            _make_schema_mutable(child)


def _normalize_enum_values(config: DictConfig) -> None:
    for path, enum_type in (
        ("model.name", ModelName),
        ("prediction_cache.mode", PredictionCacheMode),
        ("segmentation.method", SegmentationMethod),
        ("segmentation.atomic.split_mode", AtomicSplitMode),
        (
            "segmentation.confidence_quantile_method",
            ConfidenceQuantileMethod,
        ),
        ("reconstruction.mode", ReconstructionMode),
    ):
        value = OmegaConf.select(config, path, default=None)
        if value is None or isinstance(value, enum_type):
            continue
        if enum_type is PredictionCacheMode and value is False:
            value = PredictionCacheMode.OFF.value
        try:
            normalized = enum_type(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(member.value for member in enum_type)
            raise ValueError(
                f"invalid configuration value for {path}: "
                f"{value!r}; expected one of {allowed}"
            ) from exc
        OmegaConf.update(config, path, normalized, merge=False)


def _positive_int(path: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")


def _validate_config(config: PipelineConfig) -> None:
    if config.version != 2:
        raise ValueError("version must be 2")
    if config.input.sample_stride < 1:
        raise ValueError("input.sample_stride must be at least 1")
    if not config.window.size > config.window.overlap >= 1:
        raise ValueError(
            "window.size must be greater than window.overlap >= 1"
        )

    for path, ratio in (
        (
            "segmentation.confidence_keep_ratio",
            config.segmentation.confidence_keep_ratio,
        ),
        (
            "registration.confidence_keep_ratio",
            config.registration.confidence_keep_ratio,
        ),
    ):
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError(f"{path} keep_ratio must be in (0, 1]")

    for path, ratio in (
        (
            "segmentation.temporal_iou_threshold",
            config.segmentation.temporal_iou_threshold,
        ),
        (
            "anchor_propagation.correspondence_iou_threshold",
            config.anchor_propagation.correspondence_iou_threshold,
        ),
    ):
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{path} must be in [0, 1]")

    if (
        not math.isfinite(config.segmentation.depth_merge_threshold)
        or config.segmentation.depth_merge_threshold < 0
    ):
        raise ValueError(
            "segmentation.depth_merge_threshold must be non-negative"
        )
    if config.segmentation.felzenszwalb.scale <= 0:
        raise ValueError("segmentation.felzenszwalb.scale must be positive")
    if config.segmentation.felzenszwalb.sigma < 0:
        raise ValueError(
            "segmentation.felzenszwalb.sigma must be non-negative"
        )
    if config.segmentation.felzenszwalb.min_size < 1:
        raise ValueError(
            "segmentation.felzenszwalb.min_size must be at least 1"
        )
    if (
        not math.isfinite(config.segmentation.atomic.split_score_threshold)
        or config.segmentation.atomic.split_score_threshold < 0
    ):
        raise ValueError(
            "segmentation.atomic.split_score_threshold must be non-negative"
        )
    if config.segmentation.geometry.normal_method not in {"cross", "sobel"}:
        raise ValueError(
            "segmentation.geometry.normal_method must be cross or sobel"
        )
    normal_threshold = (
        config.segmentation.geometry.normal_threshold_degrees
    )
    if not math.isfinite(normal_threshold) or not 0 < normal_threshold <= 180:
        raise ValueError(
            "segmentation.geometry.normal_threshold_degrees "
            "must be in (0, 180]"
        )

    window_reference = config.segmentation.window_reference
    for name in (
        "sampling_stride",
        "max_keyframes",
        "min_region_correspondences",
    ):
        _positive_int(
            f"segmentation.window_reference.{name}",
            getattr(window_reference, name),
        )

    relative_depth_tolerance = window_reference.relative_depth_tolerance
    if (
        not math.isfinite(relative_depth_tolerance)
        or relative_depth_tolerance <= 0
    ):
        raise ValueError(
            "segmentation.window_reference.relative_depth_tolerance "
            "must be finite and in (0, inf)"
        )

    for name in (
        "min_reference_score",
        "stop_coverage_ratio",
        "min_region_coverage",
        "min_region_purity",
        "merge_vote_threshold",
    ):
        value = getattr(window_reference, name)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                f"segmentation.window_reference.{name} "
                "must be finite and in (0, 1]"
    )

    min_coverage_gain = window_reference.min_coverage_gain
    if (
        not math.isfinite(min_coverage_gain)
        or not 0.0 <= min_coverage_gain <= 1.0
    ):
        raise ValueError(
            "segmentation.window_reference.min_coverage_gain "
            "must be finite and in [0, 1]"
        )

    if config.reconstruction.mode is ReconstructionMode.NO_LOOP:
        return
    if config.loop is None:
        raise ValueError(
            f"{config.reconstruction.mode.value} requires loop configuration"
        )
    if config.loop.detection.method != "salad":
        raise ValueError("loop.detection.method must be salad")
    if (
        len(config.loop.detection.image_size) != 2
        or any(size < 1 for size in config.loop.detection.image_size)
    ):
        raise ValueError(
            "loop.detection.image_size must contain two positive integers"
        )
    if config.loop.detection.batch_size < 1:
        raise ValueError("loop.detection.batch_size must be at least 1")
    if config.loop.detection.top_k < 1:
        raise ValueError("loop.detection.top_k must be at least 1")
    if config.loop.detection.nms_frame_radius < 0:
        raise ValueError(
            "loop.detection.nms_frame_radius must be non-negative"
        )
    if config.loop.constraint.chunk_size < 1:
        raise ValueError("loop.constraint.chunk_size must be at least 1")
    if config.loop.optimizer.max_iterations < 1:
        raise ValueError(
            "loop.optimizer.max_iterations must be at least 1"
        )
    if (
        not math.isfinite(config.loop.optimizer.initial_damping)
        or config.loop.optimizer.initial_damping < 0
    ):
        raise ValueError(
            "loop.optimizer.initial_damping must be non-negative"
        )


def load_pipeline_config(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> LoadedPipelineConfig:
    """Load, resolve, validate, and fingerprint the one canonical config."""

    try:
        source = OmegaConf.load(Path(path))
        dotlist = OmegaConf.from_dotlist(list(overrides))
        has_loop_configuration = (
            OmegaConf.select(source, "loop", default=None) is not None
            or OmegaConf.select(dotlist, "loop", default=None) is not None
        )
        schema = OmegaConf.structured(
            PipelineConfig(
                loop=LoopConfig() if has_loop_configuration else None
            )
        )
        _make_schema_mutable(schema)
        _normalize_enum_values(source)
        _normalize_enum_values(dotlist)
        missing_value = object()
        for values in (source, dotlist):
            enabled = OmegaConf.select(
                values,
                "segmentation.window_reference.enabled",
                default=missing_value,
            )
            if enabled is not missing_value and not isinstance(enabled, bool):
                raise ValueError(
                    "segmentation.window_reference.enabled must be a boolean"
                )
        merged = OmegaConf.merge(schema, source, dotlist)

        missing = sorted(OmegaConf.missing_keys(merged))
        if missing:
            raise ValueError(
                "missing required configuration fields: "
                + ", ".join(missing)
            )

        resolved_yaml = OmegaConf.to_yaml(
            merged,
            resolve=True,
            sort_keys=True,
        )
        config = OmegaConf.to_object(merged)
    except ConfigKeyError as exc:
        raise ValueError(f"unknown configuration field: {exc}") from exc
    except MissingMandatoryValue as exc:
        raise ValueError(f"missing required configuration field: {exc}") from exc
    except ValidationError as exc:
        raise ValueError(f"invalid configuration value: {exc}") from exc

    if not isinstance(config, PipelineConfig):
        raise TypeError("structured configuration did not produce PipelineConfig")

    _validate_config(config)
    digest = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
    return LoadedPipelineConfig(
        config=config,
        resolved_yaml=resolved_yaml,
        sha256=digest,
    )
