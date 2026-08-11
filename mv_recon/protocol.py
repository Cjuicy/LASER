from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from pipeline.config import (
    LoadedPipelineConfig,
    LoopMethod,
    PipelineConfig,
    SegmentationMethod,
    load_pipeline_config,
)
from inference_engine.prediction_cache.fingerprint import (
    digest_image_manifest,
    sha256_file,
)
from pipeline.manifest import ImageManifest


PAPER_PROTOCOL_NAME = "laser_cvpr2026_table4_pi3"
PAPER_DATASETS = ("7scenes-dense", "NRGBD-dense")
EXPECTED_DATASET_SEQUENCE_COUNTS = {
    "7scenes-dense": 18,
    "NRGBD-dense": 9,
}

_PROTOCOL_KEYS = frozenset(
    {
        "name",
        "version",
        "strict",
        "preflight_only",
        "resume",
        "max_sequences",
        "pipeline_config",
        "pipeline_cache_dir",
        "prediction_cache_root",
        "prediction_cache_mode",
        "process_device",
        "dtype",
        "pipeline_overrides",
        "geometry",
        "paper_reference",
    }
)
_GEOMETRY_KEYS = frozenset(
    {
        "center_crop_size",
        "alignment",
        "icp_type",
        "icp_threshold_m",
        "normal_estimation",
        "fscore_thresholds_m",
    }
)
_REFERENCE_KEYS = frozenset(
    {
        "accuracy_mean_m",
        "accuracy_median_m",
        "completion_mean_m",
        "completion_median_m",
        "normal_consistency_mean",
        "normal_consistency_median",
    }
)


@dataclass(frozen=True)
class GeometryProtocol:
    center_crop_size: int
    alignment: str
    icp_type: str
    icp_threshold_m: float
    normal_estimation: str
    fscore_thresholds_m: tuple[float, ...]


@dataclass(frozen=True)
class DatasetReference:
    accuracy_mean_m: float
    accuracy_median_m: float
    completion_mean_m: float
    completion_median_m: float
    normal_consistency_mean: float
    normal_consistency_median: float


@dataclass(frozen=True)
class LaserPaperProtocol:
    name: str
    version: int
    strict: bool
    preflight_only: bool
    resume: bool
    max_sequences: int | None
    pipeline_config: Path
    pipeline_cache_dir: Path
    prediction_cache_root: Path
    prediction_cache_mode: str
    process_device: str
    dtype: str
    pipeline_overrides: tuple[str, ...]
    geometry: GeometryProtocol
    paper_reference: Mapping[str, DatasetReference]


@dataclass(frozen=True)
class ResolvedEvaluationProtocol:
    protocol: LaserPaperProtocol
    pipeline: LoadedPipelineConfig
    datasets: tuple[str, ...]
    output_dir: Path
    resolved_yaml: str
    sha256: str
    identity_sha256: str


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class DatasetPlan:
    name: str
    sequence_map_path: Path
    sequence_map_sha256: str
    expected_sequence_count: int
    sequences: tuple[SequenceSpec, ...]


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"invalid {label} fields: " + "; ".join(details))
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _resolve_path(value: object, repository_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("protocol paths must be non-empty strings")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _parse_geometry(raw: object) -> GeometryProtocol:
    values = _require_exact_keys(raw, _GEOMETRY_KEYS, "protocol.geometry")
    thresholds = values["fscore_thresholds_m"]
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("protocol.geometry.fscore_thresholds_m must be a list")
    parsed_thresholds = tuple(
        _finite_float(item, "protocol.geometry.fscore_thresholds_m")
        for item in thresholds
    )
    if any(item <= 0 for item in parsed_thresholds):
        raise ValueError("F-score thresholds must be positive")
    geometry = GeometryProtocol(
        center_crop_size=_positive_integer(
            values["center_crop_size"],
            "protocol.geometry.center_crop_size",
        ),
        alignment=str(values["alignment"]),
        icp_type=str(values["icp_type"]),
        icp_threshold_m=_finite_float(
            values["icp_threshold_m"],
            "protocol.geometry.icp_threshold_m",
        ),
        normal_estimation=str(values["normal_estimation"]),
        fscore_thresholds_m=parsed_thresholds,
    )
    if geometry.icp_threshold_m <= 0:
        raise ValueError("protocol.geometry.icp_threshold_m must be positive")
    return geometry


def _parse_references(raw: object) -> Mapping[str, DatasetReference]:
    if not isinstance(raw, dict) or tuple(raw) != PAPER_DATASETS:
        raise ValueError(
            "protocol.paper_reference must contain 7scenes-dense and "
            "NRGBD-dense in paper order"
        )
    parsed: dict[str, DatasetReference] = {}
    for dataset, value in raw.items():
        fields = _require_exact_keys(
            value,
            _REFERENCE_KEYS,
            f"protocol.paper_reference.{dataset}",
        )
        parsed[dataset] = DatasetReference(
            **{
                key: _finite_float(
                    fields[key],
                    f"protocol.paper_reference.{dataset}.{key}",
                )
                for key in _REFERENCE_KEYS
            }
        )
    return parsed


def _parse_protocol(
    node: DictConfig,
    repository_root: Path,
) -> LaserPaperProtocol:
    raw = OmegaConf.to_container(node, resolve=True)
    values = _require_exact_keys(raw, _PROTOCOL_KEYS, "protocol")
    max_sequences = values["max_sequences"]
    if max_sequences is not None:
        max_sequences = _positive_integer(
            max_sequences,
            "protocol.max_sequences",
        )
    overrides = values["pipeline_overrides"]
    if not isinstance(overrides, list) or not overrides or not all(
        isinstance(item, str) and item for item in overrides
    ):
        raise ValueError("protocol.pipeline_overrides must be non-empty strings")
    for flag in ("strict", "preflight_only", "resume"):
        if not isinstance(values[flag], bool):
            raise ValueError(f"protocol.{flag} must be boolean")
    if isinstance(values["version"], bool) or not isinstance(
        values["version"], int
    ):
        raise ValueError("protocol.version must be an integer")
    return LaserPaperProtocol(
        name=str(values["name"]),
        version=values["version"],
        strict=values["strict"],
        preflight_only=values["preflight_only"],
        resume=values["resume"],
        max_sequences=max_sequences,
        pipeline_config=_resolve_path(
            values["pipeline_config"], repository_root
        ),
        pipeline_cache_dir=_resolve_path(
            values["pipeline_cache_dir"], repository_root
        ),
        prediction_cache_root=_resolve_path(
            values["prediction_cache_root"], repository_root
        ),
        prediction_cache_mode=str(values["prediction_cache_mode"]),
        process_device=str(values["process_device"]),
        dtype=str(values["dtype"]),
        pipeline_overrides=tuple(overrides),
        geometry=_parse_geometry(values["geometry"]),
        paper_reference=_parse_references(values["paper_reference"]),
    )


def _resolve_dtype(
    requested: str,
    device: str,
    cuda_capability: tuple[int, int] | None,
) -> str:
    if requested != "auto":
        if requested not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported protocol dtype: {requested}")
        return requested
    if str(device).startswith("cpu"):
        return "float32"
    if cuda_capability is None:
        try:
            import torch

            if torch.cuda.is_available():
                cuda_capability = torch.cuda.get_device_capability()
        except (ImportError, RuntimeError):
            cuda_capability = None
    if cuda_capability is not None and cuda_capability[0] >= 8:
        return "bfloat16"
    return "float16"


def _validate_paper_locks(
    protocol: LaserPaperProtocol,
    config: PipelineConfig,
) -> None:
    expected = {
        "protocol.name": (protocol.name, PAPER_PROTOCOL_NAME),
        "protocol.version": (protocol.version, 1),
        "protocol.strict": (protocol.strict, True),
        "window.size": (config.window.size, 20),
        "window.overlap": (config.window.overlap, 5),
        "segmentation.method": (
            config.segmentation.method,
            SegmentationMethod.DEPTH,
        ),
        "segmentation.confidence_keep_ratio": (
            config.segmentation.confidence_keep_ratio,
            0.5,
        ),
        "segmentation.depth_merge_threshold": (
            config.segmentation.depth_merge_threshold,
            0.1,
        ),
        "segmentation.temporal_iou_threshold": (
            config.segmentation.temporal_iou_threshold,
            0.3,
        ),
        "segmentation.felzenszwalb.scale": (
            config.segmentation.felzenszwalb.scale,
            300,
        ),
        "segmentation.felzenszwalb.sigma": (
            config.segmentation.felzenszwalb.sigma,
            1.1,
        ),
        "segmentation.felzenszwalb.min_size": (
            config.segmentation.felzenszwalb.min_size,
            500,
        ),
        "anchor_propagation.enabled": (
            config.anchor_propagation.enabled,
            True,
        ),
        "anchor_propagation.correspondence_iou_threshold": (
            config.anchor_propagation.correspondence_iou_threshold,
            0.4,
        ),
        "loop.enabled": (config.loop.enabled, False),
        "loop.method": (config.loop.method, LoopMethod.TRADITIONAL),
        "loop.registration.confidence_keep_ratio": (
            config.loop.registration.confidence_keep_ratio,
            0.5,
        ),
        "geometry.center_crop_size": (
            protocol.geometry.center_crop_size,
            224,
        ),
        "geometry.alignment": (
            protocol.geometry.alignment,
            "umeyama_sim3_then_icp",
        ),
        "geometry.icp_type": (
            protocol.geometry.icp_type,
            "point_to_point",
        ),
        "geometry.icp_threshold_m": (
            protocol.geometry.icp_threshold_m,
            0.1,
        ),
        "geometry.normal_estimation": (
            protocol.geometry.normal_estimation,
            "open3d_default",
        ),
        "geometry.fscore_thresholds_m": (
            protocol.geometry.fscore_thresholds_m,
            (0.01, 0.02, 0.05),
        ),
    }
    drift = [
        f"{path}: expected {wanted!r}, got {actual!r}"
        for path, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if drift:
        raise ValueError("laser paper protocol drift: " + "; ".join(drift))


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def resolve_evaluation_protocol(
    hydra_cfg: DictConfig,
    repository_root: str | Path,
    *,
    cuda_capability: tuple[int, int] | None = None,
) -> ResolvedEvaluationProtocol:
    root = Path(repository_root).resolve()
    protocol = _parse_protocol(hydra_cfg.protocol, root)
    device = str(hydra_cfg.device)
    dtype = _resolve_dtype(protocol.dtype, device, cuda_capability)
    checkpoint = Path(str(hydra_cfg.pi3.checkpoint)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    output_dir = Path(str(hydra_cfg.output_dir)).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    operational = (
        f"model.checkpoint={checkpoint.resolve()}",
        f"model.inference_device={device}",
        f"model.process_device={protocol.process_device}",
        f"model.dtype={dtype}",
        f"output.cache_dir={protocol.pipeline_cache_dir}",
        f"prediction_cache.root={protocol.prediction_cache_root}",
        f"prediction_cache.mode={protocol.prediction_cache_mode}",
    )
    loaded = load_pipeline_config(
        protocol.pipeline_config,
        (*protocol.pipeline_overrides, *operational),
    )
    _validate_paper_locks(protocol, loaded.config)

    datasets = tuple(str(name) for name in hydra_cfg.eval_datasets)
    if datasets != PAPER_DATASETS:
        raise ValueError(
            f"laser paper protocol drift: expected datasets {PAPER_DATASETS}, "
            f"got {datasets}"
        )
    resolved_payload = {
        **_jsonable(asdict(protocol)),
        "eval_datasets": list(datasets),
    }
    resolved_yaml = OmegaConf.to_yaml(
        OmegaConf.create(resolved_payload),
        sort_keys=True,
    )
    identity_payload = dict(resolved_payload)
    identity_payload.pop("resume")
    identity_payload.pop("preflight_only")
    identity_yaml = OmegaConf.to_yaml(
        OmegaConf.create(identity_payload),
        sort_keys=True,
    )
    return ResolvedEvaluationProtocol(
        protocol=protocol,
        pipeline=loaded,
        datasets=datasets,
        output_dir=output_dir.resolve(),
        resolved_yaml=resolved_yaml,
        sha256=hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest(),
        identity_sha256=hashlib.sha256(
            identity_yaml.encode("utf-8")
        ).hexdigest(),
    )


def _unique_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate sequence name in sequence map: {key}")
        result[key] = value
    return result


def load_sequence_map(
    path: str | Path,
    expected_count: int,
) -> tuple[SequenceSpec, ...]:
    map_path = Path(path)
    if not map_path.is_file():
        raise FileNotFoundError(f"sequence map does not exist: {map_path}")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("expected sequence count must be a positive integer")
    try:
        raw = json.loads(
            map_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object_pairs,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sequence map JSON: {map_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("sequence map root must be a JSON object")
    if len(raw) != expected_count:
        raise ValueError(
            f"sequence map must contain exactly {expected_count} entries; "
            f"received {len(raw)}"
        )

    sequences: list[SequenceSpec] = []
    for name, frame_ids in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("sequence names must be non-empty strings")
        if not isinstance(frame_ids, list) or not frame_ids:
            raise ValueError(f"sequence {name} must contain frame IDs")
        if any(
            isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id < 0
            for frame_id in frame_ids
        ):
            raise ValueError(
                f"sequence {name} frame IDs must be non-negative integers"
            )
        if any(
            right - left != 10
            for left, right in zip(frame_ids, frame_ids[1:])
        ):
            raise ValueError(f"sequence {name} must use interval 10")
        sequences.append(SequenceSpec(name=name, frame_ids=tuple(frame_ids)))
    return tuple(sequences)


def build_dataset_plans(
    resolved: ResolvedEvaluationProtocol,
    data_config: DictConfig,
    repository_root: str | Path,
) -> tuple[DatasetPlan, ...]:
    root = Path(repository_root).resolve()
    plans: list[DatasetPlan] = []
    for dataset_name in resolved.datasets:
        if dataset_name not in EXPECTED_DATASET_SEQUENCE_COUNTS:
            raise ValueError(f"unsupported paper dataset: {dataset_name}")
        if dataset_name not in data_config:
            raise ValueError(f"missing data configuration: {dataset_name}")
        configured_path = Path(str(data_config[dataset_name].seq_id_map))
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        configured_path = configured_path.resolve()
        expected_count = EXPECTED_DATASET_SEQUENCE_COUNTS[dataset_name]
        complete_sequences = load_sequence_map(
            configured_path,
            expected_count=expected_count,
        )
        limit = resolved.protocol.max_sequences
        selected = (
            complete_sequences
            if limit is None
            else complete_sequences[:limit]
        )
        plans.append(
            DatasetPlan(
                name=dataset_name,
                sequence_map_path=configured_path,
                sequence_map_sha256=sha256_file(configured_path),
                expected_sequence_count=expected_count,
                sequences=selected,
            )
        )
    return tuple(plans)


def validate_dataset_plan(dataset: object, plan: DatasetPlan) -> None:
    if not hasattr(dataset, "sequence_list") or not hasattr(
        dataset, "get_seq_framenum"
    ):
        raise TypeError(
            "point-map dataset must expose sequence_list and get_seq_framenum"
        )
    available = set(dataset.sequence_list)
    for sequence in plan.sequences:
        if sequence.name not in available:
            raise ValueError(f"dataset is missing sequence {sequence.name}")
        frame_count = int(
            dataset.get_seq_framenum(sequence_name=sequence.name)
        )
        if frame_count < 1:
            raise ValueError(
                f"sequence {sequence.name} has invalid frame count {frame_count}"
            )
        last_frame = sequence.frame_ids[-1]
        if last_frame >= frame_count:
            raise ValueError(
                f"sequence {sequence.name} requests frame {last_frame} "
                f"but contains {frame_count} frames"
            )


def manifest_digest_for_paths(paths: Sequence[str | Path]) -> str:
    normalized = tuple(Path(path).resolve() for path in paths)
    if not normalized:
        raise ValueError("image manifest must contain at least one path")
    return digest_image_manifest(ImageManifest(paths=normalized))
