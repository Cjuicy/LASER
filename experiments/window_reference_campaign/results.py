"""Finite, resumable compact records for the window-reference campaign."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from inference_engine.prediction_cache.store import PredictionStoreStats

from .diagnostics import DiagnosticsSummary, DistributionSummary
from .matrix import RunIdentity, RunIdentitySeed, complete_identity


RUN_SCHEMA_VERSION = 1
POINTCLOUD_METRICS = (
    "accuracy_mean_m", "accuracy_median_m",
    "completion_mean_m", "completion_median_m",
    "normal_consistency_mean", "normal_consistency_median",
    "chamfer_l1_m",
    "precision_1cm", "recall_1cm", "fscore_1cm",
    "precision_2cm", "recall_2cm", "fscore_2cm",
    "precision_5cm", "recall_5cm", "fscore_5cm",
)
TRAJECTORY_METRICS = (
    "internal_ate_rmse_m",
    "internal_rpe_translation_rmse_m",
    "internal_rpe_rotation_rmse_deg",
    "internal_matched_frame_count",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_FIELDS = (
    "schema_version", "run_id", "identity_seed", "identity", "status",
    "attempt", "failure_stage", "started_at", "finished_at",
    "frame_count", "window_count", "cache_policy", "cache_stats",
    "timings", "diagnostics", "evaluation_kind", "evaluation_metrics",
    "artifact_manifest_sha256", "error",
)
_CACHE_POLICIES = {"auto", "refresh", "readonly", "off"}
_EVALUATION_KINDS = {"none", "pointcloud", "internal_trajectory"}
_RUN_ORDER = {
    "depth__wr-off": 0,
    "depth__wr-on": 1,
    "geometry__wr-off": 2,
    "geometry__wr-on": 3,
    "atomic__wr-off": 4,
    "atomic__wr-on": 5,
}
_IDENTITY_COLUMNS = (
    "dataset", "scene", "slice_id", "run_id", "segmentation_method",
    "window_reference_enabled",
)


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailureStage(str, Enum):
    PREFLIGHT = "preflight"
    STAGING = "staging"
    RECONSTRUCTION = "reconstruction"
    EVALUATION = "evaluation"
    COMPACTION = "compaction"
    CLEANUP = "cleanup"


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _finite_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> int | float:
    if integer:
        if type(value) is not int:
            raise ValueError(f"{field} must be an integer")
        normalized: int | float = value
    else:
        if type(value) not in (int, float):
            raise ValueError(f"{field} must be finite")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{field} must be finite")
    number = float(normalized)
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return normalized


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_finite_json(value: object, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} mapping keys must be strings")
            _validate_finite_json(item, f"{field}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{field}[{index}]")
        return
    raise ValueError(f"{field} contains an unsupported value")


@dataclass(frozen=True)
class CacheStats:
    ordinary_hits: int
    ordinary_misses: int
    corrupt_count: int
    read_ms: float
    write_ms: float
    saved_window_count: int
    stored_bytes: int
    events: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        for name in (
            "ordinary_hits", "ordinary_misses", "corrupt_count",
            "saved_window_count", "stored_bytes",
        ):
            _finite_number(getattr(self, name), name, minimum=0, integer=True)
        for name in ("read_ms", "write_ms"):
            normalized = float(_finite_number(getattr(self, name), name, minimum=0))
            object.__setattr__(self, name, normalized)
        if not isinstance(self.events, (tuple, list)):
            raise ValueError("cache events must be a tuple")
        frozen_events: list[Mapping[str, object]] = []
        for event in self.events:
            if not isinstance(event, Mapping):
                raise ValueError("cache events must be mappings")
            _validate_finite_json(event, "cache event")
            frozen = _freeze_json(event)
            assert isinstance(frozen, Mapping)
            frozen_events.append(frozen)
        object.__setattr__(self, "events", tuple(frozen_events))

    @classmethod
    def from_store_stats(cls, stats: PredictionStoreStats) -> "CacheStats":
        if not isinstance(stats, PredictionStoreStats):
            raise ValueError("cache stats require PredictionStoreStats")
        return cls(
            ordinary_hits=stats.ordinary_hits,
            ordinary_misses=stats.ordinary_misses,
            corrupt_count=stats.corrupt_count,
            read_ms=stats.read_ms,
            write_ms=stats.write_ms,
            saved_window_count=stats.saved_window_count,
            stored_bytes=stats.stored_bytes,
            events=tuple(dict(event) for event in stats.events),
        )


@dataclass(frozen=True)
class RunTimings:
    reconstruction_s: float | None
    evaluation_s: float | None

    def __post_init__(self) -> None:
        for name in ("reconstruction_s", "evaluation_s"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    float(_finite_number(value, name, minimum=0)),
                )


@dataclass(frozen=True)
class RunError:
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("error_type must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("error message must be a non-empty string")


@dataclass(frozen=True)
class RunRecord:
    schema_version: int
    run_id: str
    identity_seed: RunIdentitySeed
    identity: RunIdentity | None
    status: RunStatus
    attempt: int
    failure_stage: FailureStage | None
    started_at: str
    finished_at: str
    frame_count: int
    window_count: int
    cache_policy: object
    cache_stats: CacheStats
    timings: RunTimings
    diagnostics: DiagnosticsSummary | None
    evaluation_kind: object
    evaluation_metrics: Mapping[str, object] | None
    artifact_manifest_sha256: str | None
    error: RunError | None

    def to_payload(self) -> dict[str, object]:
        return _record_payload(self)


@dataclass(frozen=True)
class SummaryPaths:
    runs_csv: Path
    diagnostics_csv: Path
    trajectory_csv: Path
    pointcloud_csv: Path
    summary_json: Path
    failures_json: Path


def _distribution_payload(value: DistributionSummary | None) -> object:
    if value is None:
        return None
    return {
        "count": value.count,
        "minimum": value.minimum,
        "maximum": value.maximum,
        "mean": value.mean,
        "median": value.median,
    }


def _diagnostics_payload(value: DiagnosticsSummary | None) -> object:
    if value is None:
        return None
    payload: dict[str, object] = {
        "unique_frame_count": value.unique_frame_count,
        "window_frame_observation_count": value.window_frame_observation_count,
        "window_count": value.window_count,
        "region_count": _distribution_payload(value.region_count),
        "refinement_enabled": value.refinement_enabled,
        "refinement_state": value.refinement_state,
        "keyframe_count": _distribution_payload(value.keyframe_count),
        "keyframe_index_histogram": value.keyframe_index_histogram,
        "keyframe_index_records": value.keyframe_index_records,
        "coverage_ratio": _distribution_payload(value.coverage_ratio),
        "regions_before": _distribution_payload(value.regions_before),
        "regions_after": _distribution_payload(value.regions_after),
        "region_reduction_absolute_total": value.region_reduction_absolute_total,
        "region_reduction_relative": _distribution_payload(value.region_reduction_relative),
        "candidate_edge_total": value.candidate_edge_total,
        "accepted_edge_total": value.accepted_edge_total,
        "conflict_edge_total": value.conflict_edge_total,
        "projected_sample_total": value.projected_sample_total,
        "occluded_sample_total": value.occluded_sample_total,
        "depth_rejected_sample_total": value.depth_rejected_sample_total,
        "applied_frame_count": value.applied_frame_count,
        "applied_frame_rate": value.applied_frame_rate,
        "fallback_reason_histogram": value.fallback_reason_histogram,
    }
    return payload


def _record_payload(record: RunRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "run_id": record.run_id,
        "identity_seed": record.identity_seed.to_payload(),
        "identity": None if record.identity is None else record.identity.to_payload(),
        "status": _enum_value(record.status),
        "attempt": record.attempt,
        "failure_stage": _enum_value(record.failure_stage),
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "frame_count": record.frame_count,
        "window_count": record.window_count,
        "cache_policy": _enum_value(record.cache_policy),
        "cache_stats": {
            "ordinary_hits": record.cache_stats.ordinary_hits,
            "ordinary_misses": record.cache_stats.ordinary_misses,
            "corrupt_count": record.cache_stats.corrupt_count,
            "read_ms": record.cache_stats.read_ms,
            "write_ms": record.cache_stats.write_ms,
            "saved_window_count": record.cache_stats.saved_window_count,
            "stored_bytes": record.cache_stats.stored_bytes,
            "events": record.cache_stats.events,
        },
        "timings": {
            "reconstruction_s": record.timings.reconstruction_s,
            "evaluation_s": record.timings.evaluation_s,
        },
        "diagnostics": _diagnostics_payload(record.diagnostics),
        "evaluation_kind": _enum_value(record.evaluation_kind),
        "evaluation_metrics": record.evaluation_metrics,
        "artifact_manifest_sha256": record.artifact_manifest_sha256,
        "error": None if record.error is None else {
            "error_type": record.error.error_type,
            "message": record.error.message,
        },
    }


def _record_seed(record: RunRecord) -> RunIdentitySeed:
    return RunIdentitySeed(**record.identity_seed.to_payload())


def _validate_identity(record: RunRecord) -> None:
    if not isinstance(record.identity_seed, RunIdentitySeed):
        raise ValueError("identity_seed is invalid")
    if record.identity is None:
        return
    if not isinstance(record.identity, RunIdentity):
        raise ValueError("identity is invalid")
    expected_seed = RunIdentitySeed(
        **{
            key: value
            for key, value in record.identity.to_payload().items()
            if key != "prediction_key"
        }
    )
    if expected_seed != record.identity_seed:
        raise ValueError("identity does not match identity_seed")
    complete_identity(record.identity_seed, record.identity.prediction_key)


def _validate_metrics(kind: str, metrics: Mapping[str, object] | None) -> None:
    if kind == "none":
        if metrics is not None:
            raise ValueError("evaluation_kind none requires no metrics")
        return
    if metrics is None or not isinstance(metrics, Mapping):
        raise ValueError(f"{kind} evaluation metrics are required")
    if kind == "pointcloud":
        if set(metrics) != set(POINTCLOUD_METRICS):
            raise ValueError("pointcloud metrics must match the compact schema")
        for name in POINTCLOUD_METRICS:
            value = float(_finite_number(metrics[name], name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            if name.startswith(("precision_", "recall_", "fscore_")) and value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        return
    if kind == "internal_trajectory":
        if set(metrics) != set(TRAJECTORY_METRICS):
            raise ValueError("internal trajectory metrics must match the compact schema")
        for name in TRAJECTORY_METRICS[:3]:
            if type(metrics[name]) is not float:
                raise ValueError(f"{name} must be a float")
            _finite_number(metrics[name], name, minimum=0)
        _finite_number(metrics[TRAJECTORY_METRICS[3]], TRAJECTORY_METRICS[3], minimum=2, integer=True)
        return
    raise ValueError("evaluation_kind is invalid")


def _validate_record(record: RunRecord) -> None:
    if not isinstance(record, RunRecord):
        raise ValueError("run record is invalid")
    if record.schema_version != RUN_SCHEMA_VERSION:
        raise ValueError("run schema_version is unsupported")
    if not isinstance(record.run_id, str) or not record.run_id:
        raise ValueError("run_id must be a non-empty string")
    _validate_identity(record)
    if not isinstance(record.status, RunStatus):
        raise ValueError("status is invalid")
    _finite_number(record.attempt, "attempt", minimum=1, integer=True)
    if record.failure_stage is not None and not isinstance(record.failure_stage, FailureStage):
        raise ValueError("failure_stage is invalid")
    if not isinstance(record.started_at, str) or not record.started_at:
        raise ValueError("started_at must be a non-empty string")
    if not isinstance(record.finished_at, str) or not record.finished_at:
        raise ValueError("finished_at must be a non-empty string")
    _finite_number(record.frame_count, "frame_count", minimum=0, integer=True)
    _finite_number(record.window_count, "window_count", minimum=0, integer=True)
    cache_policy = _enum_value(record.cache_policy)
    if cache_policy not in _CACHE_POLICIES:
        raise ValueError("cache_policy is invalid")
    if not isinstance(record.cache_stats, CacheStats):
        raise ValueError("cache_stats is invalid")
    if not isinstance(record.timings, RunTimings):
        raise ValueError("timings is invalid")
    evaluation_kind = _enum_value(record.evaluation_kind)
    if evaluation_kind not in _EVALUATION_KINDS:
        raise ValueError("evaluation_kind is invalid")
    if record.evaluation_metrics is not None:
        _validate_finite_json(record.evaluation_metrics, "evaluation_metrics")
    if record.artifact_manifest_sha256 is not None and (
        not isinstance(record.artifact_manifest_sha256, str)
        or _SHA256_RE.fullmatch(record.artifact_manifest_sha256) is None
    ):
        raise ValueError("artifact_manifest_sha256 must be SHA256 hex")
    if record.error is not None and not isinstance(record.error, RunError):
        raise ValueError("error is invalid")

    if record.status is RunStatus.SUCCEEDED:
        if record.identity is None or record.diagnostics is None:
            raise ValueError("successful run requires identity and diagnostics")
        if record.failure_stage is not None or record.error is not None:
            raise ValueError("successful run cannot contain failure or error")
        if record.frame_count <= 0 or record.attempt <= 0:
            raise ValueError("successful run requires positive frame and attempt counts")
        if not isinstance(record.diagnostics, DiagnosticsSummary):
            raise ValueError("successful run diagnostics are invalid")
        expected_run_id = (
            f"{_enum_value(record.identity_seed.segmentation_method)}__wr-"
            f"{'on' if record.identity_seed.window_reference_enabled else 'off'}"
        )
        if record.run_id != expected_run_id:
            raise ValueError("run_id does not match identity seed")
        if record.diagnostics.refinement_enabled != record.identity_seed.window_reference_enabled:
            raise ValueError("diagnostics refinement state does not match identity seed")
        _validate_metrics(str(evaluation_kind), record.evaluation_metrics)
    else:
        if record.identity is not None or record.diagnostics is not None:
            raise ValueError("failed run cannot contain identity or diagnostics")
        if record.failure_stage is None or record.error is None:
            raise ValueError("failed run requires failure_stage and error")
        if record.evaluation_metrics is not None:
            raise ValueError("failed run cannot contain evaluation metrics")
        if record.artifact_manifest_sha256 is not None and record.failure_stage not in {
            FailureStage.EVALUATION,
            FailureStage.COMPACTION,
        }:
            raise ValueError("failed run artifact digest is only valid after evaluation")


def _atomic_text(path: Path, text: str, *, validate_json: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        serialized = temporary_path.read_text(encoding="utf-8")
        if serialized != text:
            raise ValueError("temporary output changed before atomic replacement")
        if validate_json:
            json.loads(serialized)
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def _json_value(value: object) -> object:
    value = _enum_value(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON payload must contain finite numbers")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


def atomic_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    if not isinstance(payload, Mapping):
        raise ValueError("JSON payload must be a mapping")
    encoded = json.dumps(
        _json_value(payload),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return _atomic_text(Path(path), encoded, validate_json=True)


def write_run_record(path: str | Path, record: RunRecord) -> Path:
    _validate_record(record)
    return atomic_json(path, record.to_payload())


def _require_mapping(payload: object, field: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return payload


def _distribution_from_payload(payload: object, field: str) -> DistributionSummary | None:
    if payload is None:
        return None
    mapping = _require_mapping(payload, field)
    try:
        return DistributionSummary(
            count=mapping["count"],
            minimum=mapping["minimum"],
            maximum=mapping["maximum"],
            mean=mapping["mean"],
            median=mapping["median"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{field} is invalid") from exc


def _diagnostics_from_payload(payload: object) -> DiagnosticsSummary | None:
    if payload is None:
        return None
    mapping = _require_mapping(payload, "diagnostics")
    try:
        histogram = mapping["keyframe_index_histogram"]
        records = mapping["keyframe_index_records"]
        fallback = mapping["fallback_reason_histogram"]
        return DiagnosticsSummary(
            unique_frame_count=mapping["unique_frame_count"],
            window_frame_observation_count=mapping["window_frame_observation_count"],
            window_count=mapping["window_count"],
            region_count=_distribution_from_payload(mapping["region_count"], "region_count"),
            refinement_enabled=mapping["refinement_enabled"],
            refinement_state=mapping["refinement_state"],
            keyframe_count=_distribution_from_payload(mapping["keyframe_count"], "keyframe_count"),
            keyframe_index_histogram=None if histogram is None else dict(_require_mapping(histogram, "keyframe_index_histogram")),
            keyframe_index_records=None if records is None else tuple(records),
            coverage_ratio=_distribution_from_payload(mapping["coverage_ratio"], "coverage_ratio"),
            regions_before=_distribution_from_payload(mapping["regions_before"], "regions_before"),
            regions_after=_distribution_from_payload(mapping["regions_after"], "regions_after"),
            region_reduction_absolute_total=mapping["region_reduction_absolute_total"],
            region_reduction_relative=_distribution_from_payload(mapping["region_reduction_relative"], "region_reduction_relative"),
            candidate_edge_total=mapping["candidate_edge_total"],
            accepted_edge_total=mapping["accepted_edge_total"],
            conflict_edge_total=mapping["conflict_edge_total"],
            projected_sample_total=mapping["projected_sample_total"],
            occluded_sample_total=mapping["occluded_sample_total"],
            depth_rejected_sample_total=mapping["depth_rejected_sample_total"],
            applied_frame_count=mapping["applied_frame_count"],
            applied_frame_rate=mapping["applied_frame_rate"],
            fallback_reason_histogram=None if fallback is None else dict(_require_mapping(fallback, "fallback_reason_histogram")),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("diagnostics payload is invalid") from exc


def _record_from_payload(payload: Mapping[str, object]) -> RunRecord:
    if set(payload) != set(_RUN_FIELDS):
        raise ValueError("run record fields do not match schema")
    seed_payload = _require_mapping(payload["identity_seed"], "identity_seed")
    try:
        seed = RunIdentitySeed(**dict(seed_payload))
    except (TypeError, ValueError) as exc:
        raise ValueError("identity_seed is invalid") from exc
    identity_payload = payload["identity"]
    identity: RunIdentity | None
    if identity_payload is None:
        identity = None
    else:
        try:
            identity = RunIdentity(**dict(_require_mapping(identity_payload, "identity")))
        except (TypeError, ValueError) as exc:
            raise ValueError("identity is invalid") from exc
    try:
        status = RunStatus(payload["status"])
        failure_raw = payload["failure_stage"]
        failure_stage = None if failure_raw is None else FailureStage(failure_raw)
        cache_payload = _require_mapping(payload["cache_stats"], "cache_stats")
        cache_stats = CacheStats(
            ordinary_hits=cache_payload["ordinary_hits"],
            ordinary_misses=cache_payload["ordinary_misses"],
            corrupt_count=cache_payload["corrupt_count"],
            read_ms=cache_payload["read_ms"],
            write_ms=cache_payload["write_ms"],
            saved_window_count=cache_payload["saved_window_count"],
            stored_bytes=cache_payload["stored_bytes"],
            events=tuple(cache_payload["events"]),
        )
        timing_payload = _require_mapping(payload["timings"], "timings")
        timings = RunTimings(
            reconstruction_s=timing_payload["reconstruction_s"],
            evaluation_s=timing_payload["evaluation_s"],
        )
        error_raw = payload["error"]
        error = None if error_raw is None else RunError(**dict(_require_mapping(error_raw, "error")))
        record = RunRecord(
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            identity_seed=seed,
            identity=identity,
            status=status,
            attempt=payload["attempt"],
            failure_stage=failure_stage,
            started_at=payload["started_at"],
            finished_at=payload["finished_at"],
            frame_count=payload["frame_count"],
            window_count=payload["window_count"],
            cache_policy=payload["cache_policy"],
            cache_stats=cache_stats,
            timings=timings,
            diagnostics=_diagnostics_from_payload(payload["diagnostics"]),
            evaluation_kind=payload["evaluation_kind"],
            evaluation_metrics=None if payload["evaluation_metrics"] is None else dict(_require_mapping(payload["evaluation_metrics"], "evaluation_metrics")),
            artifact_manifest_sha256=payload["artifact_manifest_sha256"],
            error=error,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run record payload is invalid") from exc
    _validate_record(record)
    return record


def read_run_record(path: str | Path) -> RunRecord:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("run record JSON is invalid") from exc
    return _record_from_payload(_require_mapping(payload, "run record"))


def load_valid_completed_run(
    path: str | Path,
    expected_seed: RunIdentitySeed,
) -> RunRecord:
    if not isinstance(expected_seed, RunIdentitySeed):
        raise ValueError("expected identity seed is invalid")
    record = read_run_record(path)
    if record.status is not RunStatus.SUCCEEDED or record.identity is None:
        raise ValueError("run is not a completed success")
    expected = complete_identity(expected_seed, record.identity.prediction_key)
    if record.identity_seed != expected_seed or record.identity != expected:
        raise ValueError("run identity does not exactly match expected identity")
    _validate_record(record)
    return record


def _redact_url_userinfo(value: str) -> str:
    return re.sub(
        r"(?i)(\b[a-z][a-z0-9+.-]*://)([^/@\s]+)@",
        r"\1<redacted>@",
        value,
    )


def redact_argv(argv: Sequence[str]) -> tuple[str, ...]:
    redacted: list[str] = []
    hide_next = False
    for raw in argv:
        if not isinstance(raw, str):
            raise ValueError("argv entries must be strings")
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        key = raw.split("=", 1)[0].lower()
        is_url = re.match(r"(?i)^[a-z][a-z0-9+.-]*://", raw) is not None
        if not is_url and any(word in key for word in ("password", "token", "cookie", "credential", "secret")):
            if "=" in raw:
                redacted.append(f"{raw.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(raw)
                hide_next = True
            continue
        redacted.append(_redact_url_userinfo(raw))
    return tuple(redacted)


def _redact_message(message: str) -> str:
    value = _redact_url_userinfo(message)
    return re.sub(
        r"(?i)\b(password|token|cookie|credential|secret)(\s*[:=]\s*|\s+)[^\s,;]+",
        lambda match: f"{match.group(1)}=<redacted>",
        value,
    )


def write_campaign_metadata(path: str | Path, payload: Mapping[str, object]) -> Path:
    if not isinstance(payload, Mapping):
        raise ValueError("campaign metadata must be a mapping")
    prepared = dict(payload)
    if "argv" in prepared:
        argv = prepared["argv"]
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
            raise ValueError("campaign argv must be a sequence")
        prepared["argv"] = list(redact_argv(argv))
    return atomic_json(path, prepared)


def _slice_id(seed: RunIdentitySeed) -> str:
    return f"f{seed.frame_start:06d}-{seed.frame_stop:06d}-s{seed.frame_stride}"


def _identity_axis(record: RunRecord) -> dict[str, object]:
    seed = record.identity_seed
    return {
        "dataset": seed.dataset,
        "scene": seed.scene,
        "slice_id": _slice_id(seed),
        "run_id": record.run_id,
        "segmentation_method": seed.segmentation_method,
        "window_reference_enabled": seed.window_reference_enabled,
    }


def _record_sort_key(record: RunRecord) -> tuple[object, ...]:
    axis = _identity_axis(record)
    return (
        axis["dataset"], axis["scene"], axis["slice_id"],
        _RUN_ORDER.get(record.run_id, 999), record.run_id,
    )


def _csv_cell(value: object) -> str:
    value = _enum_value(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping) or isinstance(value, (tuple, list)):
        return json.dumps(
            _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("CSV values must be finite")
    return str(value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> Path:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})
    return _atomic_text(path, stream.getvalue())


def _timing_rates(record: RunRecord) -> dict[str, float | None]:
    reconstruction = record.timings.reconstruction_s
    evaluation = record.timings.evaluation_s
    return {
        "reconstruction_per_frame_s": None if reconstruction is None or record.frame_count == 0 else reconstruction / record.frame_count,
        "reconstruction_per_window_s": None if reconstruction is None or record.window_count == 0 else reconstruction / record.window_count,
        "evaluation_per_frame_s": None if evaluation is None or record.frame_count == 0 else evaluation / record.frame_count,
        "evaluation_per_window_s": None if evaluation is None or record.window_count == 0 else evaluation / record.window_count,
    }


def _runs_row(record: RunRecord) -> dict[str, object]:
    row = _identity_axis(record)
    rates = _timing_rates(record)
    row.update({
        "status": record.status,
        "attempt": record.attempt,
        "failure_stage": record.failure_stage,
        "frame_count": record.frame_count,
        "window_count": record.window_count,
        "cache_policy": record.cache_policy,
        "ordinary_hits": record.cache_stats.ordinary_hits,
        "ordinary_misses": record.cache_stats.ordinary_misses,
        "corrupt_count": record.cache_stats.corrupt_count,
        "read_ms": record.cache_stats.read_ms,
        "write_ms": record.cache_stats.write_ms,
        "saved_window_count": record.cache_stats.saved_window_count,
        "stored_bytes": record.cache_stats.stored_bytes,
        "reconstruction_s": record.timings.reconstruction_s,
        "evaluation_s": record.timings.evaluation_s,
        **rates,
        "prediction_key": None if record.identity is None else record.identity.prediction_key,
        "artifact_manifest_sha256": record.artifact_manifest_sha256,
    })
    return row


_RUN_COLUMNS = (
    *_IDENTITY_COLUMNS,
    "status", "attempt", "failure_stage", "frame_count", "window_count",
    "cache_policy", "ordinary_hits", "ordinary_misses", "corrupt_count",
    "read_ms", "write_ms", "saved_window_count", "stored_bytes",
    "reconstruction_s", "evaluation_s", "reconstruction_per_frame_s",
    "reconstruction_per_window_s", "evaluation_per_frame_s",
    "evaluation_per_window_s", "prediction_key", "artifact_manifest_sha256",
)


def _distribution_row(row: dict[str, object], name: str, value: DistributionSummary | None) -> None:
    for suffix in ("count", "minimum", "maximum", "mean", "median"):
        row[f"{name}_{suffix}"] = None if value is None else getattr(value, suffix)


_DIAGNOSTIC_DISTRIBUTIONS = (
    "region_count", "keyframe_count", "coverage_ratio", "regions_before",
    "regions_after", "region_reduction_relative",
)
_DIAGNOSTIC_SCALARS = (
    "unique_frame_count", "window_frame_observation_count", "window_count",
    "refinement_enabled", "refinement_state", "region_reduction_absolute_total",
    "candidate_edge_total", "accepted_edge_total", "conflict_edge_total",
    "projected_sample_total", "occluded_sample_total", "depth_rejected_sample_total",
    "applied_frame_count", "applied_frame_rate",
)


def _diagnostics_row(record: RunRecord) -> dict[str, object]:
    row = _identity_axis(record)
    diagnostics = record.diagnostics
    if diagnostics is None:
        for name in _DIAGNOSTIC_SCALARS:
            row[name] = None
        for name in _DIAGNOSTIC_DISTRIBUTIONS:
            _distribution_row(row, name, None)
        row["keyframe_index_histogram"] = None
        row["keyframe_index_records"] = None
        row["fallback_reason_histogram"] = None
        return row
    for name in _DIAGNOSTIC_SCALARS:
        row[name] = getattr(diagnostics, name)
    for name in _DIAGNOSTIC_DISTRIBUTIONS:
        _distribution_row(row, name, getattr(diagnostics, name))
    row["keyframe_index_histogram"] = diagnostics.keyframe_index_histogram
    row["keyframe_index_records"] = diagnostics.keyframe_index_records
    row["fallback_reason_histogram"] = diagnostics.fallback_reason_histogram
    return row


_DIAGNOSTICS_COLUMNS = (
    *_IDENTITY_COLUMNS,
    *_DIAGNOSTIC_SCALARS,
    *(f"{name}_{suffix}" for name in _DIAGNOSTIC_DISTRIBUTIONS for suffix in ("count", "minimum", "maximum", "mean", "median")),
    "keyframe_index_histogram", "keyframe_index_records", "fallback_reason_histogram",
)


def _metric_row(record: RunRecord, metric_names: Sequence[str]) -> dict[str, object]:
    row = _identity_axis(record)
    metrics = record.evaluation_metrics or {}
    row.update({name: metrics.get(name) for name in metric_names})
    return row


def _identity_csv_columns(metric_names: Sequence[str]) -> tuple[str, ...]:
    return (*_IDENTITY_COLUMNS, *metric_names)


def _summary_payload(
    records: Sequence[RunRecord],
    expected_keys: Sequence[tuple[str, str, str, str]],
    missing_keys: Sequence[tuple[str, str, str, str]],
    paths: SummaryPaths,
) -> dict[str, object]:
    completed = [record for record in records if record.status is RunStatus.SUCCEEDED]
    failed = [record for record in records if record.status is RunStatus.FAILED]
    coverage = [
        {
            **_identity_axis(record),
            "status": record.status,
        }
        for record in records
    ]
    coverage.extend({
        "dataset": dataset,
        "scene": scene,
        "slice_id": slice_id,
        "run_id": run_id,
        "status": "missing",
    } for dataset, scene, slice_id, run_id in missing_keys)
    shared: dict[str, str] = {}
    for record in completed:
        assert record.identity is not None
        scene_key = f"{record.identity_seed.dataset}/{record.identity_seed.scene}"
        existing = shared.get(scene_key)
        if existing is not None and existing != record.identity.prediction_key:
            raise ValueError("prediction key disagreement within scene")
        shared[scene_key] = record.identity.prediction_key
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "expected_runs": len(expected_keys),
        "completed_runs": len(completed),
        "failed_runs": len(failed),
        "missing_runs": len(missing_keys),
        "coverage": coverage,
        "shared_prediction_key_by_scene": dict(sorted(shared.items())),
        "files": {
            "runs_csv": paths.runs_csv.name,
            "diagnostics_csv": paths.diagnostics_csv.name,
            "trajectory_csv": paths.trajectory_csv.name,
            "pointcloud_csv": paths.pointcloud_csv.name,
            "summary_json": paths.summary_json.name,
            "failures_json": paths.failures_json.name,
        },
        "historical_pooling": False,
        "no_historical_pooling": True,
    }


def _failure_rows(
    records: Sequence[RunRecord],
    missing_keys: Sequence[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.status is not RunStatus.FAILED:
            continue
        assert record.failure_stage is not None
        assert record.error is not None
        rows.append({
            **_identity_axis(record),
            "attempt": record.attempt,
            "failure_stage": record.failure_stage,
            "error_type": record.error.error_type,
            "message": _redact_message(record.error.message),
        })
    rows.extend({
        "dataset": dataset,
        "scene": scene,
        "slice_id": slice_id,
        "run_id": run_id,
        "attempt": 0,
        "failure_stage": "missing",
        "error_type": "MissingRunRecord",
        "message": "missing run record",
    } for dataset, scene, slice_id, run_id in missing_keys)
    return rows


def write_summaries(
    records: Sequence[RunRecord],
    expected_seeds: Mapping[tuple[str, str, str, str], RunIdentitySeed],
    output_dir: str | Path,
) -> SummaryPaths:
    if not isinstance(expected_seeds, Mapping):
        raise ValueError("expected_seeds must be a mapping")
    expected_keys = tuple(expected_seeds)
    if any(len(key) != 4 for key in expected_keys):
        raise ValueError("expected seed keys must be dataset, scene, slice_id, run_id")
    for seed in expected_seeds.values():
        if not isinstance(seed, RunIdentitySeed):
            raise ValueError("expected seed is invalid")
    by_key: dict[tuple[str, str, str, str], RunRecord] = {}
    for record in records:
        _validate_record(record)
        axis = _identity_axis(record)
        key = (axis["dataset"], axis["scene"], axis["slice_id"], axis["run_id"])
        if key in by_key:
            raise ValueError("duplicate run record")
        if key not in expected_seeds:
            raise ValueError("unexpected run record")
        if record.identity_seed != expected_seeds[key]:
            raise ValueError("run identity seed does not match expected seed")
        by_key[key] = record

    missing_keys = tuple(key for key in expected_keys if key not in by_key)
    ordered_records = tuple(sorted(by_key.values(), key=_record_sort_key))
    completed = tuple(record for record in ordered_records if record.status is RunStatus.SUCCEEDED)
    scene_keys: dict[tuple[str, str], str] = {}
    for record in completed:
        assert record.identity is not None
        scene = (record.identity_seed.dataset, record.identity_seed.scene)
        previous = scene_keys.get(scene)
        if previous is not None and previous != record.identity.prediction_key:
            raise ValueError("prediction key disagreement within scene")
        scene_keys[scene] = record.identity.prediction_key

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = SummaryPaths(
        runs_csv=output / "runs.csv",
        diagnostics_csv=output / "diagnostics.csv",
        trajectory_csv=output / "trajectory.csv",
        pointcloud_csv=output / "pointcloud.csv",
        summary_json=output / "summary.json",
        failures_json=output / "failures.json",
    )
    _write_csv(paths.runs_csv, _RUN_COLUMNS, [_runs_row(record) for record in ordered_records])
    _write_csv(paths.diagnostics_csv, _DIAGNOSTICS_COLUMNS, [_diagnostics_row(record) for record in ordered_records])
    _write_csv(paths.trajectory_csv, _identity_csv_columns(TRAJECTORY_METRICS), [_metric_row(record, TRAJECTORY_METRICS) for record in ordered_records])
    _write_csv(paths.pointcloud_csv, _identity_csv_columns(POINTCLOUD_METRICS), [_metric_row(record, POINTCLOUD_METRICS) for record in ordered_records])
    atomic_json(
        paths.summary_json,
        _summary_payload(ordered_records, expected_keys, missing_keys, paths),
    )
    atomic_json(
        paths.failures_json,
        {"failures": _failure_rows(ordered_records, missing_keys)},
    )
    return paths


__all__ = [
    "CacheStats",
    "FailureStage",
    "POINTCLOUD_METRICS",
    "RunError",
    "RunRecord",
    "RunStatus",
    "RunTimings",
    "SummaryPaths",
    "TRAJECTORY_METRICS",
    "atomic_json",
    "load_valid_completed_run",
    "read_run_record",
    "redact_argv",
    "write_campaign_metadata",
    "write_run_record",
    "write_summaries",
]
