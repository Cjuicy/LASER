"""Compact, overlap-aware summaries for reconstruction diagnostics.

The reconstruction artifact stores one segmentation diagnostic per
window/frame observation.  This module deliberately keeps those observations
as observations: an overlapping frame is counted once for every window in
which it was observed, while ``unique_frame_count`` exposes the de-duplicated
frame axis separately.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


_KEYFRAME_TOKEN = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_WINDOW_REFERENCE_FIELDS = (
    "window_reference_applied",
    "window_reference_keyframes",
    "window_reference_keyframe_count",
    "window_reference_is_keyframe",
    "window_reference_coverage_ratio",
    "window_reference_regions_before",
    "window_reference_regions_after",
    "window_reference_candidate_edges",
    "window_reference_accepted_edges",
    "window_reference_conflict_edges",
    "window_reference_projected_samples",
    "window_reference_occluded_samples",
    "window_reference_depth_rejected_samples",
    "window_reference_fallback",
)


def _finite_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> int | float:
    """Validate one scalar without allowing bool/NaN/infinity."""

    if integer:
        if type(value) is not int:
            raise ValueError(f"{field} must be an integer")
        normalized: int | float = value
    else:
        if type(value) not in (int, float):
            raise ValueError(f"{field} must be a finite number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{field} must be finite")
    numeric = float(normalized)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return normalized


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _parse_keyframes(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, str):
        raise ValueError("window_reference_keyframes must be a string")
    if raw == "":
        return ()
    tokens = tuple(raw.split(","))
    if any(not _KEYFRAME_TOKEN.fullmatch(token) for token in tokens):
        raise ValueError("window_reference_keyframes must contain decimal indices")
    return tuple(int(token) for token in tokens)


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float

    def __post_init__(self) -> None:
        _finite_number(self.count, "distribution count", minimum=1, integer=True)
        values = (self.minimum, self.maximum, self.mean, self.median)
        normalized = tuple(
            float(_finite_number(value, "distribution value")) for value in values
        )
        if normalized[0] > normalized[1]:
            raise ValueError("distribution minimum must not exceed maximum")
        object.__setattr__(self, "minimum", normalized[0])
        object.__setattr__(self, "maximum", normalized[1])
        object.__setattr__(self, "mean", normalized[2])
        object.__setattr__(self, "median", normalized[3])


def finite_distribution(
    values: Sequence[int | float],
) -> DistributionSummary | None:
    """Return finite distribution statistics, preserving no mutable input."""

    normalized: list[float] = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError("diagnostic distribution values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("diagnostic distribution values must be finite")
        normalized.append(number)
    if not normalized:
        return None
    return DistributionSummary(
        count=len(normalized),
        minimum=min(normalized),
        maximum=max(normalized),
        mean=statistics.fmean(normalized),
        median=statistics.median(normalized),
    )


@dataclass(frozen=True)
class DiagnosticsSummary:
    unique_frame_count: int
    window_frame_observation_count: int
    window_count: int
    region_count: DistributionSummary
    refinement_enabled: bool
    refinement_state: str
    keyframe_count: DistributionSummary | None
    keyframe_index_histogram: Mapping[str, int] | None
    keyframe_index_records: tuple[str, ...] | None
    coverage_ratio: DistributionSummary | None
    regions_before: DistributionSummary | None
    regions_after: DistributionSummary | None
    region_reduction_absolute_total: int | None
    region_reduction_relative: DistributionSummary | None
    candidate_edge_total: int | None
    accepted_edge_total: int | None
    conflict_edge_total: int | None
    projected_sample_total: int | None
    occluded_sample_total: int | None
    depth_rejected_sample_total: int | None
    applied_frame_count: int | None
    applied_frame_rate: float | None
    fallback_reason_histogram: Mapping[str, int] | None

    def __post_init__(self) -> None:
        for name in (
            "unique_frame_count",
            "window_frame_observation_count",
            "window_count",
        ):
            _finite_number(getattr(self, name), name, minimum=0, integer=True)
        if self.unique_frame_count > self.window_frame_observation_count:
            raise ValueError("unique_frame_count exceeds observations")
        if not isinstance(self.region_count, DistributionSummary):
            raise ValueError("region_count must be a DistributionSummary")
        _validate_distribution_semantics(
            self.region_count,
            "region_count",
            expected_count=self.window_frame_observation_count,
            integer_bounds=True,
            minimum=0,
        )
        if type(self.refinement_enabled) is not bool:
            raise ValueError("refinement_enabled must be a boolean")
        if self.refinement_state not in {"not_applicable", "fallback", "applied"}:
            raise ValueError("refinement_state is invalid")

        distribution_names = (
            "keyframe_count",
            "coverage_ratio",
            "regions_before",
            "regions_after",
            "region_reduction_relative",
        )
        for name in distribution_names:
            value = getattr(self, name)
            if value is not None and not isinstance(value, DistributionSummary):
                raise ValueError(f"{name} must be a DistributionSummary or None")

        if self.refinement_enabled:
            if self.refinement_state not in {"fallback", "applied"}:
                raise ValueError("enabled refinement must be applied or fallback")
            if any(getattr(self, name) is None for name in distribution_names):
                raise ValueError("enabled refinement diagnostics are incomplete")
            if self.keyframe_index_histogram is None:
                raise ValueError("enabled refinement keyframe histogram is required")
            if self.keyframe_index_records is None:
                raise ValueError("enabled refinement keyframe records are required")
            if self.fallback_reason_histogram is None:
                raise ValueError("enabled refinement fallback histogram is required")
            for name in (
                "region_reduction_absolute_total",
                "candidate_edge_total",
                "accepted_edge_total",
                "conflict_edge_total",
                "projected_sample_total",
                "occluded_sample_total",
                "depth_rejected_sample_total",
                "applied_frame_count",
                "applied_frame_rate",
            ):
                if getattr(self, name) is None:
                    raise ValueError(f"enabled refinement field {name} is required")
            _finite_number(
                self.applied_frame_rate,
                "applied_frame_rate",
                minimum=0,
                maximum=1,
            )
        else:
            if self.refinement_state != "not_applicable":
                raise ValueError("disabled refinement must be not_applicable")
            for name in (
                *distribution_names,
                "region_reduction_absolute_total",
                "candidate_edge_total",
                "accepted_edge_total",
                "conflict_edge_total",
                "projected_sample_total",
                "occluded_sample_total",
                "depth_rejected_sample_total",
                "applied_frame_count",
                "applied_frame_rate",
                "fallback_reason_histogram",
            ):
                if getattr(self, name) is not None:
                    raise ValueError("disabled refinement fields must be None")
            if self.keyframe_index_histogram is not None:
                raise ValueError("disabled refinement fields must be None")
            if self.keyframe_index_records is not None:
                raise ValueError("disabled refinement fields must be None")

        if self.keyframe_index_histogram is not None:
            if not isinstance(self.keyframe_index_histogram, Mapping):
                raise ValueError("keyframe index histogram must be a mapping")
            object.__setattr__(
                self,
                "keyframe_index_histogram",
                MappingProxyType(
                    {
                        key: int(
                            _validate_histogram_item(
                                key, value, "keyframe_index_histogram"
                            )
                        )
                        for key, value in self.keyframe_index_histogram.items()
                    }
                ),
            )
        if self.keyframe_index_records is not None:
            if type(self.keyframe_index_records) is not tuple:
                raise ValueError("keyframe index records must be a tuple")
            if any(not isinstance(item, str) for item in self.keyframe_index_records):
                raise ValueError("keyframe index records must be strings")
            if any(_parse_keyframes(item) is None for item in self.keyframe_index_records):
                raise ValueError("keyframe index records are invalid")
            object.__setattr__(self, "keyframe_index_records", tuple(self.keyframe_index_records))
        if self.fallback_reason_histogram is not None:
            if not isinstance(self.fallback_reason_histogram, Mapping):
                raise ValueError("fallback reason histogram must be a mapping")
            object.__setattr__(
                self,
                "fallback_reason_histogram",
                MappingProxyType(
                    {
                        key: int(
                            _validate_histogram_item(
                                key,
                                value,
                                "fallback_reason_histogram",
                                require_nonempty=True,
                            )
                        )
                        for key, value in self.fallback_reason_histogram.items()
                    }
                ),
            )

        for name in (
            "region_reduction_absolute_total",
            "candidate_edge_total",
            "accepted_edge_total",
            "conflict_edge_total",
            "projected_sample_total",
            "occluded_sample_total",
            "depth_rejected_sample_total",
            "applied_frame_count",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name, minimum=0, integer=True)
        if self.applied_frame_rate is not None:
            normalized = float(
                _finite_number(
                    self.applied_frame_rate,
                    "applied_frame_rate",
                    minimum=0,
                    maximum=1,
                )
            )
            object.__setattr__(self, "applied_frame_rate", normalized)

        if self.refinement_enabled:
            assert self.keyframe_count is not None
            assert self.coverage_ratio is not None
            assert self.regions_before is not None
            assert self.regions_after is not None
            assert self.region_reduction_relative is not None
            assert self.keyframe_index_histogram is not None
            assert self.keyframe_index_records is not None
            assert self.fallback_reason_histogram is not None
            _validate_distribution_semantics(
                self.keyframe_count,
                "keyframe_count",
                expected_count=self.window_frame_observation_count,
                integer_bounds=True,
                minimum=0,
            )
            _validate_distribution_semantics(
                self.coverage_ratio,
                "coverage_ratio",
                expected_count=self.window_frame_observation_count,
                minimum=0,
                maximum=1,
            )
            _validate_distribution_semantics(
                self.regions_before,
                "regions_before",
                expected_count=self.window_frame_observation_count,
                integer_bounds=True,
                minimum=0,
            )
            _validate_distribution_semantics(
                self.regions_after,
                "regions_after",
                expected_count=self.window_frame_observation_count,
                integer_bounds=True,
                minimum=0,
            )
            _validate_distribution_semantics(
                self.region_reduction_relative,
                "region_reduction_relative",
                expected_count=self.window_frame_observation_count,
                minimum=0,
                maximum=1,
            )
            if len(self.keyframe_index_records) != self.window_frame_observation_count:
                raise ValueError("keyframe index records count is invalid")
            histogram_from_records: dict[str, int] = {}
            for record in self.keyframe_index_records:
                for index in _parse_keyframes(record):
                    key = str(index)
                    histogram_from_records[key] = histogram_from_records.get(key, 0) + 1
            if dict(self.keyframe_index_histogram) != histogram_from_records:
                raise ValueError("keyframe index histogram does not match records")
            if sum(self.fallback_reason_histogram.values()) > self.window_frame_observation_count:
                raise ValueError("fallback histogram count is invalid")
            assert self.applied_frame_count is not None
            if self.applied_frame_count > self.window_frame_observation_count:
                raise ValueError("applied_frame_count exceeds observations")


def _validate_distribution_semantics(
    value: DistributionSummary,
    field: str,
    *,
    expected_count: int,
    minimum: float | None = None,
    maximum: float | None = None,
    integer_bounds: bool = False,
) -> None:
    if value.count != expected_count:
        raise ValueError(f"{field} distribution count is invalid")
    if value.minimum > value.mean or value.mean > value.maximum:
        raise ValueError(f"{field} mean is outside its bounds")
    if value.minimum > value.median or value.median > value.maximum:
        raise ValueError(f"{field} median is outside its bounds")
    for name in ("minimum", "maximum", "mean", "median"):
        scalar = getattr(value, name)
        if minimum is not None and scalar < minimum:
            raise ValueError(f"{field} {name} must be >= {minimum}")
        if maximum is not None and scalar > maximum:
            raise ValueError(f"{field} {name} must be <= {maximum}")
    if integer_bounds and (
        not value.minimum.is_integer() or not value.maximum.is_integer()
    ):
        raise ValueError(f"{field} bounds must preserve integer semantics")


def _validate_histogram_item(
    key: object,
    value: object,
    field: str,
    *,
    require_nonempty: bool = False,
) -> int:
    if not isinstance(key, str):
        raise ValueError(f"{field} keys must be strings")
    if require_nonempty:
        if not key:
            raise ValueError(f"{field} keys must not be empty")
    elif _KEYFRAME_TOKEN.fullmatch(key) is None:
        raise ValueError(f"{field} keys must be decimal indices")
    return int(_finite_number(value, f"{field} value", minimum=0, integer=True))


def _integer(mapping: Mapping[str, object], key: str, *, minimum: int = 0) -> int:
    if key not in mapping:
        raise ValueError(f"diagnostic observation is missing {key}")
    return int(_finite_number(mapping[key], key, minimum=minimum, integer=True))


def _number(
    mapping: Mapping[str, object],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if key not in mapping:
        raise ValueError(f"diagnostic observation is missing {key}")
    return float(
        _finite_number(mapping[key], key, minimum=minimum, maximum=maximum)
    )


def _float_number(
    mapping: Mapping[str, object],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if key not in mapping:
        raise ValueError(f"diagnostic observation is missing {key}")
    if type(mapping[key]) is not float:
        raise ValueError(f"{key} must be a float")
    return float(
        _finite_number(mapping[key], key, minimum=minimum, maximum=maximum)
    )


def _validated_observations(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("segmentation_summaries must be a sequence")
    observations = tuple(value)
    if not observations:
        raise ValueError("diagnostics require at least one segmentation observation")
    if any(not isinstance(item, Mapping) for item in observations):
        raise ValueError("segmentation observations must be mappings")
    return observations  # type: ignore[return-value]


def _validate_payload_headers(payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if not isinstance(payload, Mapping):
        raise ValueError("diagnostics payload must be a mapping")
    timings = payload.get("stage_timings_ms")
    if not isinstance(timings, Mapping):
        raise ValueError("stage_timings_ms must be a mapping")
    for key, value in timings.items():
        if not isinstance(key, str):
            raise ValueError("stage timing names must be strings")
        _finite_number(value, f"stage_timings_ms.{key}", minimum=0)
    mode_scalars = payload.get("mode_scalars")
    if not isinstance(mode_scalars, Mapping):
        raise ValueError("mode_scalars must be a mapping")
    if "window_count" not in mode_scalars:
        raise ValueError("mode_scalars.window_count is required")
    _finite_number(mode_scalars["window_count"], "mode_scalars.window_count", minimum=0, integer=True)
    return _validated_observations(payload.get("segmentation_summaries"))


def _aggregate_enabled(
    common: dict[str, object],
    observations: tuple[Mapping[str, object], ...],
) -> DiagnosticsSummary:
    keyframe_counts: list[int] = []
    keyframe_index_histogram: dict[str, int] = {}
    keyframe_index_records: list[str] = []
    coverage: list[float] = []
    regions_before: list[int] = []
    regions_after: list[int] = []
    reductions: list[float] = []
    fallback_histogram: dict[str, int] = {}
    total_fields = {
        "candidate_edge_total": "window_reference_candidate_edges",
        "accepted_edge_total": "window_reference_accepted_edges",
        "conflict_edge_total": "window_reference_conflict_edges",
        "projected_sample_total": "window_reference_projected_samples",
        "occluded_sample_total": "window_reference_occluded_samples",
        "depth_rejected_sample_total": "window_reference_depth_rejected_samples",
    }
    totals = {name: 0 for name in total_fields}
    applied_count = 0
    for observation in observations:
        missing = [key for key in _WINDOW_REFERENCE_FIELDS if key not in observation]
        if missing:
            raise ValueError(f"enabled diagnostics missing {missing[0]}")
        if type(observation["window_reference_applied"]) is not bool:
            raise ValueError("window_reference_applied must be a boolean")
        if type(observation["window_reference_is_keyframe"]) is not bool:
            raise ValueError("window_reference_is_keyframe must be a boolean")
        if not isinstance(observation["window_reference_fallback"], str):
            raise ValueError("window_reference_fallback must be a string")
        fallback = observation["window_reference_fallback"]
        if fallback == "":
            raise ValueError("window_reference_fallback must not be empty")
        keyframes = _parse_keyframes(observation["window_reference_keyframes"])
        keyframe_count = _integer(
            observation,
            "window_reference_keyframe_count",
            minimum=0,
        )
        if keyframe_count != len(keyframes):
            raise ValueError("window_reference_keyframe_count disagrees with keyframes")
        keyframe_counts.append(keyframe_count)
        raw_keyframes = observation["window_reference_keyframes"]
        keyframe_index_records.append(raw_keyframes)  # type: ignore[arg-type]
        for index in keyframes:
            label = str(index)
            keyframe_index_histogram[label] = keyframe_index_histogram.get(label, 0) + 1
        coverage.append(
            _float_number(
                observation,
                "window_reference_coverage_ratio",
                minimum=0,
                maximum=1,
            )
        )
        before = _integer(observation, "window_reference_regions_before", minimum=0)
        after = _integer(observation, "window_reference_regions_after", minimum=0)
        if after > before:
            raise ValueError("window_reference_regions_after must not exceed before")
        regions_before.append(before)
        regions_after.append(after)
        reductions.append(0.0 if before == 0 else (before - after) / before)
        for name, field in total_fields.items():
            totals[name] += _integer(observation, field, minimum=0)
        if observation["window_reference_applied"]:
            applied_count += 1
        if not observation["window_reference_is_keyframe"]:
            fallback_histogram[fallback] = fallback_histogram.get(fallback, 0) + 1

    observation_count = len(observations)
    return DiagnosticsSummary(
        **common,
        refinement_state="applied" if applied_count else "fallback",
        keyframe_count=finite_distribution(keyframe_counts),
        keyframe_index_histogram=dict(sorted(keyframe_index_histogram.items())),
        keyframe_index_records=tuple(keyframe_index_records),
        coverage_ratio=finite_distribution(coverage),
        regions_before=finite_distribution(regions_before),
        regions_after=finite_distribution(regions_after),
        region_reduction_absolute_total=sum(
            before - after for before, after in zip(regions_before, regions_after, strict=True)
        ),
        region_reduction_relative=finite_distribution(reductions),
        **totals,
        applied_frame_count=applied_count,
        applied_frame_rate=applied_count / observation_count,
        fallback_reason_histogram=dict(sorted(fallback_histogram.items())),
    )


def aggregate_diagnostics(
    payload: Mapping[str, object],
    *,
    refinement_enabled: bool,
) -> DiagnosticsSummary:
    """Aggregate artifact diagnostics without mutating the artifact payload."""

    if type(refinement_enabled) is not bool:
        raise ValueError("refinement_enabled must be a boolean")
    observations = _validate_payload_headers(payload)
    frame_ids = tuple(_integer(item, "frame_index") for item in observations)
    windows = {_integer(item, "window_index") for item in observations}
    regions = tuple(_integer(item, "region_count") for item in observations)
    region_distribution = finite_distribution(regions)
    if region_distribution is None:
        raise ValueError("diagnostics require at least one region observation")
    mode_scalars = payload["mode_scalars"]
    assert isinstance(mode_scalars, Mapping)
    window_count = int(mode_scalars["window_count"])
    if window_count < len(windows):
        raise ValueError("mode_scalars.window_count is smaller than observed windows")
    common = {
        "unique_frame_count": len(set(frame_ids)),
        "window_frame_observation_count": len(observations),
        "window_count": window_count,
        "region_count": region_distribution,
        "refinement_enabled": refinement_enabled,
    }
    if not refinement_enabled:
        return DiagnosticsSummary(
            **common,
            refinement_state="not_applicable",
            keyframe_count=None,
            keyframe_index_histogram=None,
            keyframe_index_records=None,
            coverage_ratio=None,
            regions_before=None,
            regions_after=None,
            region_reduction_absolute_total=None,
            region_reduction_relative=None,
            candidate_edge_total=None,
            accepted_edge_total=None,
            conflict_edge_total=None,
            projected_sample_total=None,
            occluded_sample_total=None,
            depth_rejected_sample_total=None,
            applied_frame_count=None,
            applied_frame_rate=None,
            fallback_reason_histogram=None,
        )
    return _aggregate_enabled(common, observations)


__all__ = [
    "DiagnosticsSummary",
    "DistributionSummary",
    "aggregate_diagnostics",
    "finite_distribution",
]
