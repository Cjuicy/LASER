from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import (
    AtomicSplitMode,
    DatasetKind,
    LoadedCampaignConfig,
    ModelName,
    ReconstructionMode,
    SegmentationMethod,
    _CANONICAL_METHODS,
    _CANONICAL_REFINEMENTS,
    _WINDOW_REFERENCE_PAYLOAD,
    _canonical_json_bytes,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else value  # type: ignore[no-any-return]


def _require_sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{path} must be SHA256 hex")
    return value


@dataclass(frozen=True)
class RunVariant:
    segmentation_method: SegmentationMethod
    window_reference_enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.segmentation_method, SegmentationMethod):
            raise ValueError("run variant segmentation method is invalid")
        if type(self.window_reference_enabled) is not bool:
            raise ValueError("run variant refinement must be a boolean")

    @property
    def run_id(self) -> str:
        state = "on" if self.window_reference_enabled else "off"
        return f"{self.segmentation_method.value}__wr-{state}"


@dataclass(frozen=True)
class PlannedRun:
    scene_id: str
    dataset: DatasetKind
    scene: str
    slice_id: str
    variant: RunVariant
    relative_run_dir: Path

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("planned scene_id must be a non-empty string")
        if not isinstance(self.dataset, DatasetKind):
            raise ValueError("planned dataset is invalid")
        if not isinstance(self.scene, str) or not self.scene:
            raise ValueError("planned scene must be a non-empty string")
        if not isinstance(self.slice_id, str) or not self.slice_id:
            raise ValueError("planned slice_id must be a non-empty string")
        if not isinstance(self.variant, RunVariant):
            raise ValueError("planned variant is invalid")
        if not isinstance(self.relative_run_dir, Path):
            raise ValueError("planned relative_run_dir must be a path")


@dataclass(frozen=True)
class CampaignPlan:
    campaign_id: str
    preset: str
    runs: tuple[PlannedRun, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id:
            raise ValueError("plan campaign_id must be a non-empty string")
        if not isinstance(self.preset, str) or not self.preset:
            raise ValueError("plan preset must be a non-empty string")
        if not isinstance(self.runs, tuple):
            raise ValueError("plan runs must be a tuple")
        if any(not isinstance(item, PlannedRun) for item in self.runs):
            raise ValueError("plan contains an invalid run")


@dataclass(frozen=True)
class RunIdentitySeed:
    schema_version: int
    campaign_config_sha256: str
    source_commit: str
    source_dirty: bool
    dataset: str
    scene: str
    frame_start: int
    frame_stop: int
    frame_stride: int
    staged_manifest_sha256: str
    segmentation_method: str
    atomic_split_mode: str
    window_reference_enabled: bool
    window_reference_config: Mapping[str, int | float]
    reconstruction_mode: str
    window_size: int
    overlap: int
    model_name: str
    model_dtype: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("identity schema_version must be 1")
        _require_sha256(self.campaign_config_sha256, "campaign_config_sha256")
        _require_sha256(self.staged_manifest_sha256, "staged_manifest_sha256")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if not isinstance(self.source_commit, str) or (
            self.source_commit != "unknown"
            and _COMMIT_RE.fullmatch(self.source_commit) is None
        ):
            raise ValueError("source_commit must be a 40-character commit or unknown")
        if type(self.source_dirty) is not bool:
            raise ValueError("source_dirty must be a boolean")
        if _enum_value(self.dataset) not in {item.value for item in DatasetKind}:
            raise ValueError("identity dataset is invalid")
        if not isinstance(self.scene, str) or not self.scene:
            raise ValueError("identity scene must be a non-empty string")
        if type(self.frame_start) is not int or self.frame_start < 0:
            raise ValueError("frame_start must be a non-negative integer")
        if type(self.frame_stop) is not int or self.frame_stop <= self.frame_start:
            raise ValueError("frame_stop must be greater than frame_start")
        if type(self.frame_stride) is not int or self.frame_stride < 1:
            raise ValueError("frame_stride must be a positive integer")
        if _enum_value(self.segmentation_method) not in {
            item.value for item in _CANONICAL_METHODS
        }:
            raise ValueError("identity segmentation_method is invalid")
        if _enum_value(self.atomic_split_mode) != AtomicSplitMode.CONSERVATIVE.value:
            raise ValueError("identity atomic_split_mode must be conservative")
        if type(self.window_reference_enabled) is not bool:
            raise ValueError("window_reference_enabled must be a boolean")
        if not isinstance(self.window_reference_config, Mapping):
            raise ValueError("window_reference_config must be a mapping")
        if set(self.window_reference_config) != set(_WINDOW_REFERENCE_PAYLOAD):
            raise ValueError("window_reference_config must contain all ten keys")
        for key, expected in _WINDOW_REFERENCE_PAYLOAD.items():
            value = self.window_reference_config[key]
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"window_reference_config.{key} is not the approved value")
        if _enum_value(self.reconstruction_mode) != ReconstructionMode.NO_LOOP.value:
            raise ValueError("identity reconstruction_mode must be no_loop")
        if type(self.window_size) is not int or self.window_size != 75:
            raise ValueError("identity window_size must be 75")
        if type(self.overlap) is not int or self.overlap != 30:
            raise ValueError("identity overlap must be 30")
        if _enum_value(self.model_name) != ModelName.PI3.value:
            raise ValueError("identity model_name must be pi3")
        if self.model_dtype != "bfloat16":
            raise ValueError("identity model_dtype must be bfloat16")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_config_sha256": self.campaign_config_sha256,
            "source_commit": self.source_commit,
            "source_dirty": self.source_dirty,
            "dataset": _enum_value(self.dataset),
            "scene": self.scene,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "frame_stride": self.frame_stride,
            "staged_manifest_sha256": self.staged_manifest_sha256,
            "segmentation_method": _enum_value(self.segmentation_method),
            "atomic_split_mode": _enum_value(self.atomic_split_mode),
            "window_reference_enabled": self.window_reference_enabled,
            "window_reference_config": dict(self.window_reference_config),
            "reconstruction_mode": _enum_value(self.reconstruction_mode),
            "window_size": self.window_size,
            "overlap": self.overlap,
            "model_name": _enum_value(self.model_name),
            "model_dtype": self.model_dtype,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True)
class RunIdentity(RunIdentitySeed):
    prediction_key: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_sha256(self.prediction_key, "prediction_key")

    def to_payload(self) -> dict[str, object]:
        payload = super().to_payload()
        payload["prediction_key"] = self.prediction_key
        return payload


def complete_identity(seed: RunIdentitySeed, prediction_key: str) -> RunIdentity:
    if not isinstance(seed, RunIdentitySeed):
        raise ValueError("identity completion requires RunIdentitySeed")
    _require_sha256(prediction_key, "prediction_key")
    return RunIdentity(**seed.to_payload(), prediction_key=prediction_key)


def identity_sha256(identity: RunIdentity) -> str:
    if not isinstance(identity, RunIdentity):
        raise ValueError("identity digest requires RunIdentity")
    return hashlib.sha256(_canonical_json_bytes(identity.to_payload())).hexdigest()


def expand_matrix(
    methods: tuple[SegmentationMethod, ...] = (),
    refinements: tuple[bool, ...] = (),
) -> tuple[RunVariant, ...]:
    selected_methods = methods or _CANONICAL_METHODS
    selected_refinements = refinements or _CANONICAL_REFINEMENTS
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("matrix methods must be unique")
    if any(not isinstance(value, SegmentationMethod) for value in selected_methods):
        raise ValueError("matrix method is not supported")
    if any(type(value) is not bool for value in selected_refinements):
        raise ValueError("matrix refinement values must be booleans")
    if len(set(selected_refinements)) != len(selected_refinements):
        raise ValueError("matrix refinement values must be unique")
    if any(value not in _CANONICAL_METHODS for value in selected_methods):
        raise ValueError("matrix method is not supported")
    return tuple(
        RunVariant(method, enabled)
        for method in _CANONICAL_METHODS
        if method in selected_methods
        for enabled in _CANONICAL_REFINEMENTS
        if enabled in selected_refinements
    )


def _slice_id(start: int, stop: int | None, stride: int) -> str:
    stop_label = "end" if stop is None else f"{stop:06d}"
    return f"f{start:06d}-{stop_label}-s{stride}"


def build_plan(loaded: LoadedCampaignConfig) -> CampaignPlan:
    if not isinstance(loaded, LoadedCampaignConfig):
        raise ValueError("plan requires LoadedCampaignConfig")
    config = loaded.config
    variants = expand_matrix(config.matrix.methods, config.matrix.refinement)
    runs: list[PlannedRun] = []
    for selected in config.selected_scenes:
        scene_config = config.scenes[selected.scene_id]
        slice_id = _slice_id(selected.start, selected.stop, selected.stride)
        for variant in variants:
            relative = (
                Path("runs")
                / scene_config.dataset.value
                / slice_id
                / variant.run_id
            )
            runs.append(
                PlannedRun(
                    scene_id=scene_config.scene_id,
                    dataset=scene_config.dataset,
                    scene=scene_config.scene,
                    slice_id=slice_id,
                    variant=variant,
                    relative_run_dir=relative,
                )
            )
    return CampaignPlan(config.campaign_id, config.selected_preset, tuple(runs))


def plan_payload(
    plan: CampaignPlan,
    loaded: LoadedCampaignConfig,
) -> dict[str, object]:
    if not isinstance(plan, CampaignPlan) or not isinstance(loaded, LoadedCampaignConfig):
        raise ValueError("plan payload requires a campaign plan and loaded config")
    config = loaded.config
    return {
        "campaign_id": plan.campaign_id,
        "preset": plan.preset,
        "config_sha256": loaded.sha256,
        "protocol": {
            "reconstruction_mode": config.matrix.reconstruction_mode.value,
            "window_size": config.matrix.window_size,
            "overlap": config.matrix.overlap,
            "atomic_split_mode": config.matrix.atomic_split_mode.value,
            "input_sample_stride": 1,
        },
        "window_reference": config.window_reference.to_payload(),
        "runs": [
            {
                "scene_id": run.scene_id,
                "dataset": run.dataset.value,
                "scene": run.scene,
                "slice_id": run.slice_id,
                "run_id": run.variant.run_id,
                "segmentation_method": run.variant.segmentation_method.value,
                "window_reference_enabled": run.variant.window_reference_enabled,
                "relative_run_dir": str(run.relative_run_dir),
            }
            for run in plan.runs
        ],
    }


__all__ = [
    "CampaignPlan",
    "PlannedRun",
    "RunIdentity",
    "RunIdentitySeed",
    "RunVariant",
    "build_plan",
    "complete_identity",
    "expand_matrix",
    "identity_sha256",
    "plan_payload",
]
