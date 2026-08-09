from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


PREDICTION_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class WindowSpec:
    index: int
    frame_start: int
    frame_end: int

    def __post_init__(self) -> None:
        for field_name in ("index", "frame_start", "frame_end"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"WindowSpec {field_name} must be an integer")
        if self.index < 0:
            raise ValueError("WindowSpec index must be non-negative")
        if self.frame_start < 0 or self.frame_end <= self.frame_start:
            raise ValueError(
                "WindowSpec range must satisfy "
                "0 <= frame_start < frame_end"
            )

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start

    def to_payload(self) -> dict[str, int]:
        return {
            "index": self.index,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
        }

    @classmethod
    def from_payload(cls, payload) -> "WindowSpec":
        if not isinstance(payload, dict):
            raise ValueError("WindowSpec payload must be a dictionary")
        try:
            return cls(
                index=payload["index"],
                frame_start=payload["frame_start"],
                frame_end=payload["frame_end"],
            )
        except KeyError as exc:
            raise ValueError(
                f"WindowSpec payload is missing {exc.args[0]}"
            ) from exc


def build_window_specs(
    frame_count: int,
    window_size: int,
    overlap: int,
) -> tuple[WindowSpec, ...]:
    for field_name, value in (
        ("frame_count", frame_count),
        ("window_size", window_size),
        ("overlap", overlap),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
    if frame_count <= overlap:
        raise ValueError("frame_count must be greater than overlap")
    if window_size <= overlap or overlap < 1:
        raise ValueError(
            "window_size must be greater than overlap >= 1"
        )

    step = window_size - overlap
    specs = []
    for frame_start in range(0, frame_count, step):
        if frame_start != 0 and frame_count - frame_start <= overlap:
            continue
        specs.append(
            WindowSpec(
                index=len(specs),
                frame_start=frame_start,
                frame_end=min(frame_start + window_size, frame_count),
            )
        )
    return tuple(specs)


def validate_window_specs(
    specs: Sequence[WindowSpec],
    *,
    frame_count: int,
    window_size: int,
    overlap: int,
) -> tuple[WindowSpec, ...]:
    normalized = tuple(specs)
    expected = build_window_specs(frame_count, window_size, overlap)
    if normalized != expected:
        raise ValueError(
            "window specs do not match the canonical sliding schedule"
        )
    return normalized


def _validate_artifact_tensor(
    field_name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{field_name} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{field_name} must be stored on CPU")
    if not value.is_floating_point():
        raise ValueError(f"{field_name} must use a floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{field_name} must contain only finite values")


@dataclass(frozen=True)
class OrdinaryWindowArtifact:
    spec: WindowSpec
    depth: torch.Tensor
    confidence: torch.Tensor
    camera_poses: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.spec, WindowSpec):
            raise ValueError("ordinary window artifact spec is invalid")
        for field_name in ("depth", "confidence", "camera_poses"):
            _validate_artifact_tensor(
                field_name,
                getattr(self, field_name),
            )
        if self.depth.ndim != 3:
            raise ValueError("depth must have shape (N,H,W)")
        expected_dense_shape = (
            self.spec.frame_count,
            self.depth.shape[1],
            self.depth.shape[2],
        )
        if tuple(self.depth.shape) != expected_dense_shape:
            raise ValueError(
                f"depth shape must be {expected_dense_shape}"
            )
        if tuple(self.confidence.shape) != expected_dense_shape:
            raise ValueError(
                "confidence shape must match depth shape "
                f"{expected_dense_shape}"
            )
        expected_pose_shape = (self.spec.frame_count, 4, 4)
        if tuple(self.camera_poses.shape) != expected_pose_shape:
            raise ValueError(
                f"camera_poses shape must be {expected_pose_shape}"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_payload(),
            "depth": self.depth,
            "confidence": self.confidence,
            "camera_poses": self.camera_poses,
        }

    @classmethod
    def from_payload(cls, payload) -> "OrdinaryWindowArtifact":
        if not isinstance(payload, dict):
            raise ValueError(
                "ordinary window artifact payload must be a dictionary"
            )
        try:
            return cls(
                spec=WindowSpec.from_payload(payload["spec"]),
                depth=payload["depth"],
                confidence=payload["confidence"],
                camera_poses=payload["camera_poses"],
            )
        except KeyError as exc:
            raise ValueError(
                "ordinary window artifact payload is missing "
                f"{exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class SequenceArtifact:
    reference_intrinsic: torch.Tensor

    def __post_init__(self) -> None:
        _validate_artifact_tensor(
            "reference_intrinsic",
            self.reference_intrinsic,
        )
        if tuple(self.reference_intrinsic.shape) != (3, 3):
            raise ValueError(
                "reference_intrinsic must have shape (3,3)"
            )
