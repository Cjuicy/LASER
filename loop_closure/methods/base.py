from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch

from inference_engine.streaming_window_engine import StreamingWindowEngine
from pipeline.config import LoopMethod


WINDOW_CACHE_SCHEMA_VERSION = 1
Sim3 = tuple[torch.Tensor | float, torch.Tensor, torch.Tensor]


def validate_sim3(transform: Sim3, *, context: str = "Sim(3)") -> None:
    if not isinstance(transform, tuple) or len(transform) != 3:
        raise ValueError(f"{context} must be a (scale, rotation, translation) tuple")
    scale, rotation, translation = transform
    scale_tensor = torch.as_tensor(scale)
    rotation_tensor = torch.as_tensor(rotation)
    translation_tensor = torch.as_tensor(translation)
    if scale_tensor.numel() != 1:
        raise ValueError(f"{context} scale must be scalar")
    scale_value = float(scale_tensor.detach().cpu().item())
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError(f"{context} scale must be finite and positive")
    if tuple(rotation_tensor.shape) != (3, 3):
        raise ValueError(f"{context} rotation must have shape (3, 3)")
    if tuple(translation_tensor.shape) != (3,):
        raise ValueError(f"{context} translation must have shape (3,)")
    if not torch.isfinite(rotation_tensor).all() or not torch.isfinite(
        translation_tensor
    ).all():
        raise ValueError(f"{context} components must be finite")


@dataclass(frozen=True)
class LoopCandidate:
    frame_a: int
    frame_b: int
    similarity: float

    def __post_init__(self) -> None:
        if self.frame_a < 0 or self.frame_b < 0:
            raise ValueError("loop candidate frame indices must be non-negative")
        if self.frame_a <= self.frame_b:
            raise ValueError(
                "loop candidate must use canonical frame_a > frame_b order"
            )
        if (
            not math.isfinite(self.similarity)
            or not -1.0 <= self.similarity <= 1.0
        ):
            raise ValueError("loop candidate similarity must be finite in [-1, 1]")


@dataclass(frozen=True)
class LoopConstraint:
    window_a: int
    window_b: int
    measurement: Sim3
    candidate: LoopCandidate

    def __post_init__(self) -> None:
        if self.window_a < 0 or self.window_b < 0:
            raise ValueError("loop constraint window indices must be non-negative")
        if self.window_a == self.window_b:
            raise ValueError("loop constraint must connect different windows")
        validate_sim3(self.measurement, context="loop constraint Sim(3)")


@dataclass(frozen=True)
class LoopSolution:
    optimized_transforms: tuple[Sim3, ...]
    constraints: tuple[LoopConstraint, ...]
    used_no_loop_path: bool

    def __post_init__(self) -> None:
        for index, transform in enumerate(self.optimized_transforms):
            validate_sim3(
                transform,
                context=f"optimized Sim(3) at index {index}",
            )


@dataclass(frozen=True)
class ReconstructionResult:
    payload: Mapping[str, object]
    summary: Mapping[str, object]


@dataclass
class WindowCache:
    schema_version: int
    loop_method: LoopMethod
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_labels: tuple[np.ndarray, ...]
    anchor_scale_mask: torch.Tensor | None
    loop_state: dict[str, object]
    segmentation_diagnostics: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != WINDOW_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "window cache schema_version must be "
                f"{WINDOW_CACHE_SCHEMA_VERSION}"
            )
        if not isinstance(self.loop_method, LoopMethod):
            try:
                self.loop_method = LoopMethod(self.loop_method)
            except (TypeError, ValueError) as exc:
                raise ValueError("window cache loop_method is invalid") from exc
        if self.window_index < 0:
            raise ValueError("window_index must be non-negative")
        if self.frame_start < 0 or self.frame_end <= self.frame_start:
            raise ValueError(
                "window cache frame range must satisfy "
                "0 <= frame_start < frame_end"
            )
        for field in ("local_points", "camera_poses", "confidence"):
            if not isinstance(getattr(self, field), torch.Tensor):
                raise ValueError(f"window cache {field} must be a tensor")
        if not isinstance(self.segmentation_labels, tuple):
            raise ValueError("segmentation_labels must be a tuple")
        for labels in self.segmentation_labels:
            labels = np.asarray(labels)
            if labels.ndim != 2:
                raise ValueError(
                    "each segmentation label map must be two-dimensional"
                )
        if (
            self.anchor_scale_mask is not None
            and not isinstance(self.anchor_scale_mask, torch.Tensor)
        ):
            raise ValueError("anchor_scale_mask must be a tensor or None")
        if not isinstance(self.loop_state, dict):
            raise ValueError("loop_state must be a dictionary")
        if not isinstance(self.segmentation_diagnostics, tuple):
            raise ValueError("segmentation_diagnostics must be a tuple")
        if not all(
            isinstance(diagnostics, Mapping)
            for diagnostics in self.segmentation_diagnostics
        ):
            raise ValueError(
                "each segmentation diagnostic entry must be a mapping"
            )
        state_tag = self.loop_state.get("tag")
        if state_tag != self.loop_method.value:
            raise ValueError(
                "window cache state tag does not match loop method: "
                f"{state_tag!r} != {self.loop_method.value!r}"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "loop_method": self.loop_method.value,
            "window_index": self.window_index,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "local_points": self.local_points,
            "camera_poses": self.camera_poses,
            "confidence": self.confidence,
            "segmentation_labels": self.segmentation_labels,
            "anchor_scale_mask": self.anchor_scale_mask,
            "loop_state": dict(self.loop_state),
            "segmentation_diagnostics": tuple(
                dict(diagnostics)
                for diagnostics in self.segmentation_diagnostics
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        expected_method: LoopMethod,
    ) -> "WindowCache":
        if not isinstance(payload, Mapping):
            raise ValueError("window cache payload must be a mapping")
        try:
            schema_version = payload["schema_version"]
            raw_method = payload["loop_method"]
        except KeyError as exc:
            raise ValueError(
                f"window cache payload is missing {exc.args[0]}"
            ) from exc
        if schema_version != WINDOW_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "window cache schema_version mismatch: "
                f"{schema_version!r} != {WINDOW_CACHE_SCHEMA_VERSION}"
            )
        try:
            actual_method = LoopMethod(raw_method)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"window cache loop_method is invalid: {raw_method!r}"
            ) from exc
        if actual_method is not expected_method:
            raise ValueError(
                "window cache loop method mismatch: "
                f"{actual_method.value} cache cannot load as "
                f"{expected_method.value}"
            )
        try:
            return cls(
                schema_version=int(schema_version),
                loop_method=actual_method,
                window_index=payload["window_index"],
                frame_start=payload["frame_start"],
                frame_end=payload["frame_end"],
                local_points=payload["local_points"],
                camera_poses=payload["camera_poses"],
                confidence=payload["confidence"],
                segmentation_labels=payload["segmentation_labels"],
                anchor_scale_mask=payload["anchor_scale_mask"],
                loop_state=payload["loop_state"],
                segmentation_diagnostics=payload[
                    "segmentation_diagnostics"
                ],
            )
        except KeyError as exc:
            raise ValueError(
                f"window cache payload is missing {exc.args[0]}"
            ) from exc


class LoopClosureStrategy(Protocol):
    name: LoopMethod

    def create_window_engine(self, **dependencies) -> StreamingWindowEngine:
        raise NotImplementedError

    def build_constraints(
        self,
        caches: Sequence[WindowCache],
        candidates: tuple[LoopCandidate, ...],
    ) -> list[LoopConstraint]:
        raise NotImplementedError

    def optimize(
        self,
        caches: Sequence[WindowCache],
        constraints: Sequence[LoopConstraint],
    ) -> LoopSolution:
        raise NotImplementedError

    def aggregate(
        self,
        caches: Sequence[WindowCache],
        solution: LoopSolution,
    ) -> ReconstructionResult:
        raise NotImplementedError
