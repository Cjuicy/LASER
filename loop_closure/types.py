from __future__ import annotations

import math
from dataclasses import dataclass

import torch


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
        if not math.isfinite(self.similarity) or not -1.0 <= self.similarity <= 1.0:
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
            validate_sim3(transform, context=f"optimized Sim(3) at index {index}")
