from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from inference_engine.utils.geometry import (
    accumulate_sim3,
    apply_sim3_to_pose,
    homogenize_points,
)
from loop_closure.evidence import LoopEvidenceProvider, LoopWindow
from loop_closure.types import (
    LoopCandidate,
    LoopConstraint,
    LoopSolution,
    Sim3,
    validate_sim3,
)
from loop_closure.utils.sim3loop import Sim3LoopOptimizer
from pipeline.config import OptimizerConfig, ReconstructionMode


logger = logging.getLogger(__name__)


def identity_sim3(device: str | torch.device = "cpu") -> Sim3:
    return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)


def compute_sim3_ab(transform_a: Sim3, transform_b: Sim3) -> Sim3:
    """Preserve LASER's baseline common-frame relative Sim(3) formula."""

    scale_a, rotation_a, translation_a = transform_a
    scale_b, rotation_b, translation_b = transform_b
    scale_ab = scale_b / scale_a
    rotation_ab = rotation_b @ rotation_a.T
    translation_ab = translation_b - scale_ab * (rotation_ab @ translation_a)
    return scale_ab, rotation_ab, translation_ab


class TraditionalWindow(LoopWindow, Protocol):
    anchor_scale_mask: torch.Tensor | None
    relative_sim3: Sim3


@dataclass(frozen=True)
class TraditionalAggregation:
    local_points: torch.Tensor
    global_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    mode_scalars: Mapping[str, int | float]


class TraditionalLoopProcessor:
    """Pure deferred Traditional constraint, optimization, and aggregation math."""

    name = ReconstructionMode.TRADITIONAL

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        *,
        optimizer: Sim3LoopOptimizer | None = None,
        apply_pose_sim3: Callable = apply_sim3_to_pose,
    ) -> None:
        self.optimizer_config = optimizer_config
        self.optimizer = optimizer or Sim3LoopOptimizer(
            optimizer_config,
            device="cpu",
        )
        self._apply_pose_sim3 = apply_pose_sim3

    @staticmethod
    def _window_for_frame(
        states: Sequence[TraditionalWindow],
        frame: int,
    ) -> TraditionalWindow:
        matches = [
            state
            for state in states
            if state.frame_start <= frame < state.frame_end
        ]
        if not matches:
            raise ValueError(f"loop candidate frame {frame} is not cached")
        return max(matches, key=lambda state: state.frame_start)

    def build_constraints(
        self,
        states: Sequence[TraditionalWindow],
        candidates: tuple[LoopCandidate, ...],
        evidence: LoopEvidenceProvider,
    ) -> list[LoopConstraint]:
        constraints: list[LoopConstraint] = []
        seen_pairs: set[tuple[int, int]] = set()
        for candidate in candidates:
            try:
                state_a = self._window_for_frame(states, candidate.frame_a)
                state_b = self._window_for_frame(states, candidate.frame_b)
            except ValueError as error:
                logger.warning(
                    "Skipping traditional loop candidate frames=%s->%s: %s",
                    candidate.frame_a,
                    candidate.frame_b,
                    error,
                )
                continue
            pair = (state_a.window_index, state_b.window_index)
            if pair[0] == pair[1] or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            try:
                alignment_a, alignment_b = evidence.estimate(
                    state_a,
                    state_b,
                    candidate,
                )
                measurement = compute_sim3_ab(alignment_a, alignment_b)
                validate_sim3(measurement, context="traditional loop measurement")
            except ValueError as error:
                logger.warning(
                    "Skipping traditional loop candidate frames=%s->%s: %s",
                    candidate.frame_a,
                    candidate.frame_b,
                    error,
                )
                continue
            constraints.append(
                LoopConstraint(
                    window_a=pair[0],
                    window_b=pair[1],
                    measurement=measurement,
                    candidate=candidate,
                )
            )
        return constraints

    def optimize(
        self,
        states: Sequence[TraditionalWindow],
        constraints: Sequence[LoopConstraint],
    ) -> LoopSolution:
        original = tuple(state.relative_sim3 for state in states)
        for index, transform in enumerate(original):
            validate_sim3(transform, context=f"traditional transform {index}")
        if not constraints:
            return LoopSolution(
                optimized_transforms=original,
                constraints=(),
                used_no_loop_path=True,
            )

        optimized_tail = self.optimizer.optimize(
            list(original[1:]),
            [
                (
                    constraint.window_a,
                    constraint.window_b,
                    constraint.measurement,
                )
                for constraint in constraints
            ],
        )
        optimized = (original[0], *tuple(optimized_tail))
        if len(optimized) != len(states):
            raise ValueError(
                "traditional optimizer transform count does not match states"
            )
        return LoopSolution(
            optimized_transforms=optimized,
            constraints=tuple(constraints),
            used_no_loop_path=False,
        )

    def aggregate(
        self,
        states: Sequence[TraditionalWindow],
        solution: LoopSolution,
    ) -> TraditionalAggregation:
        if not states:
            raise ValueError("traditional aggregation requires states")
        if len(states) != len(solution.optimized_transforms):
            raise ValueError(
                "traditional solution transform count does not match states"
            )

        reference = identity_sim3("cpu")
        point_chunks = []
        pose_chunks = []
        confidence_chunks = []
        previous_frame_end = None
        for state, relative in zip(
            states,
            solution.optimized_transforms,
            strict=True,
        ):
            absolute = accumulate_sim3(reference, relative)
            scale, rotation, translation = absolute
            local_points = state.local_points.clone()
            camera_poses = state.camera_poses.clone()
            confidence = state.confidence.clone()
            if state.anchor_scale_mask is not None:
                local_points = (
                    torch.as_tensor(
                        reference[0],
                        device=local_points.device,
                        dtype=local_points.dtype,
                    )
                    * state.anchor_scale_mask.to(local_points)
                    * local_points
                )
            else:
                local_points = (
                    torch.as_tensor(
                        scale,
                        device=local_points.device,
                        dtype=local_points.dtype,
                    )
                    * local_points
                )
            camera_poses = self._apply_pose_sim3(
                camera_poses,
                scale,
                rotation.to(camera_poses),
                translation.to(camera_poses),
            )

            trim = 0
            if previous_frame_end is not None:
                trim = max(0, previous_frame_end - state.frame_start)
            point_chunks.append(local_points[trim:])
            pose_chunks.append(camera_poses[trim:])
            confidence_chunks.append(confidence[trim:])
            previous_frame_end = max(
                state.frame_end,
                previous_frame_end or state.frame_end,
            )
            reference = absolute

        local_points = torch.cat(point_chunks, dim=0)
        camera_poses = torch.cat(pose_chunks, dim=0)
        confidence = torch.cat(confidence_chunks, dim=0)
        global_points = torch.einsum(
            "nij,nhwj->nhwi",
            camera_poses,
            homogenize_points(local_points),
        )[..., :3]
        return TraditionalAggregation(
            local_points=local_points,
            global_points=global_points,
            camera_poses=camera_poses,
            confidence=confidence,
            mode_scalars={
                "window_count": len(states),
                "used_no_loop_path": int(solution.used_no_loop_path),
            },
        )
