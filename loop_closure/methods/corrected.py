from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from inference_engine.utils.geometry import (
    accumulate_sim3,
    apply_sim3_to_pose,
    closed_form_inverse_sim3,
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


def identity_sim3_like(reference: Sim3 | None = None) -> Sim3:
    if reference is None:
        return 1.0, torch.eye(3), torch.zeros(3)
    _, rotation, translation = reference
    return (
        1.0,
        torch.eye(
            rotation.shape[-1],
            dtype=rotation.dtype,
            device=rotation.device,
        ),
        torch.zeros_like(translation),
    )


def build_local_loop_constraint(
    sim3_abs_a: Sim3,
    sim3_abs_b: Sim3,
    global_alignment_a: Sim3,
    global_alignment_b: Sim3,
) -> Sim3:
    for context, transform in (
        ("sim3_abs_a", sim3_abs_a),
        ("sim3_abs_b", sim3_abs_b),
        ("global_alignment_a", global_alignment_a),
        ("global_alignment_b", global_alignment_b),
    ):
        validate_sim3(transform, context=context)

    global_correction = accumulate_sim3(
        global_alignment_b,
        closed_form_inverse_sim3(*global_alignment_a),
    )
    constraint_ab = accumulate_sim3(
        closed_form_inverse_sim3(*sim3_abs_b),
        accumulate_sim3(global_correction, sim3_abs_a),
    )
    validate_sim3(constraint_ab, context="corrected local loop constraint")
    return constraint_ab


class CorrectedWindow(LoopWindow, Protocol):
    sim3_abs: Sim3
    sim3_edge: Sim3 | None


@dataclass(frozen=True)
class CorrectedAggregation:
    local_points: torch.Tensor
    global_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    mode_scalars: Mapping[str, int | float]


class CorrectedLoopProcessor:
    """Pure corrected constraint, edge optimization, and one-delta math."""

    name = ReconstructionMode.CORRECTED

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
        states: Sequence[CorrectedWindow],
        frame: int,
    ) -> CorrectedWindow:
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
        states: Sequence[CorrectedWindow],
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
                    "Skipping corrected loop candidate frames=%s->%s: %s",
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
                measurement = build_local_loop_constraint(
                    state_a.sim3_abs,
                    state_b.sim3_abs,
                    alignment_a,
                    alignment_b,
                )
            except ValueError as error:
                logger.warning(
                    "Skipping corrected loop candidate frames=%s->%s: %s",
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
        states: Sequence[CorrectedWindow],
        constraints: Sequence[LoopConstraint],
    ) -> LoopSolution:
        original_absolute = tuple(state.sim3_abs for state in states)
        for index, transform in enumerate(original_absolute):
            validate_sim3(transform, context=f"corrected absolute transform {index}")
        if not constraints:
            return LoopSolution(
                optimized_transforms=original_absolute,
                constraints=(),
                used_no_loop_path=True,
            )

        edges = []
        for index, state in enumerate(states[1:], start=1):
            if state.sim3_edge is None:
                raise ValueError(f"corrected state {index} is missing sequential edge")
            validate_sim3(state.sim3_edge, context=f"corrected edge {index - 1}")
            edges.append(state.sim3_edge)
        optimized_edges = self.optimizer.optimize(
            edges,
            [
                (
                    constraint.window_a,
                    constraint.window_b,
                    constraint.measurement,
                )
                for constraint in constraints
            ],
        )
        if len(optimized_edges) != len(edges):
            raise ValueError(
                "corrected optimizer edge count does not match sequential edges"
            )
        optimized_absolute = [identity_sim3_like(original_absolute[0])]
        for edge in optimized_edges:
            optimized_absolute.append(
                accumulate_sim3(optimized_absolute[-1], edge)
            )
        return LoopSolution(
            optimized_transforms=tuple(optimized_absolute),
            constraints=tuple(constraints),
            used_no_loop_path=False,
        )

    def aggregate(
        self,
        states: Sequence[CorrectedWindow],
        solution: LoopSolution,
    ) -> CorrectedAggregation:
        if not states:
            raise ValueError("corrected aggregation requires states")
        if len(states) != len(solution.optimized_transforms):
            raise ValueError("corrected solution/state count mismatch")

        local_chunks = []
        pose_chunks = []
        confidence_chunks = []
        log_scale_deltas = []
        previous_frame_end = None
        for state, optimized_abs in zip(
            states,
            solution.optimized_transforms,
            strict=True,
        ):
            delta = accumulate_sim3(
                optimized_abs,
                closed_form_inverse_sim3(*state.sim3_abs),
            )
            validate_sim3(delta, context="corrected optimization delta")
            scale, rotation, translation = delta
            scale_value = float(torch.as_tensor(scale).item())
            log_scale_deltas.append(abs(math.log(scale_value)))
            local_points = (
                torch.as_tensor(
                    scale,
                    device=state.local_points.device,
                    dtype=state.local_points.dtype,
                )
                * state.local_points.clone()
            )
            camera_poses = self._apply_pose_sim3(
                state.camera_poses.clone(),
                scale,
                rotation.to(state.camera_poses),
                translation.to(state.camera_poses),
            )
            confidence = state.confidence.clone()

            trim = 0
            if previous_frame_end is not None:
                trim = max(0, previous_frame_end - state.frame_start)
            local_chunks.append(local_points[trim:])
            pose_chunks.append(camera_poses[trim:])
            confidence_chunks.append(confidence[trim:])
            previous_frame_end = max(
                state.frame_end,
                previous_frame_end or state.frame_end,
            )

        local_points = torch.cat(local_chunks, dim=0)
        camera_poses = torch.cat(pose_chunks, dim=0)
        confidence = torch.cat(confidence_chunks, dim=0)
        global_points = torch.einsum(
            "nij,nhwj->nhwi",
            camera_poses,
            homogenize_points(local_points),
        )[..., :3]
        return CorrectedAggregation(
            local_points=local_points,
            global_points=global_points,
            camera_poses=camera_poses,
            confidence=confidence,
            mode_scalars={
                "window_count": len(states),
                "used_no_loop_path": int(solution.used_no_loop_path),
                "max_abs_log_scale_delta": max(log_scale_deltas),
                "mean_abs_log_scale_delta": (
                    sum(log_scale_deltas) / len(log_scale_deltas)
                ),
            },
        )
