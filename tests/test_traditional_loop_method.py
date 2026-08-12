from __future__ import annotations

import numpy as np
import pytest
import torch

from loop_closure.methods.traditional import TraditionalLoopProcessor
from loop_closure.types import LoopCandidate, LoopSolution
from pipeline.config import load_pipeline_config
from reconstruction.modes.traditional import TraditionalWindowState


def identity_sim3(scale=1.0):
    return scale, torch.eye(3), torch.zeros(3)


def traditional_state(
    index: int,
    *,
    relative_scale: float = 1.0,
    anchor_scale: float | None = None,
):
    return TraditionalWindowState(
        window_index=index,
        frame_start=index,
        frame_end=index + 2,
        local_points=torch.ones((2, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        segmentation_labels=(
            np.zeros((1, 1), dtype=np.intp),
            np.zeros((1, 1), dtype=np.intp),
        ),
        anchor_scale_mask=(
            None
            if anchor_scale is None
            else torch.full((2, 1, 1, 1), anchor_scale)
        ),
        relative_sim3=identity_sim3(relative_scale),
        segmentation_diagnostics=(),
    )


def traditional_states():
    return (
        traditional_state(0),
        traditional_state(1, relative_scale=2.0, anchor_scale=3.0),
    )


def processor_fixture(optimizer=None):
    optimizer_config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return TraditionalLoopProcessor(
        optimizer_config,
        optimizer=optimizer,
    )


def test_traditional_aggregation_applies_delayed_transforms_once():
    states = traditional_states()
    solution = LoopSolution(
        optimized_transforms=(identity_sim3(), identity_sim3(scale=2.0)),
        constraints=(),
        used_no_loop_path=False,
    )

    result = processor_fixture().aggregate(states, solution)

    assert result.local_points.shape[0] == 3
    torch.testing.assert_close(
        result.local_points[-1],
        torch.full_like(result.local_points[-1], 3.0),
    )
    for state in states:
        torch.testing.assert_close(
            state.local_points,
            torch.ones_like(state.local_points),
        )


class FixedEvidence:
    def __init__(self, alignment_a, alignment_b):
        self.alignment_a = alignment_a
        self.alignment_b = alignment_b

    def estimate(self, window_a, window_b, candidate):
        del window_a, window_b, candidate
        return self.alignment_a, self.alignment_b


def test_traditional_converts_joint_evidence_to_common_frame():
    evidence = FixedEvidence(
        identity_sim3(scale=2.0),
        identity_sim3(scale=6.0),
    )

    constraint = processor_fixture().build_constraints(
        traditional_states(),
        (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
        evidence,
    )[0]

    assert torch.as_tensor(constraint.measurement[0]).item() == pytest.approx(
        3.0
    )


def test_traditional_no_constraints_does_not_invoke_optimizer():
    class FailingOptimizer:
        def optimize(self, *args):
            raise AssertionError("optimizer must not be called")

    states = traditional_states()
    solution = processor_fixture(FailingOptimizer()).optimize(states, [])

    assert solution.used_no_loop_path is True
    assert solution.optimized_transforms == tuple(
        state.relative_sim3 for state in states
    )


def test_traditional_candidate_value_error_is_isolated_and_pair_deduplicated():
    states = tuple(traditional_state(index) for index in range(3))
    calls = []

    class RecordingEvidence:
        def estimate(self, window_a, window_b, candidate):
            del window_a, window_b
            calls.append(candidate)
            if len(calls) == 1:
                raise ValueError("no mutual confidence")
            return identity_sim3(), identity_sim3()

    first = LoopCandidate(frame_a=2, frame_b=0, similarity=0.8)
    second = LoopCandidate(frame_a=3, frame_b=0, similarity=0.7)
    duplicate = LoopCandidate(frame_a=2, frame_b=1, similarity=0.6)

    constraints = processor_fixture().build_constraints(
        states,
        (first, second, duplicate),
        RecordingEvidence(),
    )

    assert [constraint.candidate for constraint in constraints] == [duplicate]
    assert calls == [first, duplicate]


def test_traditional_optimizer_receives_relative_tail_and_constraints():
    class RecordingOptimizer:
        def optimize(self, edges, constraints):
            self.edges = edges
            self.constraints = constraints
            return edges

    optimizer = RecordingOptimizer()
    states = traditional_states()
    processor = processor_fixture(optimizer)
    constraint = processor.build_constraints(
        states,
        (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
        FixedEvidence(identity_sim3(), identity_sim3()),
    )[0]

    solution = processor.optimize(states, [constraint])

    assert optimizer.edges == [states[1].relative_sim3]
    assert optimizer.constraints == [(1, 0, constraint.measurement)]
    assert len(solution.optimized_transforms) == len(states)
