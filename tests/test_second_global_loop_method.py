from __future__ import annotations

import pytest
import torch

from loop_closure.methods.second_global import SecondGlobalLoopProcessor
from loop_closure.types import LoopCandidate, LoopSolution
from pipeline.config import ReconstructionMode, load_pipeline_config
from reconstruction.modes.traditional_second_global import SecondGlobalWindowState


def _identity_sim3(scale: float = 1.0):
    return scale, torch.eye(3), torch.zeros(3)


def _assert_sim3_equal(actual, expected):
    assert torch.as_tensor(actual[0]).item() == pytest.approx(
        torch.as_tensor(expected[0]).item()
    )
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])


def _state(
    index: int,
    *,
    sim3_abs=None,
    sim3_edge=None,
) -> SecondGlobalWindowState:
    return SecondGlobalWindowState(
        window_index=index,
        frame_start=index,
        frame_end=index + 2,
        local_points=torch.ones((2, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        sim3_abs=sim3_abs or _identity_sim3(),
        sim3_edge=sim3_edge,
    )


def _states():
    return (
        _state(0),
        _state(
            1,
            sim3_abs=_identity_sim3(2.0),
            sim3_edge=_identity_sim3(2.0),
        ),
    )


def _processor(optimizer=None):
    optimizer_config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return SecondGlobalLoopProcessor(
        optimizer_config,
        optimizer=optimizer,
    )


class _FixedEvidence:
    def __init__(self, alignment_a, alignment_b):
        self.alignment_a = alignment_a
        self.alignment_b = alignment_b

    def estimate(self, window_a, window_b, candidate):
        del window_a, window_b, candidate
        return self.alignment_a, self.alignment_b


def test_second_global_processor_uses_residual_node_coordinates():
    constraint = _processor().build_constraints(
        _states(),
        (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
        _FixedEvidence(_identity_sim3(3.0), _identity_sim3(6.0)),
    )[0]

    assert torch.as_tensor(constraint.measurement[0]).item() == pytest.approx(
        4.0
    )


def test_second_global_optimizer_receives_rebuilt_edges_and_constraints():
    class RecordingOptimizer:
        def optimize(self, edges, constraints):
            self.edges = edges
            self.constraints = constraints
            return edges

    optimizer = RecordingOptimizer()
    processor = _processor(optimizer)
    states = _states()
    constraint = processor.build_constraints(
        states,
        (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
        _FixedEvidence(_identity_sim3(), _identity_sim3()),
    )[0]

    solution = processor.optimize(states, [constraint])

    assert optimizer.edges == [states[1].sim3_edge]
    assert optimizer.constraints == [(1, 0, constraint.measurement)]
    assert len(solution.optimized_transforms) == len(states)


def test_second_global_no_constraints_skips_optimizer():
    class FailingOptimizer:
        def optimize(self, *arguments):
            raise AssertionError(f"unexpected optimizer call: {arguments}")

    solution = _processor(FailingOptimizer()).optimize(_states(), ())

    assert solution.used_no_loop_path is True
    for actual, state in zip(
        solution.optimized_transforms,
        _states(),
        strict=True,
    ):
        _assert_sim3_equal(actual, state.sim3_abs)


def test_second_global_aggregation_applies_only_optimization_delta():
    states = _states()
    solution = LoopSolution(
        optimized_transforms=(_identity_sim3(), _identity_sim3(4.0)),
        constraints=(),
        used_no_loop_path=False,
    )

    result = _processor().aggregate(states, solution)

    torch.testing.assert_close(
        result.local_points[-1],
        torch.full_like(result.local_points[-1], 2.0),
    )
    torch.testing.assert_close(
        states[1].local_points,
        torch.ones_like(states[1].local_points),
    )
    assert _processor().name is ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
