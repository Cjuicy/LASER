from __future__ import annotations

import numpy as np
import pytest
import torch

from inference_engine.utils.geometry import accumulate_sim3, closed_form_inverse_sim3
from loop_closure.methods.corrected import (
    CorrectedLoopProcessor,
    build_local_loop_constraint,
)
from loop_closure.methods.registry import LOOP_PROCESSORS, build_loop_processor
from loop_closure.types import LoopCandidate, LoopConstraint, LoopSolution
from pipeline.config import ReconstructionMode, load_pipeline_config
from reconstruction.modes.corrected import CorrectedWindowState


def sim3(scale=1.0, translation=None):
    return (
        scale,
        torch.eye(3),
        torch.zeros(3)
        if translation is None
        else torch.as_tensor(translation, dtype=torch.float32),
    )


def corrected_state(
    index: int,
    *,
    depth: float = 1.0,
    absolute_scale: float = 1.0,
    edge_scale: float | None = None,
):
    return CorrectedWindowState(
        window_index=index,
        frame_start=index,
        frame_end=index + 2,
        local_points=torch.full((2, 1, 1, 3), depth),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        segmentation_labels=(
            np.zeros((1, 1), dtype=np.intp),
            np.zeros((1, 1), dtype=np.intp),
        ),
        anchor_scale_mask=None,
        sim3_abs=sim3(absolute_scale),
        sim3_edge=None if edge_scale is None else sim3(edge_scale),
        segmentation_diagnostics=(),
    )


def corrected_states():
    return (
        corrected_state(0),
        corrected_state(1, depth=6.0, absolute_scale=2.0, edge_scale=2.0),
    )


def processor_fixture(optimizer=None):
    config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return CorrectedLoopProcessor(config, optimizer=optimizer)


def test_corrected_loop_measurement_is_local_coordinate_constraint():
    absolute_a = sim3(2.0)
    absolute_b = sim3(4.0)
    constraint = build_local_loop_constraint(
        absolute_a,
        absolute_b,
        sim3(),
        sim3(),
    )
    sequential = accumulate_sim3(
        closed_form_inverse_sim3(*absolute_a),
        absolute_b,
    )
    residual = accumulate_sim3(constraint, sequential)
    assert torch.as_tensor(residual[0]).item() == pytest.approx(1.0)
    torch.testing.assert_close(residual[1], torch.eye(3))
    torch.testing.assert_close(residual[2], torch.zeros(3))


class FixedEvidence:
    def __init__(self, alignment_a, alignment_b):
        self.alignment_a = alignment_a
        self.alignment_b = alignment_b

    def estimate(self, window_a, window_b, candidate):
        del window_a, window_b, candidate
        return self.alignment_a, self.alignment_b


def test_corrected_converts_joint_evidence_to_local_measurement():
    constraint = processor_fixture().build_constraints(
        corrected_states(),
        (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
        FixedEvidence(sim3(scale=2.0), sim3(scale=6.0)),
    )[0]

    assert torch.as_tensor(constraint.measurement[0]).item() == pytest.approx(
        6.0
    )


def test_corrected_aggregation_applies_only_optimization_delta_once():
    solution = LoopSolution(
        optimized_transforms=(sim3(), sim3(4.0)),
        constraints=(),
        used_no_loop_path=False,
    )

    result = processor_fixture().aggregate(corrected_states(), solution)

    torch.testing.assert_close(
        result.local_points[-1],
        torch.full_like(result.local_points[-1], 12.0),
    )
    assert result.mode_scalars["max_abs_log_scale_delta"] == pytest.approx(
        np.log(2.0)
    )


def test_corrected_no_constraints_returns_original_absolute_transforms():
    states = corrected_states()
    solution = processor_fixture().optimize(states, [])
    assert solution.used_no_loop_path is True
    assert solution.optimized_transforms == tuple(
        state.sim3_abs for state in states
    )


def test_corrected_optimizer_receives_one_edge_per_window_transition():
    class RecordingOptimizer:
        def optimize(self, edges, constraints):
            self.edges = edges
            return edges

    optimizer = RecordingOptimizer()
    states = corrected_states()
    constraint = LoopConstraint(
        window_a=1,
        window_b=0,
        measurement=sim3(),
        candidate=LoopCandidate(2, 0, 0.8),
    )

    processor_fixture(optimizer).optimize(states, [constraint])

    assert len(optimizer.edges) == len(states) - 1


def test_corrected_aggregate_rejects_state_count_mismatch():
    with pytest.raises(ValueError, match="count"):
        processor_fixture().aggregate(
            corrected_states(),
            LoopSolution(
                optimized_transforms=(sim3(),),
                constraints=(),
                used_no_loop_path=True,
            ),
        )


def test_loop_processor_registry_has_exact_loop_modes():
    assert set(LOOP_PROCESSORS) == {
        ReconstructionMode.TRADITIONAL,
        ReconstructionMode.CORRECTED,
    }
    assert build_loop_processor(
        ReconstructionMode.CORRECTED,
        optimizer_config=load_pipeline_config(
            "configs/pipeline/test.yaml",
            ("loop.optimizer.implementation=python",),
        ).config.loop.optimizer,
    ).name is ReconstructionMode.CORRECTED
