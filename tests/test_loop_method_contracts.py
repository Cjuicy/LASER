from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from loop_closure.types import LoopCandidate, LoopConstraint, LoopSolution
from loop_closure.utils.sim3loop import Sim3LoopOptimizer
from pipeline.config import load_pipeline_config


def identity_sim3(scale=1.0):
    return scale, torch.eye(3), torch.zeros(3)


def test_loop_candidate_is_immutable_and_scored():
    candidate = LoopCandidate(frame_a=90, frame_b=10, similarity=0.81)
    assert candidate.frame_a == 90
    with pytest.raises(FrozenInstanceError):
        candidate.similarity = 0.5


@pytest.mark.parametrize(
    "kwargs",
    (
        {"frame_a": -1, "frame_b": 0, "similarity": 0.8},
        {"frame_a": 1, "frame_b": 1, "similarity": 0.8},
        {"frame_a": 2, "frame_b": 1, "similarity": float("nan")},
    ),
)
def test_loop_candidate_rejects_invalid_frame_range_or_score(kwargs):
    with pytest.raises(ValueError):
        LoopCandidate(**kwargs)


@pytest.mark.parametrize("scale", (0.0, -1.0, float("nan")))
def test_constraint_rejects_non_positive_or_nonfinite_sim3(scale):
    candidate = LoopCandidate(90, 10, 0.81)
    with pytest.raises(ValueError, match="scale"):
        LoopConstraint(
            window_a=2,
            window_b=0,
            measurement=identity_sim3(scale),
            candidate=candidate,
        )


def test_solution_rejects_nonfinite_sim3_components():
    with pytest.raises(ValueError, match="finite"):
        LoopSolution(
            optimized_transforms=(
                (
                    1.0,
                    torch.eye(3),
                    torch.tensor([float("inf"), 0.0, 0.0]),
                ),
            ),
            constraints=(),
            used_no_loop_path=True,
        )


def test_sim3_optimizer_uses_typed_optimizer_config():
    config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        (
            "loop.optimizer.implementation=python",
            "loop.optimizer.max_iterations=17",
            "loop.optimizer.initial_damping=0.0002",
        ),
    ).config.loop.optimizer

    optimizer = Sim3LoopOptimizer(config, device="cpu")

    assert optimizer.solve_system_version == "python"
    assert optimizer.max_iterations == 17
    assert optimizer.initial_damping == pytest.approx(0.0002)
