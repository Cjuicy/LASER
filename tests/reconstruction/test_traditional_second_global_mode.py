from __future__ import annotations

import numpy as np
import pytest
import torch

from inference_engine.utils.geometry import homogenize_points
from loop_closure.methods.traditional import TraditionalLoopProcessor
from loop_closure.types import LoopSolution
from pipeline.config import load_pipeline_config
from reconstruction.modes.traditional import TraditionalWindowState
from reconstruction.modes.traditional_second_global import (
    materialize_traditional_stage1_windows,
)


def _identity_sim3(scale: float = 1.0):
    return scale, torch.eye(3), torch.zeros(3)


def _state(
    index: int,
    *,
    relative_scale: float = 1.0,
    anchor_scale: float | None = None,
) -> TraditionalWindowState:
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
        relative_sim3=_identity_sim3(relative_scale),
        segmentation_diagnostics=(),
    )


def _processor() -> TraditionalLoopProcessor:
    optimizer_config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return TraditionalLoopProcessor(optimizer_config)


def test_stage1_materialization_exactly_reproduces_traditional_aggregate():
    states = (
        _state(0),
        _state(1, relative_scale=2.0, anchor_scale=3.0),
    )
    solution = LoopSolution(
        optimized_transforms=(_identity_sim3(), _identity_sim3(2.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    expected = _processor().aggregate(states, solution)

    windows = materialize_traditional_stage1_windows(states, solution)
    local_points = torch.cat(
        (windows[0].local_points, windows[1].local_points[1:]),
        dim=0,
    )
    camera_poses = torch.cat(
        (windows[0].camera_poses, windows[1].camera_poses[1:]),
        dim=0,
    )
    confidence = torch.cat(
        (windows[0].confidence, windows[1].confidence[1:]),
        dim=0,
    )
    global_points = torch.einsum(
        "nij,nhwj->nhwi",
        camera_poses,
        homogenize_points(local_points),
    )[..., :3]

    assert torch.equal(local_points, expected.local_points)
    assert torch.equal(camera_poses, expected.camera_poses)
    assert torch.equal(confidence, expected.confidence)
    assert torch.equal(global_points, expected.global_points)
    assert torch.equal(states[1].local_points, torch.ones_like(states[1].local_points))


def test_stage1_materialization_rejects_state_solution_count_mismatch():
    states = (_state(0), _state(1))
    solution = LoopSolution(
        optimized_transforms=(_identity_sim3(),),
        constraints=(),
        used_no_loop_path=True,
    )

    with pytest.raises(ValueError, match="materialization.*count"):
        materialize_traditional_stage1_windows(states, solution)
