from __future__ import annotations

import pytest
import torch

from loop_closure.types import LoopSolution
from loop_closure.methods.corrected import (
    CorrectedLoopProcessor,
)
from loop_closure.methods.traditional import (
    TraditionalLoopProcessor,
)
from pipeline.config import ReconstructionMode, load_pipeline_config

from .fixtures import (
    SPECS,
    LiteralProvider,
    OneRegionSegmenter,
    SequencedAnchor,
    identity_sim3,
)
from reconstruction.modes.corrected import CorrectedWindowState
from reconstruction.modes.no_loop import NoLoopReconstructionMode
from reconstruction.modes.traditional import TraditionalWindowState
from reconstruction.modes.base import ReconstructionContext
from reconstruction.prediction_stream import iter_window_predictions


def _optimizer_config():
    return load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer


def _no_loop_context(anchor):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser_no_loop.yaml",
        ("window.size=2", "window.overlap=1", "model.process_device=cpu"),
    ).config
    return ReconstructionContext(
        predictions=iter_window_predictions(
            LiteralProvider(),
            SPECS,
            torch.zeros((4, 3, 1, 1)),
            "cpu",
        ),
        frame_ids=(0, 1, 2, 3),
        segmentation_strategy=OneRegionSegmenter(),
        anchor_propagator=anchor,
        segmentation_config=config.segmentation,
        anchor_config=config.anchor_propagation,
        registration_config=config.registration,
        window_config=config.window,
        reconstruction_mode=ReconstructionMode.NO_LOOP,
    )


def test_legacy_no_loop_uses_corrected_predecessor_for_next_registration():
    registration_sources = []
    registration_scales = iter((2.0, 3.0))

    def register(source_points, target_points, source_poses, target_poses, mask):
        del source_poses, target_poses, mask
        registration_sources.append(
            (
                float(source_points[-1, 0, 0, 2]),
                float(target_points[0, 0, 0, 2]),
            )
        )
        return identity_sim3(next(registration_scales))

    anchor = SequencedAnchor()
    result = NoLoopReconstructionMode(
        register_adjacent=register,
        apply_pose_sim3=lambda poses, scale, rotation, translation: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
    ).run(_no_loop_context(anchor))

    assert registration_sources == [(1.0, 1.0), (10.0, 1.0)]
    assert anchor.calls == [(1.0, 2.0), (10.0, 3.0)]
    assert result.local_points[:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        10.0,
        21.0,
    ]


def test_legacy_traditional_records_scale_before_final_application(
):
    states = (
        TraditionalWindowState(
            window_index=0,
            frame_start=0,
            frame_end=2,
            local_points=torch.ones((2, 1, 1, 3)),
            camera_poses=torch.eye(4).repeat(2, 1, 1),
            confidence=torch.ones((2, 1, 1)),
            segmentation_labels=(),
            anchor_scale_mask=None,
            relative_sim3=identity_sim3(),
            segmentation_diagnostics=(),
        ),
        TraditionalWindowState(
            window_index=1,
            frame_start=1,
            frame_end=3,
            local_points=torch.ones((2, 1, 1, 3)),
            camera_poses=torch.eye(4).repeat(2, 1, 1),
            confidence=torch.ones((2, 1, 1)),
            segmentation_labels=(),
            anchor_scale_mask=torch.full((2, 1, 1, 1), 3.0),
            relative_sim3=identity_sim3(2.0),
            segmentation_diagnostics=(),
        ),
    )
    solution = LoopSolution(
        optimized_transforms=(identity_sim3(), identity_sim3(2.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    result = TraditionalLoopProcessor(_optimizer_config()).aggregate(
        states,
        solution,
    )

    assert states[1].local_points[:, 0, 0, 2].tolist() == [1.0, 1.0]
    assert states[1].anchor_scale_mask[:, 0, 0, 0].tolist() == [3.0, 3.0]
    assert result.local_points[:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        3.0,
    ]


def test_legacy_corrected_applies_online_scale_and_final_delta_once(
):
    states = (
        CorrectedWindowState(
            window_index=0,
            frame_start=0,
            frame_end=2,
            local_points=torch.ones((2, 1, 1, 3)),
            camera_poses=torch.eye(4).repeat(2, 1, 1),
            confidence=torch.ones((2, 1, 1)),
            segmentation_labels=(),
            anchor_scale_mask=None,
            sim3_abs=identity_sim3(),
            sim3_edge=None,
            segmentation_diagnostics=(),
        ),
        CorrectedWindowState(
            window_index=1,
            frame_start=1,
            frame_end=3,
            local_points=torch.full((2, 1, 1, 3), 6.0),
            camera_poses=torch.eye(4).repeat(2, 1, 1),
            confidence=torch.ones((2, 1, 1)),
            segmentation_labels=(),
            anchor_scale_mask=torch.full((2, 1, 1, 1), 3.0),
            sim3_abs=identity_sim3(2.0),
            sim3_edge=identity_sim3(2.0),
            segmentation_diagnostics=(),
        ),
    )
    solution = LoopSolution(
        optimized_transforms=(identity_sim3(), identity_sim3(4.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    result = CorrectedLoopProcessor(_optimizer_config()).aggregate(
        states,
        solution,
    )

    assert states[1].local_points[:, 0, 0, 2].tolist() == [6.0, 6.0]
    assert result.local_points[:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        pytest.approx(12.0),
    ]
