import numpy as np
import torch

from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.segmentation import SegmentationResult
from inference_engine.streaming_window_engine import (
    STOP_SIGNAL,
    StreamingWindowEngine,
)
from inference_engine.utils.depth import match_segmentation_seq
from pipeline.config import SegmentationMethod


def identity_anchor_fixture():
    points = np.zeros((2, 2, 3, 3), dtype=np.float32)
    points[..., 2] = 1.0
    labels = [
        np.zeros((2, 3), dtype=np.intp),
        np.zeros((2, 3), dtype=np.intp),
    ]
    return (
        points.copy(),
        points.copy(),
        match_segmentation_seq(labels, iou_thresh=0.3),
        match_segmentation_seq(labels, iou_thresh=0.3),
    )


def test_anchor_propagation_keeps_existing_identity_scale_result():
    (
        source_points,
        target_points,
        source_graphs,
        target_graphs,
    ) = identity_anchor_fixture()
    propagator = AnchorPropagator(correspondence_iou_threshold=0.4)
    scale = propagator.propagate(
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap=1,
    )
    assert tuple(scale.shape) == (*target_points.shape[:-1], 1)
    torch.testing.assert_close(scale, torch.ones_like(scale))


class SpySegmentationStrategy:
    name = SegmentationMethod.ATOMIC

    def __init__(self):
        self.calls = []

    def segment(self, point_maps, confidence, images):
        self.calls.append((point_maps, confidence, images))
        return [
            SegmentationResult(
                labels=np.zeros(point_maps.shape[1:3], dtype=np.intp),
                diagnostics={"method": "atomic", "region_count": 1},
            )
            for _ in point_maps
        ]


class SpyAnchorPropagator:
    def __init__(self):
        self.calls = []

    def propagate(
        self,
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap,
    ):
        self.calls.append(
            (
                source_points,
                target_points,
                source_graphs,
                target_graphs,
                overlap,
            )
        )
        return torch.ones((*target_points.shape[:-1], 1))


def _window():
    frames, height, width = 2, 2, 3
    points = torch.zeros((1, frames, height, width, 3))
    points[..., 2] = 1.0
    return {
        "local_points": points,
        "camera_poses": torch.eye(4).repeat(1, frames, 1, 1),
        "conf": torch.ones((1, frames, height, width)),
        "images": torch.zeros((1, frames, 3, height, width)),
    }


def test_each_window_transition_invokes_one_anchor_propagator_once(
    monkeypatch,
    tmp_path,
):
    from inference_engine import streaming_window_engine as engine_module

    strategy = SpySegmentationStrategy()
    propagator = SpyAnchorPropagator()
    engine = StreamingWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=strategy,
        anchor_propagator=propagator,
        registration_confidence_keep_ratio=0.5,
        anchor_enabled=True,
        temporal_iou_threshold=0.3,
        window_size=2,
        overlap=1,
        cache_root=str(tmp_path),
        intermediate_device="cpu",
        process_device="cpu",
        benchmark_latency=False,
    )

    monkeypatch.setattr(
        engine_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda points: (points[..., 2], torch.eye(3)[None]),
    )
    monkeypatch.setattr(
        engine_module,
        "unproject_depth_to_local_points",
        lambda depth, intrinsic: torch.stack(
            (torch.zeros_like(depth), torch.zeros_like(depth), depth),
            dim=-1,
        ),
    )
    monkeypatch.setattr(
        engine_module,
        "register_adjacent_windows",
        lambda *args: (1.0, torch.eye(3), torch.zeros(3)),
    )
    monkeypatch.setattr(
        engine_module,
        "apply_sim3_to_pose",
        lambda poses, *args: poses,
    )
    engine._save_cache = lambda: None

    engine.registration_queue.put((_window(), 0.0))
    engine.registration_queue.put((_window(), 0.0))
    engine.registration_queue.put(STOP_SIGNAL)
    engine._registration_worker()

    assert len(strategy.calls) == 2
    assert len(propagator.calls) == 1
    assert propagator.calls[0][-1] == 1
    for _, _, images in strategy.calls:
        assert images.shape == (2, 3, 2, 3)
