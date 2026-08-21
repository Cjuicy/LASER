from __future__ import annotations

import numpy as np
import torch

from inference_engine.segmentation.base import SegmentationResult
from reconstruction.shared import segment_and_refine_window


class _RecordingStrategy:
    name = "fixture"

    def __init__(self, events, results):
        self.events = events
        self.results = results

    def segment(self, point_maps, confidence, images):
        assert isinstance(point_maps, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert isinstance(images, np.ndarray)
        self.events.append("segment")
        return self.results


class _RecordingRefiner:
    enabled = True

    def __init__(self, events, refined_results):
        self.events = events
        self.refined_results = refined_results
        self.received = None

    def refine(
        self,
        results,
        *,
        point_maps,
        camera_poses,
        confidence,
        reference_intrinsic,
    ):
        self.events.append("refine")
        self.received = (
            results,
            point_maps,
            camera_poses,
            confidence,
            reference_intrinsic,
        )
        return self.refined_results


def _tensors():
    point_maps = torch.ones((2, 1, 1, 3))
    camera_poses = torch.eye(4).repeat(2, 1, 1)
    confidence = torch.ones((2, 1, 1))
    images = torch.zeros((2, 3, 1, 1))
    reference_intrinsic = torch.eye(3)
    return point_maps, camera_poses, confidence, images, reference_intrinsic


def test_segment_and_refine_window_orders_calls_and_returns_refined_results():
    events = []
    initial_results = [
        SegmentationResult(np.zeros((1, 1), dtype=np.intp), {"stage": "initial"})
        for _ in range(2)
    ]
    refined_results = [
        SegmentationResult(np.zeros((1, 1), dtype=np.intp), {"stage": "refined"})
        for _ in range(2)
    ]
    strategy = _RecordingStrategy(events, initial_results)
    refiner = _RecordingRefiner(events, refined_results)
    tensors = _tensors()

    returned = segment_and_refine_window(
        strategy=strategy,
        refiner=refiner,
        point_maps=tensors[0],
        camera_poses=tensors[1],
        confidence=tensors[2],
        images=tensors[3],
        reference_intrinsic=tensors[4],
    )

    assert events == ["segment", "refine"]
    assert returned is refined_results
    assert refiner.received == (
        initial_results,
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[4],
    )


def test_segment_and_refine_window_disabled_returns_strategy_results_without_refine():
    events = []
    results = [
        SegmentationResult(np.zeros((1, 1), dtype=np.intp), {"stage": "initial"})
        for _ in range(2)
    ]
    strategy = _RecordingStrategy(events, results)

    class DisabledRefiner:
        enabled = False

        def refine(self, *args, **kwargs):
            raise AssertionError("disabled refiner must be bypassed")

    tensors = _tensors()
    returned = segment_and_refine_window(
        strategy=strategy,
        refiner=DisabledRefiner(),
        point_maps=tensors[0],
        camera_poses=tensors[1],
        confidence=tensors[2],
        images=tensors[3],
        reference_intrinsic=tensors[4],
    )

    assert events == ["segment"]
    assert returned is results
