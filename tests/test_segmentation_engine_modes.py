import numpy as np
import pytest
import torch

from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.segmentation import (
    SegmentationResult,
    build_segmentation_strategy,
)
from inference_engine.streaming_window_engine import StreamingWindowEngine
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC
from pipeline.config import SegmentationMethod, load_pipeline_config


def _strategy(method):
    config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        (f"segmentation.method={method}",),
    ).config.segmentation
    return build_segmentation_strategy(config)


def _engine(tmp_path, method="depth", **kwargs):
    defaults = {
        "segmentation_strategy": _strategy(method),
        "anchor_propagator": AnchorPropagator(),
        "registration_confidence_keep_ratio": 0.3,
        "anchor_enabled": True,
        "temporal_iou_threshold": 0.3,
        "window_size": 10,
        "overlap": 5,
    }
    defaults.update(kwargs)
    return StreamingWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        benchmark_latency=False,
        **defaults,
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("depth", SegmentationMethod.DEPTH),
        ("geometry", SegmentationMethod.GEOMETRY),
        ("atomic", SegmentationMethod.ATOMIC),
    ),
)
def test_engine_accepts_each_injected_segmentation_strategy(
    tmp_path,
    method,
    expected,
):
    engine = _engine(tmp_path / method, method=method)
    assert engine.segmentation_strategy.name is expected


def test_anchor_can_be_disabled_independently_of_segmentation(tmp_path):
    engine = _engine(
        tmp_path,
        method="atomic",
        anchor_enabled=False,
    )
    assert engine.anchor_enabled is False
    assert engine.segmentation_strategy.name is SegmentationMethod.ATOMIC


def test_build_segment_graph_passes_images_and_threshold_once(
    monkeypatch,
    tmp_path,
):
    class SpyStrategy:
        name = SegmentationMethod.GEOMETRY

        def __init__(self):
            self.calls = []

        def segment(self, point_maps, confidence, images):
            self.calls.append((point_maps, confidence, images))
            return [
                SegmentationResult(
                    labels=np.zeros(point_maps.shape[1:3], dtype=np.intp),
                    diagnostics={"method": "geometry", "region_count": 1},
                )
                for _ in point_maps
            ]

    from inference_engine import streaming_window_engine as engine_module

    graph_calls = []

    def fake_build_temporal_graphs(results, threshold):
        graph_calls.append((results, threshold))
        return "graph"

    monkeypatch.setattr(
        engine_module,
        "build_temporal_graphs",
        fake_build_temporal_graphs,
    )
    strategy = SpyStrategy()
    engine = _engine(
        tmp_path,
        segmentation_strategy=strategy,
        temporal_iou_threshold=0.27,
    )
    points = torch.zeros((2, 2, 3, 3))
    confidence = torch.ones((2, 2, 3))
    images = torch.zeros((2, 3, 2, 3))

    graph = engine._build_segment_graph(points, confidence, images)

    assert graph == "graph"
    assert len(strategy.calls) == 1
    assert len(graph_calls) == 1
    assert graph_calls[0][1] == pytest.approx(0.27)
    np.testing.assert_array_equal(strategy.calls[0][2], images.numpy())


def test_loop_engine_uses_same_injected_services(tmp_path):
    strategy = _strategy("atomic")
    propagator = AnchorPropagator(0.35)
    engine = StreamingWindowEngineLC(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=strategy,
        anchor_propagator=propagator,
        registration_confidence_keep_ratio=0.3,
        anchor_enabled=True,
        temporal_iou_threshold=0.3,
        window_size=10,
        overlap=5,
        cache_root=str(tmp_path),
        process_device="cpu",
        benchmark_latency=False,
    )
    assert engine.segmentation_strategy is strategy
    assert engine.anchor_propagator is propagator
