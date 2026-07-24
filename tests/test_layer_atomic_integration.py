import numpy as np

from inference_engine.segmentation import atomic as atomic_module
from inference_engine.segmentation.registry import (
    STRATEGY_FACTORIES,
    build_segmentation_strategy,
)
from inference_engine.utils.post_merge_split import SplitDiagnostics
from pipeline.config import SegmentationMethod, load_pipeline_config


def test_atomic_strategy_dispatches_complete_point_maps(monkeypatch):
    point_maps = np.arange(2 * 3 * 4 * 3, dtype=np.float64).reshape(2, 3, 4, 3)
    conf_map = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    rgb = np.zeros((2, 3, 4, 3), dtype=np.float32)
    calls = []

    def fake_segment(point_map, **kwargs):
        calls.append((point_map, kwargs))
        return (
            np.zeros(point_map.shape[:2], dtype=np.intp),
            SplitDiagnostics(split_mode="normal_only"),
        )

    monkeypatch.setattr(
        atomic_module,
        "segment_point_map_atomic",
        fake_segment,
    )
    config = load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            "segmentation.method=atomic",
            "segmentation.atomic.split_mode=normal_only",
            "segmentation.confidence_keep_ratio=0.25",
        ),
    ).config.segmentation
    strategy = build_segmentation_strategy(config)
    results = strategy.segment(point_maps, conf_map, rgb)

    assert len(results) == 2
    assert len(calls) == 2
    for frame_index, (received_points, kwargs) in enumerate(calls):
        np.testing.assert_array_equal(
            received_points,
            point_maps[frame_index],
        )
        np.testing.assert_array_equal(
            kwargs["conf_map"],
            conf_map[frame_index],
        )
        np.testing.assert_array_equal(
            kwargs["rgb_images"],
            rgb[frame_index],
        )
        assert kwargs["confidence_keep_ratio"] == 0.25
        assert kwargs["split_mode"].value == "normal_only"
    assert all(
        result.diagnostics["split_mode"] == "normal_only"
        for result in results
    )


def test_segmentation_registry_has_exactly_three_public_methods():
    assert set(STRATEGY_FACTORIES) == {
        SegmentationMethod.DEPTH,
        SegmentationMethod.GEOMETRY,
        SegmentationMethod.ATOMIC,
    }
