import numpy as np
import pytest

from pipeline.config import load_pipeline_config
from inference_engine.segmentation import (
    SegmentationResult,
    build_segmentation_strategy,
    select_numpy_top_confidence_mask,
)


def _point_batch(frames=2, height=12, width=16):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    depth = np.broadcast_to(
        1.0 + xx / 100.0,
        (frames, height, width),
    )
    x = np.broadcast_to(xx, depth.shape)
    y = np.broadcast_to(yy, depth.shape)
    return np.stack((x, y, depth), axis=-1).copy()


@pytest.mark.parametrize("method", ("depth", "geometry"))
def test_strategy_returns_compact_full_coverage_labels(method):
    config = load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            f"segmentation.method={method}",
            "segmentation.felzenszwalb.min_size=4",
        ),
    ).config.segmentation
    points = _point_batch()
    confidence = np.ones(points.shape[:-1], dtype=np.float32)
    strategy = build_segmentation_strategy(config)
    results = strategy.segment(points, confidence, images=None)
    assert len(results) == points.shape[0]
    for result in results:
        assert isinstance(result, SegmentationResult)
        assert result.labels.shape == points.shape[1:3]
        np.testing.assert_array_equal(
            np.unique(result.labels),
            np.arange(np.unique(result.labels).size),
        )
        assert result.diagnostics["method"] == method
        assert result.diagnostics["region_count"] >= 1


def test_geometry_receives_explicit_normal_threshold():
    config = load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            "segmentation.method=geometry",
            "segmentation.geometry.normal_threshold_degrees=17",
        ),
    ).config.segmentation
    strategy = build_segmentation_strategy(config)
    assert strategy.normal_threshold_degrees == pytest.approx(17.0)


def test_numpy_confidence_keep_ratio_has_positive_semantics():
    confidence = np.arange(10, dtype=np.float32)
    mask = select_numpy_top_confidence_mask(confidence, 0.3)
    assert set(confidence[mask].tolist()) == {7.0, 8.0, 9.0}


def test_numpy_confidence_ignores_nonfinite_values():
    confidence = np.array([0.0, 1.0, np.nan, np.inf], dtype=np.float32)
    mask = select_numpy_top_confidence_mask(confidence, 0.5)
    np.testing.assert_array_equal(mask, [False, True, False, False])


def test_strategy_rejects_invalid_point_shape():
    config = load_pipeline_config(
        "configs/pipeline/default.yaml",
        ("segmentation.method=depth",),
    ).config.segmentation
    strategy = build_segmentation_strategy(config)
    with pytest.raises(ValueError, match="point_maps"):
        strategy.segment(
            np.zeros((2, 3, 4), dtype=np.float32),
            confidence=None,
            images=None,
        )
