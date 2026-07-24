from pathlib import Path

import pytest

from pipeline.config import (
    AtomicSplitMode,
    LoopMethod,
    SegmentationMethod,
    load_pipeline_config,
)


DEFAULT = Path("configs/pipeline/default.yaml")


def test_default_config_has_approved_methods_and_defaults():
    loaded = load_pipeline_config(DEFAULT)
    assert loaded.config.segmentation.method is SegmentationMethod.ATOMIC
    assert (
        loaded.config.segmentation.atomic.split_mode
        is AtomicSplitMode.CONSERVATIVE
    )
    assert loaded.config.loop.method is LoopMethod.CORRECTED
    assert loaded.config.segmentation.felzenszwalb.scale == 300
    assert loaded.config.segmentation.felzenszwalb.sigma == pytest.approx(1.1)
    assert loaded.config.segmentation.felzenszwalb.min_size == 500
    assert len(loaded.sha256) == 64


def test_dotlist_overrides_use_new_field_paths_only():
    loaded = load_pipeline_config(
        DEFAULT,
        (
            "segmentation.method=geometry",
            "loop.method=traditional",
            "loop.registration.confidence_keep_ratio=0.4",
        ),
    )
    assert loaded.config.segmentation.method is SegmentationMethod.GEOMETRY
    assert loaded.config.loop.method is LoopMethod.TRADITIONAL
    assert (
        loaded.config.loop.registration.confidence_keep_ratio
        == pytest.approx(0.4)
    )


@pytest.mark.parametrize(
    "text",
    (
        "segmentation:\n  segment_mode: depth\n",
        "segmentation:\n  geometry_seg_profile: legacy\n",
        "loop:\n  registration_top_confidence_ratio: 0.3\n",
        "anchor_propagation:\n  depth_refine: true\n",
    ),
)
def test_legacy_fields_are_rejected(tmp_path, text):
    path = tmp_path / "legacy.yaml"
    path.write_text("version: 1\n" + text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown|missing|legacy"):
        load_pipeline_config(path)


def test_missing_required_field_is_rejected(tmp_path):
    path = tmp_path / "missing.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_pipeline_config(path)


@pytest.mark.parametrize("ratio", (0.0, -0.1, 1.1))
def test_invalid_keep_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="keep_ratio"):
        load_pipeline_config(
            DEFAULT,
            (f"loop.registration.confidence_keep_ratio={ratio}",),
        )


def test_window_overlap_must_be_strictly_smaller_than_size():
    with pytest.raises(ValueError, match="window.size"):
        load_pipeline_config(
            DEFAULT,
            ("window.size=5", "window.overlap=5"),
        )
