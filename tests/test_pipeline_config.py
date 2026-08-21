from pathlib import Path

import pytest

from pipeline.config import (
    AtomicSplitMode,
    ConfidenceQuantileMethod,
    ModelName,
    PredictionCacheMode,
    ReconstructionMode,
    SegmentationMethod,
    load_pipeline_config,
)


DEFAULT = Path("configs/pipeline/default.yaml")
RECONSTRUCTION = Path("configs/reconstruction/pi3_laser.yaml")
NO_LOOP = Path("configs/reconstruction/pi3_laser_no_loop.yaml")


def test_version_two_selects_exact_reconstruction_mode():
    loaded = load_pipeline_config(
        RECONSTRUCTION,
        ("reconstruction.mode=traditional",),
    )
    assert loaded.config.version == 2
    assert loaded.config.reconstruction.mode is ReconstructionMode.TRADITIONAL
    assert loaded.config.registration.confidence_keep_ratio == pytest.approx(
        0.5
    )
    assert (
        loaded.config.segmentation.confidence_quantile_method
        is ConfidenceQuantileMethod.HIGHER
    )


def test_retired_loop_enable_and_method_are_rejected(tmp_path):
    path = tmp_path / "legacy.yaml"
    source = RECONSTRUCTION.read_text(encoding="utf-8")
    path.write_text(
        source.replace(
            "loop:\n",
            "loop:\n  enabled: false\n  method: traditional\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown configuration field"):
        load_pipeline_config(path)


def test_no_loop_config_may_omit_loop_section():
    loaded = load_pipeline_config(NO_LOOP)
    assert loaded.config.reconstruction.mode is ReconstructionMode.NO_LOOP
    assert loaded.config.loop is None


def test_loop_mode_requires_loop_configuration():
    with pytest.raises(ValueError, match="traditional requires loop configuration"):
        load_pipeline_config(
            NO_LOOP,
            ("reconstruction.mode=traditional",),
        )


@pytest.mark.parametrize(
    ("size", "overlap"),
    ((10, 5), (20, 5), (20, 10)),
)
def test_window_configuration_remains_experiment_controlled(size, overlap):
    loaded = load_pipeline_config(
        NO_LOOP,
        (f"window.size={size}", f"window.overlap={overlap}"),
    )
    assert (loaded.config.window.size, loaded.config.window.overlap) == (
        size,
        overlap,
    )


def test_default_config_has_approved_methods_and_defaults():
    loaded = load_pipeline_config(DEFAULT)
    assert loaded.config.model.name is ModelName.PI3
    assert (
        loaded.config.prediction_cache.mode
        is PredictionCacheMode.AUTO
    )
    assert (
        loaded.config.prediction_cache.root
        == "inference_cache/predictions"
    )
    assert loaded.config.segmentation.method is SegmentationMethod.ATOMIC
    assert (
        loaded.config.segmentation.atomic.split_mode
        is AtomicSplitMode.CONSERVATIVE
    )
    assert (
        loaded.config.reconstruction.mode is ReconstructionMode.CORRECTED
    )
    assert loaded.config.segmentation.felzenszwalb.scale == 300
    assert loaded.config.segmentation.felzenszwalb.sigma == pytest.approx(1.1)
    assert loaded.config.segmentation.felzenszwalb.min_size == 500
    window_reference = loaded.config.segmentation.window_reference
    assert window_reference.enabled is False
    assert window_reference.sampling_stride == 4
    assert window_reference.max_keyframes == 4
    assert window_reference.relative_depth_tolerance == 0.05
    assert window_reference.min_reference_score == 0.30
    assert window_reference.stop_coverage_ratio == 0.90
    assert window_reference.min_coverage_gain == 0.03
    assert window_reference.min_region_correspondences == 8
    assert window_reference.min_region_coverage == 0.10
    assert window_reference.min_region_purity == 0.80
    assert window_reference.merge_vote_threshold == 0.80
    assert len(loaded.sha256) == 64


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "segmentation.window_reference.enabled=1",
            "enabled",
        ),
        (
            "segmentation.window_reference.sampling_stride=0",
            "sampling_stride",
        ),
        (
            "segmentation.window_reference.max_keyframes=true",
            "max_keyframes",
        ),
        (
            "segmentation.window_reference.relative_depth_tolerance=0",
            "relative_depth_tolerance",
        ),
        (
            "segmentation.window_reference.min_reference_score=0",
            "min_reference_score",
        ),
        (
            "segmentation.window_reference.stop_coverage_ratio=1.1",
            "stop_coverage_ratio",
        ),
        (
            "segmentation.window_reference.min_coverage_gain=-0.01",
            "min_coverage_gain",
        ),
        (
            "segmentation.window_reference.min_region_correspondences=false",
            "min_region_correspondences",
        ),
        (
            "segmentation.window_reference.min_region_coverage=nan",
            "min_region_coverage",
        ),
        (
            "segmentation.window_reference.min_region_purity=0",
            "min_region_purity",
        ),
        (
            "segmentation.window_reference.merge_vote_threshold=inf",
            "merge_vote_threshold",
        ),
    ],
)
def test_window_reference_config_rejects_invalid_boundaries(
    override,
    message,
):
    with pytest.raises(ValueError, match=message):
        load_pipeline_config(DEFAULT, (override,))


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("auto", PredictionCacheMode.AUTO),
        ("refresh", PredictionCacheMode.REFRESH),
        ("readonly", PredictionCacheMode.READONLY),
        ("off", PredictionCacheMode.OFF),
    ),
)
def test_prediction_cache_mode_override_selects_exact_behavior(
    text,
    expected,
):
    loaded = load_pipeline_config(
        DEFAULT,
        (f"prediction_cache.mode={text}",),
    )
    assert loaded.config.prediction_cache.mode is expected


def test_non_pi3_model_name_is_rejected():
    with pytest.raises(ValueError, match="model.name"):
        load_pipeline_config(DEFAULT, ("model.name=pi3x",))


def test_unknown_prediction_cache_mode_is_rejected():
    with pytest.raises(ValueError, match="prediction_cache.mode"):
        load_pipeline_config(
            DEFAULT,
            ("prediction_cache.mode=warm",),
        )


def test_dotlist_overrides_use_new_field_paths_only():
    loaded = load_pipeline_config(
        DEFAULT,
        (
            "segmentation.method=geometry",
            "reconstruction.mode=traditional",
            "registration.confidence_keep_ratio=0.4",
        ),
    )
    assert loaded.config.segmentation.method is SegmentationMethod.GEOMETRY
    assert loaded.config.reconstruction.mode is ReconstructionMode.TRADITIONAL
    assert (
        loaded.config.registration.confidence_keep_ratio
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
            (f"registration.confidence_keep_ratio={ratio}",),
        )


def test_window_overlap_must_be_strictly_smaller_than_size():
    with pytest.raises(ValueError, match="window.size"):
        load_pipeline_config(
            DEFAULT,
            ("window.size=5", "window.overlap=5"),
        )


def test_public_pipeline_source_contains_no_legacy_parameter_names():
    roots = [
        Path("pipeline"),
        Path("run_laser.py"),
        Path("inference_engine/segmentation"),
        Path("loop_closure/methods"),
    ]
    forbidden = {
        "top_conf_percentile",
        "depth_refine",
        "segment_mode",
        "geometry_seg_profile",
        "split_aux_confirmation",
        "registration_top_confidence_ratio",
    }
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*.py"))
    )
    for name in forbidden:
        assert name not in text


def test_new_runtime_source_contains_no_retired_hart_identifiers():
    roots = [
        Path("pipeline"),
        Path("inference_engine/anchor_propagation.py"),
        Path("loop_closure/methods"),
    ]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*.py"))
    )
    for name in ("HART", "HART_AP", "hart_anchor"):
        assert name not in text
