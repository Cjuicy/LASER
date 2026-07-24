from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine.streaming_window_engine import StreamingWindowEngine
from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.segmentation import build_segmentation_strategy
from inference_engine.utils import registration_confidence
from loop_closure.methods import (
    CorrectedWindowEngine,
    TraditionalWindowEngine,
)
from inference_engine.utils.registration_confidence import (
    select_top_confidence_mask,
    validate_confidence_keep_ratio,
)
from pipeline.config import load_pipeline_config


def test_top_thirty_percent_uses_point_seven_quantile():
    confidence = torch.arange(10, dtype=torch.float32)

    mask = select_top_confidence_mask(confidence, keep_ratio=0.3)

    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_top_confidence_mask_supports_mixed_precision_cpu_tensors(dtype):
    confidence = torch.arange(10, dtype=dtype)

    mask = select_top_confidence_mask(confidence, keep_ratio=0.3)

    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_non_finite_confidence_is_never_selected():
    confidence = torch.tensor([1.0, float("nan"), float("inf"), 4.0])

    mask = select_top_confidence_mask(confidence, keep_ratio=0.5)

    assert mask.tolist() == [False, False, False, True]


def test_confidence_selection_rejects_no_finite_values():
    confidence = torch.tensor([float("nan"), float("inf")])

    with pytest.raises(ValueError, match="no finite values"):
        select_top_confidence_mask(confidence, keep_ratio=0.3)


def test_mutual_confidence_mask_rejects_disjoint_selections():
    source = torch.tensor([True, False])
    target = torch.tensor([False, True])

    with pytest.raises(
        ValueError,
        match="loop candidate has no shared high-confidence pixels",
    ):
        registration_confidence.intersect_confidence_masks(
            source,
            target,
            context="loop candidate",
        )


def test_mutual_confidence_mask_returns_shared_selection():
    source = torch.tensor([True, True, False])
    target = torch.tensor([False, True, True])

    mutual = registration_confidence.intersect_confidence_masks(
        source,
        target,
    )

    assert mutual.tolist() == [False, True, False]


@pytest.mark.parametrize("keep_ratio", [0.0, -0.1, 1.1, float("nan")])
def test_confidence_keep_ratio_must_be_in_unit_interval(keep_ratio):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        validate_confidence_keep_ratio(keep_ratio)


def make_base_engine(tmp_path, **kwargs):
    segmentation = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("segmentation.method=depth",),
    ).config.segmentation
    defaults = {
        "segmentation_strategy": build_segmentation_strategy(segmentation),
        "anchor_propagator": AnchorPropagator(),
        "registration_confidence_keep_ratio": 0.5,
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


def test_base_engine_uses_explicit_registration_keep_ratio(tmp_path):
    engine = make_base_engine(
        tmp_path,
        registration_confidence_keep_ratio=0.3,
    )

    assert engine.registration_confidence_keep_ratio == 0.3


@pytest.mark.parametrize(
    "engine_type",
    (TraditionalWindowEngine, CorrectedWindowEngine),
)
def test_loop_window_engines_use_same_registration_field(
    tmp_path,
    engine_type,
):
    segmentation = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("segmentation.method=depth",),
    ).config.segmentation
    engine = engine_type(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=build_segmentation_strategy(segmentation),
        anchor_propagator=AnchorPropagator(),
        registration_confidence_keep_ratio=0.3,
        anchor_enabled=True,
        temporal_iou_threshold=0.3,
        window_size=10,
        overlap=5,
        process_device="cpu",
        cache_root=str(tmp_path),
    )

    assert engine.registration_confidence_keep_ratio == 0.3


@pytest.mark.parametrize(
    "legacy_field",
    (
        "top_conf_percentile",
        "registration_top_confidence_ratio",
        "depth_refine",
        "segment_mode",
        "geometry_seg_profile",
    ),
)
def test_base_engine_rejects_removed_legacy_parameters(
    tmp_path,
    legacy_field,
):
    with pytest.raises(TypeError, match=legacy_field):
        make_base_engine(tmp_path, **{legacy_field: 0.5})
