import numpy as np
import pytest
import torch

from inference_engine.segmentation import (
    SegmentationResult,
    build_window_reference_refiner,
)
from inference_engine.segmentation.window_reference import (
    _centered_axis,
    _confidence_mask_and_quality,
    _confidence_probability,
)
from pipeline.config import load_pipeline_config


def _segmentation_config(*overrides):
    return load_pipeline_config(
        "configs/pipeline/default.yaml",
        overrides,
    ).config.segmentation


def test_disabled_refiner_returns_exact_result_list():
    config = _segmentation_config()
    results = [
        SegmentationResult(
            np.array([[0, 1]], dtype=np.intp),
            {"region_count": 2},
        )
    ]
    refiner = build_window_reference_refiner(config)

    refined = refiner.refine(
        results,
        point_maps=torch.zeros((1, 1, 2, 3)),
        camera_poses=torch.eye(4).repeat(1, 1, 1),
        confidence=torch.zeros((1, 1, 2)),
        reference_intrinsic=None,
    )

    assert refined is results
    assert refined[0].diagnostics is results[0].diagnostics


def _identity_fixture(frame_count=2):
    intrinsic = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
    )
    local = torch.tensor(
        [
            [[-0.5, -0.5, 1.0], [0.0, -0.5, 1.0], [0.5, -0.5, 1.0]],
            [[-0.5, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0]],
            [[-0.5, 0.5, 1.0], [0.0, 0.5, 1.0], [0.5, 0.5, 1.0]],
        ]
    )
    point_maps = local.unsqueeze(0).repeat(frame_count, 1, 1, 1)
    camera_poses = torch.eye(4).repeat(frame_count, 1, 1)
    confidence = torch.zeros((frame_count, 3, 3))
    results = [
        SegmentationResult(
            np.zeros((3, 3), dtype=np.intp),
            {"method": "fixture", "region_count": 1},
        )
        for _ in range(frame_count)
    ]
    return results, point_maps, camera_poses, confidence, intrinsic


def _run_enabled(
    *,
    results,
    point_maps,
    camera_poses,
    confidence,
    intrinsic,
):
    refiner = build_window_reference_refiner(
        _segmentation_config("segmentation.window_reference.enabled=true")
    )
    return refiner.refine(
        results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        reference_intrinsic=intrinsic,
    )


@pytest.mark.parametrize(
    ("name", "point_maps", "camera_poses", "confidence", "results"),
    [
        (
            "frame_count",
            torch.zeros((1, 3, 3, 3)),
            torch.eye(4).repeat(2, 1, 1),
            torch.zeros((1, 3, 3)),
            [SegmentationResult(np.zeros((3, 3), dtype=np.intp), {})],
        ),
        (
            "point_rank",
            torch.zeros((2, 3, 3)),
            torch.eye(4).repeat(2, 1, 1),
            torch.zeros((2, 3, 3)),
            [
                SegmentationResult(np.zeros((3, 3), dtype=np.intp), {})
                for _ in range(2)
            ],
        ),
        (
            "pose_rank",
            torch.zeros((2, 3, 3, 3)),
            torch.zeros((2, 4, 4, 1)),
            torch.zeros((2, 3, 3)),
            [
                SegmentationResult(np.zeros((3, 3), dtype=np.intp), {})
                for _ in range(2)
            ],
        ),
        (
            "confidence_rank",
            torch.zeros((2, 3, 3, 3)),
            torch.eye(4).repeat(2, 1, 1),
            torch.zeros((2, 3, 3, 1)),
            [
                SegmentationResult(np.zeros((3, 3), dtype=np.intp), {})
                for _ in range(2)
            ],
        ),
        (
            "spatial",
            torch.zeros((2, 3, 4, 3)),
            torch.eye(4).repeat(2, 1, 1),
            torch.zeros((2, 3, 4)),
            [
                SegmentationResult(np.zeros((3, 3), dtype=np.intp), {})
                for _ in range(2)
            ],
        ),
    ],
)
def test_enabled_refiner_rejects_programming_shape_errors(
    name,
    point_maps,
    camera_poses,
    confidence,
    results,
):
    del name
    with pytest.raises(ValueError):
        _run_enabled(
            results=results,
            point_maps=point_maps,
            camera_poses=camera_poses,
            confidence=confidence,
            intrinsic=torch.eye(3),
        )


@pytest.mark.parametrize(
    "reason",
    (
        "single_frame",
        "missing_intrinsic",
        "invalid_intrinsic",
        "intrinsic_incompatible",
        "invalid_geometry",
        "no_reference",
    ),
)
def test_enabled_refiner_returns_copied_results_for_runtime_fallback(reason):
    results, point_maps, camera_poses, confidence, intrinsic = _identity_fixture()
    if reason == "single_frame":
        results = results[:1]
        point_maps = point_maps[:1]
        camera_poses = camera_poses[:1]
        confidence = confidence[:1]
    elif reason == "missing_intrinsic":
        intrinsic = None
    elif reason == "invalid_intrinsic":
        intrinsic = intrinsic.clone()
        intrinsic[2, 2] = 2.0
    elif reason == "intrinsic_incompatible":
        point_maps = point_maps.clone()
        point_maps[..., 0] = 1.0
    elif reason == "invalid_geometry":
        camera_poses = camera_poses.clone()
        camera_poses[:, 3, 3] = 0.0
    elif reason == "no_reference":
        confidence = torch.full_like(confidence, float("nan"))

    original_labels = [result.labels.copy() for result in results]
    original_diagnostics = [dict(result.diagnostics) for result in results]
    original_point_maps = point_maps.clone()
    original_camera_poses = camera_poses.clone()
    original_confidence = confidence.clone()

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
    )

    assert refined is not results
    assert len(refined) == len(results)
    assert [result.diagnostics["window_reference_fallback"] for result in refined] == [
        reason
    ] * len(results)
    for result, original, labels, diagnostics in zip(
        refined,
        results,
        original_labels,
        original_diagnostics,
    ):
        np.testing.assert_array_equal(result.labels, labels)
        assert result.labels is not original.labels
        assert dict(result.diagnostics) == {
            **diagnostics,
            "window_reference_applied": False,
            "window_reference_fallback": reason,
            "window_reference_regions_before": 1,
            "window_reference_regions_after": 1,
        }
        assert result.diagnostics is not original.diagnostics
    torch.testing.assert_close(point_maps, original_point_maps, equal_nan=True)
    torch.testing.assert_close(camera_poses, original_camera_poses, equal_nan=True)
    torch.testing.assert_close(confidence, original_confidence, equal_nan=True)
    for result, labels, diagnostics in zip(
        results, original_labels, original_diagnostics
    ):
        np.testing.assert_array_equal(result.labels, labels)
        assert dict(result.diagnostics) == diagnostics


def test_centered_axis_uses_the_shared_stride_rule_without_special_cases():
    np.testing.assert_array_equal(_centered_axis(3, 4), [1])
    np.testing.assert_array_equal(_centered_axis(7, 3), [0, 3, 6])


def test_confidence_probability_clips_logits_stably():
    actual = _confidence_probability(np.array([-100.0, 0.0, 100.0]))
    np.testing.assert_allclose(
        actual,
        [1.0 / (1.0 + np.exp(20.0)), 0.5, 1.0 / (1.0 + np.exp(-20.0))],
    )


def test_confidence_mask_uses_valid_points_quantile_and_keeps_equal_values():
    points = np.ones((2, 2, 3), dtype=np.float64)
    points[..., 2] = 1.0
    logits = np.array([[0.0, 1.0], [1.0, np.nan]])
    mask, quality = _confidence_mask_and_quality(
        points,
        logits,
        keep_ratio=0.5,
        method="higher",
    )
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])
    assert quality == pytest.approx(
        (2.0 / 4.0) * (1.0 / (1.0 + np.exp(-1.0)))
    )
