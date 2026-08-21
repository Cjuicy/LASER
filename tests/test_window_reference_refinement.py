import numpy as np
import pytest
import torch

import inference_engine.segmentation.window_reference as window_reference

from inference_engine.segmentation import (
    SegmentationResult,
    build_window_reference_refiner,
)
from inference_engine.segmentation.window_reference import (
    _centered_axis,
    _confidence_mask_and_quality,
    _confidence_probability,
    _RegionMapping,
    _adjacent_region_edges,
    _dominant_region_mappings,
    _MergeEdge,
    _PairProjection,
    _ReferenceSelection,
    _merge_region_components,
    _merge_vote_edges,
    _project_pair,
    _select_references,
    _tensor_numpy,
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


def _two_region_identity_fixture():
    results, point_maps, camera_poses, confidence, intrinsic = _identity_fixture()
    target_labels = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.intp,
    )
    results[1] = SegmentationResult(
        target_labels,
        {"method": "fixture", "region_count": 2, "custom": "kept"},
    )
    return results, point_maps, camera_poses, confidence, intrinsic


def _run_enabled(
    *,
    results,
    point_maps,
    camera_poses,
    confidence,
    intrinsic,
    overrides=(),
):
    refiner = build_window_reference_refiner(
        _segmentation_config(
            "segmentation.window_reference.enabled=true",
            *overrides,
        )
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
            "window_reference_keyframes": "",
            "window_reference_keyframe_count": 0,
            "window_reference_is_keyframe": False,
            "window_reference_coverage_ratio": 0.0,
            "window_reference_candidate_edges": 0,
            "window_reference_accepted_edges": 0,
            "window_reference_conflict_edges": 0,
            "window_reference_projected_samples": 0,
            "window_reference_occluded_samples": 0,
            "window_reference_depth_rejected_samples": 0,
            "window_reference_fallback": reason,
            "window_reference_regions_before": 1,
            "window_reference_regions_after": 1,
            "region_count": 1,
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


def test_enabled_fallback_diagnostics_are_complete_finite_scalars():
    results, point_maps, camera_poses, confidence, _ = _identity_fixture()
    results[0] = SegmentationResult(
        np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=np.intp),
        {"method": "fixture", "region_count": 99, "custom": "kept"},
    )

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=None,
    )

    expected_types = {
        "window_reference_applied": bool,
        "window_reference_keyframes": str,
        "window_reference_keyframe_count": int,
        "window_reference_is_keyframe": bool,
        "window_reference_coverage_ratio": float,
        "window_reference_regions_before": int,
        "window_reference_regions_after": int,
        "window_reference_candidate_edges": int,
        "window_reference_accepted_edges": int,
        "window_reference_conflict_edges": int,
        "window_reference_projected_samples": int,
        "window_reference_occluded_samples": int,
        "window_reference_depth_rejected_samples": int,
        "window_reference_fallback": str,
    }
    diagnostics = refined[0].diagnostics
    assert diagnostics["custom"] == "kept"
    assert diagnostics["region_count"] == 2
    assert set(expected_types).issubset(diagnostics)
    for key, expected_type in expected_types.items():
        assert type(diagnostics[key]) is expected_type
        if expected_type is float:
            assert np.isfinite(diagnostics[key])


def test_enabled_refiner_merges_adjacent_target_regions_and_preserves_keyframe():
    results, point_maps, camera_poses, confidence, intrinsic = (
        _two_region_identity_fixture()
    )
    original_labels = [result.labels.copy() for result in results]
    original_diagnostics = [dict(result.diagnostics) for result in results]

    overrides = (
        "segmentation.window_reference.sampling_stride=1",
        "segmentation.window_reference.min_region_correspondences=1",
    )
    first = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=overrides,
    )
    second = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=overrides,
    )

    np.testing.assert_array_equal(first[0].labels, original_labels[0])
    np.testing.assert_array_equal(first[1].labels, np.zeros((3, 3), dtype=np.intp))
    assert first[1].labels.dtype == np.intp
    assert np.unique(first[1].labels).tolist() == [0]
    np.testing.assert_array_equal(first[0].labels, second[0].labels)
    np.testing.assert_array_equal(first[1].labels, second[1].labels)
    assert [result.diagnostics for result in first] == [
        result.diagnostics for result in second
    ]

    assert first[0].diagnostics["window_reference_is_keyframe"] is True
    assert first[0].diagnostics["window_reference_applied"] is False
    assert first[1].diagnostics["window_reference_is_keyframe"] is False
    assert first[1].diagnostics["window_reference_applied"] is True
    assert first[1].diagnostics["window_reference_fallback"] == "none"
    assert first[1].diagnostics["window_reference_regions_before"] == 2
    assert first[1].diagnostics["window_reference_regions_after"] == 1
    assert first[1].diagnostics["region_count"] == 1
    assert first[1].diagnostics["window_reference_accepted_edges"] == 1
    assert first[1].diagnostics["window_reference_candidate_edges"] == 1
    assert first[1].diagnostics["window_reference_keyframes"] == "0"
    assert first[1].diagnostics["window_reference_keyframe_count"] == 1

    expected_types = {
        "window_reference_applied": bool,
        "window_reference_keyframes": str,
        "window_reference_keyframe_count": int,
        "window_reference_is_keyframe": bool,
        "window_reference_coverage_ratio": float,
        "window_reference_regions_before": int,
        "window_reference_regions_after": int,
        "window_reference_candidate_edges": int,
        "window_reference_accepted_edges": int,
        "window_reference_conflict_edges": int,
        "window_reference_projected_samples": int,
        "window_reference_occluded_samples": int,
        "window_reference_depth_rejected_samples": int,
        "window_reference_fallback": str,
    }
    for result in first:
        for key, expected_type in expected_types.items():
            assert type(result.diagnostics[key]) is expected_type
            if expected_type is float:
                assert np.isfinite(result.diagnostics[key])

    for result, labels, diagnostics in zip(
        results, original_labels, original_diagnostics, strict=True
    ):
        np.testing.assert_array_equal(result.labels, labels)
        assert dict(result.diagnostics) == diagnostics


def test_enabled_refiner_uses_insufficient_support_fallback_without_splitting():
    results, point_maps, camera_poses, confidence, intrinsic = (
        _two_region_identity_fixture()
    )
    results[1] = SegmentationResult(
        results[1].labels.astype(np.int32),
        dict(results[1].diagnostics),
    )
    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=(
            "segmentation.window_reference.sampling_stride=1",
            "segmentation.window_reference.min_region_correspondences=9",
        ),
    )

    np.testing.assert_array_equal(refined[1].labels, results[1].labels)
    assert refined[1].labels.dtype == np.intp
    assert refined[1].diagnostics["window_reference_fallback"] == (
        "insufficient_support"
    )
    assert refined[1].diagnostics["window_reference_regions_before"] == 2
    assert refined[1].diagnostics["window_reference_regions_after"] == 2
    assert refined[1].diagnostics["region_count"] == 2


def test_enabled_refiner_compacts_bool_nonkeyframe_fallback_labels():
    results, point_maps, camera_poses, confidence, intrinsic = (
        _two_region_identity_fixture()
    )
    results[1] = SegmentationResult(
        np.array(
            [
                [False, False, True],
                [False, False, True],
                [False, False, True],
            ],
            dtype=bool,
        ),
        dict(results[1].diagnostics),
    )

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=(
            "segmentation.window_reference.sampling_stride=1",
            "segmentation.window_reference.min_region_correspondences=9",
        ),
    )

    np.testing.assert_array_equal(refined[1].labels, results[1].labels)
    assert refined[1].labels.dtype == np.intp
    assert refined[1].diagnostics["window_reference_fallback"] == (
        "insufficient_support"
    )


def test_enabled_refiner_accepts_direct_weighted_edge_with_weak_dissent(
    monkeypatch,
):
    height, width = 5, 2
    intrinsic = torch.eye(3)
    rows, columns = np.indices((height, width))
    points = np.stack(
        [columns.astype(np.float64), rows.astype(np.float64), np.ones((height, width))],
        axis=-1,
    )
    point_maps = torch.from_numpy(np.stack([points, points, points]))
    camera_poses = torch.eye(4).repeat(3, 1, 1)
    confidence = torch.zeros((3, height, width))
    target_labels = np.repeat(np.arange(2, dtype=np.int32)[None, :], height, axis=0)
    source_zero = np.zeros((height, width), dtype=np.intp)
    source_one = np.zeros((height, width), dtype=np.intp)
    source_one.flat[1] = 1
    results = [
        SegmentationResult(source_zero, {"method": "fixture", "region_count": 1}),
        SegmentationResult(source_one, {"method": "fixture", "region_count": 2}),
        SegmentationResult(target_labels, {"method": "fixture", "region_count": 2}),
    ]

    target_by_region = {
        region: np.flatnonzero(target_labels.reshape(-1) == region)
        for region in range(2)
    }

    def projection(assignments, *, score):
        source_indices = []
        target_indices = []
        for target_region, source_index, full in assignments:
            target_region_indices = target_by_region[target_region]
            if not full:
                target_region_indices = target_region_indices[:1]
            target_indices.append(target_region_indices)
            source_indices.append(
                np.full(target_region_indices.shape, source_index, dtype=np.int64)
            )
        source_indices = np.concatenate(source_indices)
        target_indices = np.concatenate(target_indices)
        return _PairProjection(
            source_indices,
            target_indices,
            np.ones(target_indices.shape, dtype=np.float64),
            score=score,
            projected_samples=int(target_indices.size),
        )

    selected = _ReferenceSelection(
        indices=(0, 1),
        best_scores=np.array([0.0, 0.0, 1.0]),
        pair_projections=(
            (
                0,
                2,
                projection(((0, 0, True), (1, 0, True)), score=1.0),
            ),
            (
                1,
                2,
                projection(((0, 0, True), (1, 1, True)), score=0.2),
            ),
        ),
        coverage_ratio=1.0,
    )
    monkeypatch.setattr(
        "inference_engine.segmentation.window_reference._select_references",
        lambda **kwargs: selected,
    )

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=(
            "segmentation.window_reference.sampling_stride=1",
            "segmentation.window_reference.min_region_correspondences=1",
            "segmentation.window_reference.min_reference_score=0.1",
        ),
    )

    np.testing.assert_array_equal(refined[2].labels, np.zeros_like(target_labels))
    assert refined[2].labels.dtype == np.intp
    assert refined[2].diagnostics["window_reference_fallback"] == "none"
    assert refined[2].diagnostics["window_reference_accepted_edges"] == 1
    assert refined[2].diagnostics["window_reference_conflict_edges"] == 0
    assert refined[2].diagnostics["window_reference_candidate_edges"] == 1
    assert refined[2].diagnostics["window_reference_regions_after"] == 1


def test_enabled_refiner_accepts_weighted_direct_merge_with_weak_separate_vote(
    monkeypatch,
):
    """A passing weighted edge is not vetoed by its own weak dissent."""

    height, width = 5, 2
    intrinsic = torch.eye(3)
    rows, columns = np.indices((height, width))
    points = np.stack(
        [columns.astype(np.float64), rows.astype(np.float64), np.ones((height, width))],
        axis=-1,
    )
    point_maps = torch.from_numpy(np.stack([points, points, points]))
    camera_poses = torch.eye(4).repeat(3, 1, 1)
    confidence = torch.zeros((3, height, width))
    target_labels = np.repeat(np.arange(2, dtype=np.int32)[None, :], height, axis=0)
    source_zero = np.zeros((height, width), dtype=np.intp)
    source_one = np.zeros((height, width), dtype=np.intp)
    source_one.flat[1] = 1
    results = [
        SegmentationResult(source_zero, {"method": "fixture", "region_count": 1}),
        SegmentationResult(source_one, {"method": "fixture", "region_count": 2}),
        SegmentationResult(target_labels, {"method": "fixture", "region_count": 2}),
    ]
    target_by_region = {
        region: np.flatnonzero(target_labels.reshape(-1) == region)
        for region in range(2)
    }

    def projection(assignments, *, score):
        source_indices = []
        target_indices = []
        for target_region, source_index, full in assignments:
            target_region_indices = target_by_region[target_region]
            if not full:
                target_region_indices = target_region_indices[:1]
            target_indices.append(target_region_indices)
            source_indices.append(
                np.full(target_region_indices.shape, source_index, dtype=np.int64)
            )
        source_indices = np.concatenate(source_indices)
        target_indices = np.concatenate(target_indices)
        return _PairProjection(
            source_indices,
            target_indices,
            np.ones(target_indices.shape, dtype=np.float64),
            score=score,
            projected_samples=int(target_indices.size),
        )

    selected = _ReferenceSelection(
        indices=(0, 1),
        best_scores=np.array([0.0, 0.0, 1.0]),
        pair_projections=(
            (
                0,
                2,
                projection(((0, 0, True), (1, 0, True)), score=1.0),
            ),
            (
                1,
                2,
                projection(((0, 0, True), (1, 1, True)), score=0.2),
            ),
        ),
        coverage_ratio=1.0,
    )
    monkeypatch.setattr(
        "inference_engine.segmentation.window_reference._select_references",
        lambda **kwargs: selected,
    )

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=(
            "segmentation.window_reference.sampling_stride=1",
            "segmentation.window_reference.min_region_correspondences=1",
        ),
    )

    np.testing.assert_array_equal(refined[2].labels, np.zeros_like(target_labels))
    assert refined[2].diagnostics["window_reference_accepted_edges"] == 1
    assert refined[2].diagnostics["window_reference_conflict_edges"] == 0


def test_enabled_refiner_blocks_transitive_conflict_after_weighted_direct_merge(
    monkeypatch,
):
    """A later union cannot extend a component with an earlier mixed signature."""

    height, width = 3, 3
    intrinsic = torch.eye(3)
    rows, columns = np.indices((height, width))
    points = np.stack(
        [columns.astype(np.float64), rows.astype(np.float64), np.ones((height, width))],
        axis=-1,
    )
    point_maps = torch.from_numpy(np.stack([points] * 4))
    camera_poses = torch.eye(4).repeat(4, 1, 1)
    confidence = torch.zeros((4, height, width))
    target_labels = np.repeat(np.arange(3, dtype=np.int32)[None, :], height, axis=0)
    source_labels = [
        np.repeat(np.array([[0, 1, 1]], dtype=np.intp), height, axis=0),
        np.repeat(np.array([[0, 0, 0]], dtype=np.intp), height, axis=0),
        np.repeat(np.array([[0, 0, 1]], dtype=np.intp), height, axis=0),
    ]
    results = [
        SegmentationResult(labels, {"method": "fixture", "region_count": 3})
        for labels in (*source_labels, target_labels)
    ]
    target_indices = np.arange(target_labels.size, dtype=np.int64)
    source0_projection = _PairProjection(
        target_indices,
        target_indices,
        np.ones(target_indices.shape, dtype=np.float64),
        score=0.1,
        projected_samples=int(target_indices.size),
    )
    source1_projection = _PairProjection(
        target_indices,
        target_indices,
        np.ones(target_indices.shape, dtype=np.float64),
        score=1.0,
        projected_samples=int(target_indices.size),
    )
    source2_target = np.concatenate(
        [target_indices, np.array([2, 2, 2], dtype=np.int64)]
    )
    source2_source = np.concatenate(
        [target_indices, np.array([8, 8, 0], dtype=np.int64)]
    )
    source2_projection = _PairProjection(
        source2_source,
        source2_target,
        np.ones(source2_target.shape, dtype=np.float64),
        score=0.3,
        projected_samples=int(source2_target.size),
    )
    projections = (
        (0, 3, source0_projection),
        (1, 3, source1_projection),
        (2, 3, source2_projection),
    )
    selected = _ReferenceSelection(
        indices=(0, 1, 2),
        best_scores=np.array([0.0, 0.0, 0.0, 1.0]),
        pair_projections=projections,
        coverage_ratio=1.0,
    )
    monkeypatch.setattr(
        "inference_engine.segmentation.window_reference._select_references",
        lambda **kwargs: selected,
    )

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=(
            "segmentation.window_reference.sampling_stride=1",
            "segmentation.window_reference.min_region_correspondences=1",
            "segmentation.window_reference.min_reference_score=0.05",
        ),
    )

    np.testing.assert_array_equal(
        refined[3].labels,
        np.repeat(np.array([[0, 0, 1]], dtype=np.intp), height, axis=0),
    )
    assert refined[3].diagnostics["window_reference_accepted_edges"] == 1
    assert refined[3].diagnostics["window_reference_conflict_edges"] == 1


def test_tensor_numpy_honors_requested_dtype_and_converts_bfloat16_to_float32():
    points = _tensor_numpy(torch.ones((1, 2, 3), dtype=torch.float64), dtype=np.float32)
    poses = _tensor_numpy(torch.eye(4, dtype=torch.float32), dtype=np.float64)
    assert points.dtype == np.dtype(np.float32)
    assert poses.dtype == np.dtype(np.float64)
    if hasattr(torch, "bfloat16"):
        bfloat_points = _tensor_numpy(
            torch.ones((1, 2, 3), dtype=torch.bfloat16),
            dtype=np.float32,
        )
        assert bfloat_points.dtype == np.dtype(np.float32)


def test_selection_discards_pair_correspondence_arrays_after_scoring():
    pair = _PairProjection(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        score=1.0,
        projected_samples=1,
    )
    config = _segmentation_config().window_reference
    selection = _select_references(
        qualities=np.array([0.9, 0.8]),
        frame_count=2,
        evaluate=lambda source, target: pair,
        config=config,
    )

    assert selection.pair_projections
    for _, _, summary in selection.pair_projections:
        assert not isinstance(summary, _PairProjection)
        assert not hasattr(summary, "source_flat_indices")
        assert not hasattr(summary, "target_flat_indices")
        assert not hasattr(summary, "correspondence_weights")


def test_dominant_mapping_groups_evidence_once(monkeypatch):
    grouped = getattr(window_reference, "_aggregate_region_evidence", None)
    assert grouped is not None
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(1)
        return grouped(*args, **kwargs)

    monkeypatch.setattr(window_reference, "_aggregate_region_evidence", wrapped)
    labels = np.repeat(np.arange(64, dtype=np.intp)[:, None], 2, axis=1)
    evidence = [
        (int(pixel), int(label), 1.0)
        for pixel, label in enumerate(labels.flat)
    ]
    mapping = _dominant_region_mappings(
        labels,
        evidence,
        sampling_stride=1,
        min_region_correspondences=1,
        min_region_coverage=0.1,
        min_region_purity=0.8,
    )

    assert len(mapping) == labels.max() + 1
    assert calls == [1]


def test_low_score_selected_pair_keeps_projection_diagnostics_without_merge():
    results, point_maps, camera_poses, confidence, intrinsic = _two_region_identity_fixture()
    point_maps = point_maps.clone()
    point_maps[1, 1, 1, 2] = 2.0
    original_labels = results[1].labels.copy()

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=intrinsic,
        overrides=(
            "segmentation.window_reference.sampling_stride=1",
            "segmentation.window_reference.min_reference_score=0.95",
        ),
    )

    np.testing.assert_array_equal(refined[1].labels, original_labels)
    assert refined[1].diagnostics["window_reference_fallback"] == (
        "insufficient_support"
    )
    assert refined[1].diagnostics["window_reference_projected_samples"] > 0
    assert refined[1].diagnostics["window_reference_depth_rejected_samples"] > 0


def test_intrinsic_compatibility_uses_full_matrix_including_skew():
    results, point_maps, camera_poses, confidence, _ = _identity_fixture()
    skewed_intrinsic = torch.tensor(
        [[2.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
    )

    refined = _run_enabled(
        results=results,
        point_maps=point_maps,
        camera_poses=camera_poses,
        confidence=confidence,
        intrinsic=skewed_intrinsic,
        overrides=("segmentation.window_reference.sampling_stride=1",),
    )

    assert refined[0].diagnostics["window_reference_fallback"] == (
        "intrinsic_incompatible"
    )


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


def test_selection_chooses_highest_quality_before_temporal_ties():
    pair_scores = np.array(
        [
            [0.0, 0.8, 0.8],
            [0.8, 0.0, 0.8],
            [0.8, 0.8, 0.0],
        ]
    )
    config = _segmentation_config().window_reference

    selection = _select_references(
        qualities=np.array([0.9, 0.8, 0.8]),
        frame_count=3,
        evaluate=lambda source, target: pair_scores[source][target],
        config=config,
    )

    assert selection.indices == (0,)
    assert selection.rejected_indices == ()
    np.testing.assert_allclose(selection.best_scores, [0.0, 0.8, 0.8])
    assert selection.coverage_ratio == pytest.approx(1.0)


def test_selection_uses_temporal_center_for_first_quality_tie():
    pair_scores = np.array(
        [
            [0.0, 0.8, 0.8],
            [0.8, 0.0, 0.8],
            [0.8, 0.8, 0.0],
        ]
    )
    config = _segmentation_config().window_reference

    selection = _select_references(
        qualities=np.array([0.8, 0.8, 0.2]),
        frame_count=3,
        evaluate=lambda source, target: pair_scores[source][target],
        config=config,
    )

    assert selection.indices == (1,)
    assert selection.coverage_ratio == pytest.approx(1.0)


def test_selection_uses_lower_index_for_temporal_center_tie():
    pair_scores = np.full((4, 4), 0.8)
    np.fill_diagonal(pair_scores, 0.0)
    config = _segmentation_config().window_reference

    selection = _select_references(
        qualities=np.array([0.2, 0.8, 0.8, 0.2]),
        frame_count=4,
        evaluate=lambda source, target: pair_scores[source][target],
        config=config,
    )

    assert selection.indices == (1,)
    assert selection.coverage_ratio == pytest.approx(1.0)


def test_selection_rejects_low_gain_candidate_and_continues():
    pair_scores = np.array(
        [
            [0.0, 0.2, 0.2, 0.0],
            [0.5, 0.0, 0.21, 0.0],
            [0.0, 0.9, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    config = _segmentation_config(
        "segmentation.window_reference.min_reference_score=0.30",
        "segmentation.window_reference.stop_coverage_ratio=0.90",
        "segmentation.window_reference.min_coverage_gain=0.03",
        "segmentation.window_reference.max_keyframes=4",
    ).window_reference

    selection = _select_references(
        qualities=np.array([0.9, 0.85, 0.8, 0.0]),
        frame_count=4,
        evaluate=lambda source, target: pair_scores[source][target],
        config=config,
    )

    assert selection.indices == (0, 2)
    assert selection.rejected_indices == (1,)
    np.testing.assert_allclose(selection.best_scores, [0.0, 0.9, 0.2, 0.8])
    assert selection.coverage_ratio == pytest.approx(1.0)
    assert [(source, target) for source, target, _ in selection.pair_projections] == [
        (2, 1),
        (2, 3),
    ]


def test_selection_honors_keyframe_safety_ceiling():
    pair_scores = np.array(
        [
            [0.0, 0.4, 0.2],
            [0.4, 0.0, 0.4],
            [0.4, 0.4, 0.0],
        ]
    )
    config = _segmentation_config(
        "segmentation.window_reference.min_reference_score=0.30",
        "segmentation.window_reference.stop_coverage_ratio=0.90",
        "segmentation.window_reference.min_coverage_gain=0.03",
        "segmentation.window_reference.max_keyframes=1",
    ).window_reference

    selection = _select_references(
        qualities=np.array([0.9, 0.8, 0.7]),
        frame_count=3,
        evaluate=lambda source, target: pair_scores[source][target],
        config=config,
    )

    assert selection.indices == (0,)
    assert selection.rejected_indices == ()
    assert selection.coverage_ratio == pytest.approx(2.0 / 3.0)


def test_selection_exhausts_zero_quality_candidates_without_rejection():
    pair_scores = np.zeros((3, 3))
    config = _segmentation_config().window_reference

    selection = _select_references(
        qualities=np.array([0.9, 0.0, 0.0]),
        frame_count=3,
        evaluate=lambda source, target: pair_scores[source][target],
        config=config,
    )

    assert selection.indices == (0,)
    assert selection.rejected_indices == ()
    assert selection.coverage_ratio == pytest.approx(1.0 / 3.0)


def test_selection_breaks_equal_candidate_priority_by_lower_index():
    pair_scores = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.8],
            [0.0, 0.8, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    config = _segmentation_config(
        "segmentation.window_reference.min_reference_score=0.30",
        "segmentation.window_reference.stop_coverage_ratio=0.90",
        "segmentation.window_reference.min_coverage_gain=0.03",
        "segmentation.window_reference.max_keyframes=4",
    ).window_reference

    selection = _select_references(
        qualities=np.array([0.9, 0.8, 0.8, 0.1]),
        frame_count=4,
        evaluate=lambda source, target: pair_scores[source][target],
        config=config,
    )

    assert selection.indices == (0, 1)
    assert selection.rejected_indices == ()


def _projection_fixture_inputs():
    results, point_maps, camera_poses, confidence, intrinsic = _identity_fixture()
    del results, confidence
    points = point_maps.numpy()
    poses = camera_poses.numpy()
    probabilities = np.full((2, 3, 3), 0.5, dtype=np.float64)
    masks = np.ones((2, 3, 3), dtype=bool)
    return (
        points[0],
        points[1],
        poses[0],
        poses[1],
        masks[0],
        masks[1],
        probabilities[0],
        probabilities[1],
        intrinsic.numpy(),
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )


def _project_fixture_pair(inputs, *, sampling_stride=4, relative_depth_tolerance=0.05):
    (
        source_points,
        target_points,
        source_pose,
        target_pose,
        source_mask,
        target_mask,
        source_probability,
        target_probability,
        intrinsic,
        source_rows,
        source_columns,
    ) = inputs
    return _project_pair(
        source_points=source_points,
        target_points=target_points,
        source_pose=source_pose,
        target_pose=target_pose,
        source_mask=source_mask,
        target_mask=target_mask,
        source_probability=source_probability,
        target_probability=target_probability,
        intrinsic=intrinsic,
        source_rows=source_rows,
        source_columns=source_columns,
        sampling_stride=sampling_stride,
        relative_depth_tolerance=relative_depth_tolerance,
    )


def test_identity_projection_maps_center_sample_to_same_target_pixel():
    pair = _project_fixture_pair(_projection_fixture_inputs())

    assert pair.source_flat_indices.tolist() == [4]
    assert pair.target_flat_indices.tolist() == [4]


def test_c2w_translation_projects_source_center_one_pixel_right():
    inputs = list(_projection_fixture_inputs())
    source_pose = inputs[2].copy()
    source_pose[0, 3] = 0.5
    inputs[2] = source_pose

    pair = _project_fixture_pair(inputs)

    assert pair.source_flat_indices.tolist() == [4]
    assert pair.target_flat_indices.tolist() == [5]


def test_projection_z_buffer_prefers_nearest_depth_and_lower_flat_index_on_tie():
    inputs = list(_projection_fixture_inputs())
    source_points = inputs[0].copy()
    source_points[1, 1] = [0.0, 0.0, 2.0]
    source_points[1, 2] = [0.0, 0.0, 1.0]
    inputs[0] = source_points
    inputs[9] = np.array([1], dtype=np.int64)
    inputs[10] = np.array([1, 2], dtype=np.int64)

    pair = _project_fixture_pair(inputs)

    assert pair.source_flat_indices.tolist() == [5]
    assert pair.target_flat_indices.tolist() == [4]
    assert pair.projected_samples == 2
    assert pair.occluded_samples == 1

    source_points[1, 1] = [0.0, 0.0, 1.0]
    inputs[0] = source_points
    pair = _project_fixture_pair(inputs)

    assert pair.source_flat_indices.tolist() == [4]


def test_projection_rejects_out_of_bounds_and_nonpositive_depth_samples():
    inputs = list(_projection_fixture_inputs())
    source_points = inputs[0].copy()
    source_points[1, 1] = [2.0, 0.0, 1.0]
    source_points[1, 2] = [0.0, 0.0, -1.0]
    inputs[0] = source_points
    inputs[9] = np.array([1], dtype=np.int64)
    inputs[10] = np.array([1, 2], dtype=np.int64)

    pair = _project_fixture_pair(inputs)

    assert pair.source_flat_indices.size == 0
    assert pair.target_flat_indices.size == 0
    assert pair.projected_samples == 0
    assert pair.out_of_bounds_samples == 1
    assert pair.nonpositive_depth_samples == 1


def test_projection_abstains_for_low_confidence_target_pixels():
    inputs = list(_projection_fixture_inputs())
    target_mask = inputs[5].copy()
    target_mask[1, 1] = False
    inputs[5] = target_mask

    pair = _project_fixture_pair(inputs)

    assert pair.source_flat_indices.size == 0
    assert pair.target_flat_indices.size == 0
    assert pair.coverage == pytest.approx(0.0)
    assert pair.score == pytest.approx(0.0)


def test_projection_uses_strict_relative_depth_tolerance():
    inputs = list(_projection_fixture_inputs())
    target_points = inputs[1].copy()
    target_points[1, 1, 2] = 1.0 / (1.0 - 0.05)
    inputs[1] = target_points

    pair = _project_fixture_pair(inputs)

    assert pair.source_flat_indices.size == 0
    assert pair.depth_rejected_samples == 1
    assert pair.geometry_ratio == pytest.approx(0.0)


def test_projection_rejects_exact_relative_depth_boundary_strictly():
    inputs = list(_projection_fixture_inputs())
    target_points = inputs[1].copy()
    target_points[1, 1, 2] = 2.0
    inputs[1] = target_points

    pair = _project_fixture_pair(inputs, relative_depth_tolerance=0.5)

    assert pair.source_flat_indices.size == 0
    assert pair.depth_rejected_samples == 1


def test_projection_counts_positive_source_transformed_to_nonpositive_target_depth():
    inputs = list(_projection_fixture_inputs())
    target_pose = inputs[3].copy()
    target_pose[2, 3] = 2.0
    inputs[3] = target_pose

    pair = _project_fixture_pair(inputs)

    assert pair.source_flat_indices.size == 0
    assert pair.nonpositive_depth_samples == 1


def test_projection_is_deterministic_immutable_and_does_not_mutate_inputs():
    inputs = list(_projection_fixture_inputs())
    originals = [value.copy() for value in inputs]

    first = _project_fixture_pair(inputs)
    second = _project_fixture_pair(inputs)

    np.testing.assert_array_equal(first.source_flat_indices, second.source_flat_indices)
    np.testing.assert_array_equal(first.target_flat_indices, second.target_flat_indices)
    np.testing.assert_array_equal(
        first.correspondence_weights,
        second.correspondence_weights,
    )
    assert first.coverage == second.coverage
    assert first.geometry_ratio == second.geometry_ratio
    assert first.mean_confidence == second.mean_confidence
    assert first.score == second.score
    for value, original in zip(inputs, originals):
        np.testing.assert_array_equal(value, original)
    for value in (
        first.source_flat_indices,
        first.target_flat_indices,
        first.correspondence_weights,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        first.source_flat_indices[0] = 0


def test_projection_rejects_noncanonical_compatibility_call_forms():
    with pytest.raises(TypeError):
        _project_pair(
            *_projection_fixture_inputs(),
            sampling_stride=4,
            relative_depth_tolerance=0.05,
        )


def test_projection_reports_perfect_pair_score_and_zero_denominators():
    pair = _project_fixture_pair(_projection_fixture_inputs())

    assert pair.coverage == pytest.approx(1.0)
    assert pair.geometry_ratio == pytest.approx(1.0)
    assert pair.mean_confidence == pytest.approx(0.5)
    assert pair.score == pytest.approx(np.sqrt(0.5))
    np.testing.assert_allclose(pair.correspondence_weights, [0.5])

    inputs = list(_projection_fixture_inputs())
    inputs[4] = np.zeros((3, 3), dtype=bool)
    inputs[5] = np.zeros((3, 3), dtype=bool)
    empty_pair = _project_fixture_pair(inputs)
    assert empty_pair.coverage == pytest.approx(0.0)
    assert empty_pair.geometry_ratio == pytest.approx(0.0)
    assert empty_pair.mean_confidence == pytest.approx(0.0)
    assert empty_pair.score == pytest.approx(0.0)


def test_dominant_mapping_requires_unique_hit_minimum():
    labels = np.zeros((2, 2), dtype=np.intp)
    evidence = [(0, 7, 1.0), (1, 7, 1.0), (2, 7, 1.0)]

    mapping = _dominant_region_mappings(
        labels,
        evidence,
        sampling_stride=1,
        min_region_correspondences=4,
        min_region_coverage=0.1,
        min_region_purity=0.8,
    )

    assert mapping == {}


def test_dominant_mapping_uses_stride_coverage_and_support_minimum():
    labels = np.zeros((2, 2), dtype=np.intp)
    evidence = [(0, 7, 2.0)]

    mapping = _dominant_region_mappings(
        labels,
        evidence,
        sampling_stride=2,
        min_region_correspondences=1,
        min_region_coverage=1.0,
        min_region_purity=0.1,
    )

    assert mapping[0].source_label == 7
    assert mapping[0].coverage == pytest.approx(1.0)
    assert mapping[0].purity == pytest.approx(1.0)
    assert mapping[0].support == pytest.approx(1.0)


def test_dominant_mapping_requires_purity_and_breaks_weight_ties_by_label():
    labels = np.zeros((1, 4), dtype=np.intp)
    evidence = [(0, 9, 1.0), (1, 9, 1.0), (2, 3, 2.0), (3, 3, 1.0)]

    mapping = _dominant_region_mappings(
        labels,
        evidence,
        sampling_stride=1,
        min_region_correspondences=1,
        min_region_coverage=0.1,
        min_region_purity=0.8,
    )

    assert mapping == {}

    tied = _dominant_region_mappings(
        labels,
        [(0, 9, 1.0), (1, 3, 1.0)],
        sampling_stride=1,
        min_region_correspondences=1,
        min_region_coverage=0.1,
        min_region_purity=0.5,
    )
    assert tied[0].source_label == 3
    assert tied[0].purity == pytest.approx(0.5)


def test_adjacent_region_edges_are_literal_four_connected_and_sorted():
    labels = np.array(
        [
            [0, 0, 1],
            [0, 2, 1],
            [3, 3, 2],
        ],
        dtype=np.intp,
    )

    assert _adjacent_region_edges(labels) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3),
    )


def test_merge_votes_are_weighted_and_abstain_on_unreliable_mapping():
    labels = np.array([[0, 1]], dtype=np.intp)
    reliable = {
        0: _RegionMapping(5, 2, 1.0, 0.9, 0.9),
        1: _RegionMapping(5, 2, 1.0, 0.9, 0.9),
    }
    disagreement = {
        0: _RegionMapping(5, 2, 1.0, 0.9, 0.9),
        1: _RegionMapping(6, 2, 1.0, 0.9, 0.9),
    }

    eligible = _merge_vote_edges(
        labels,
        [(1.0, reliable), (0.2, disagreement)],
        merge_vote_threshold=0.80,
    )

    assert len(eligible) == 1
    assert eligible[0].label_pair == (0, 1)
    assert eligible[0].merge_evidence == pytest.approx(0.9)
    assert eligible[0].separate_evidence == pytest.approx(0.18)
    assert eligible[0].ratio == pytest.approx(0.9 / 1.08)

    abstained = _merge_vote_edges(
        labels,
        [(1.0, {0: reliable[0]})],
        merge_vote_threshold=0.0,
    )
    assert abstained == ()


def test_component_union_keeps_nonadjacent_regions_separate_and_rejects_conflict():
    labels = np.array([[0, 1, 2, 1, 0]], dtype=np.intp)
    same_reference = {
        0: _RegionMapping(4, 1, 1.0, 1.0, 1.0),
        2: _RegionMapping(4, 1, 1.0, 1.0, 1.0),
    }
    assert _merge_vote_edges(
        labels,
        [(1.0, same_reference)],
        merge_vote_threshold=0.8,
    ) == ()

    edges = (
        # The first edge is accepted; the second conflicts on reference 0.
        _MergeEdge((0, 1), 1.0, 0.0, 1.0),
        _MergeEdge((1, 2), 0.9, 0.0, 0.9),
    )
    roots, accepted, conflicts = _merge_region_components(
        3,
        edges,
        {
            0: {0: 4},
            1: {0: 4, 1: 8},
            2: {0: 5, 1: 8},
        },
    )

    np.testing.assert_array_equal(roots, [0, 0, 2])
    assert accepted == 1
    assert conflicts == 1
