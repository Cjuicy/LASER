import math

import numpy as np
import pytest

from inference_engine.diagnostics.merge import DiagnosticParityError
from inference_engine.diagnostics.segmentation import (
    compare_labelings,
    summarize_labels,
    trace_segmentation_frame,
)
from inference_engine.utils.depth import segment_depth_felzenszwalb_rag
from inference_engine.utils import post_merge_split as pms
from inference_engine.utils.layer_atomic_geometry import (
    segment_point_map_layer_atomic_split_stages,
)


def test_label_summary_reports_area_boundaries_entropy_and_atom_growth():
    labels = np.array([[0, 0, 1, 1], [0, 2, 2, 2]])
    atoms = np.array([[10, 10, 20, 20], [10, 30, 40, 40]])

    summary = summarize_labels(labels, initial_labels=atoms)

    assert summary["valid"] is True
    assert summary["segment_count"] == 3
    assert summary["largest_segment_ratio"] == 3 / 8
    assert summary["top_k_area_ratio"] == {"1": 3 / 8, "3": 1.0, "5": 1.0}
    assert summary["boundary_edges"] == 5
    assert summary["possible_boundary_edges"] == 10
    assert summary["boundary_ratio"] == 0.5
    assert summary["atom_count"] == 4
    assert summary["atom_compression_ratio"] == 4 / 3
    assert summary["max_atoms_per_segment"] == 2
    assert summary["largest_growth_ratio"] == 1.5
    assert math.isclose(summary["effective_segment_count"], math.exp(summary["area_entropy"]))


def test_label_summary_marks_invalid_inputs_instead_of_returning_zero():
    summary = summarize_labels(np.empty((0, 0), dtype=np.int64))
    assert summary["valid"] is False
    assert summary["segment_count"] is None
    assert summary["invalid_reason"] == "empty_labels"


def test_partition_comparison_is_symmetric_for_vi_and_directional_for_merging():
    fine = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    coarse = np.zeros_like(fine)

    forward = compare_labelings(coarse, fine)
    reverse = compare_labelings(fine, coarse)

    assert forward["valid"] is True
    assert forward["variation_of_information"] == reverse["variation_of_information"]
    assert forward["overmerge_conditional_entropy"] > 0
    assert forward["oversplit_conditional_entropy"] == 0
    assert reverse["overmerge_conditional_entropy"] == 0
    assert reverse["oversplit_conditional_entropy"] > 0
    assert forward["boundary_disagreement_ratio"] == 1.0
    assert forward["best_match_pixel_agreement"] == 0.5


def test_partition_comparison_rejects_shape_mismatch_without_sentinel_numbers():
    result = compare_labelings(np.zeros((2, 2)), np.zeros((3, 2)))
    assert result["valid"] is False
    assert result["variation_of_information"] is None
    assert result["invalid_reason"] == "shape_mismatch"


def test_depth_trace_handles_single_frame_confidence_with_batch_equivalent_semantics():
    depth = np.linspace(1, 3, 36).reshape(6, 6)
    points = np.stack(np.meshgrid(np.arange(6), np.arange(6), indexing="xy") + [depth], axis=-1)
    confidence = np.linspace(0, 1, 36).reshape(6, 6)
    formal = segment_depth_felzenszwalb_rag(
        depth, .1, confidence[None], .5, seg_scale=2, seg_sigma=0,
        seg_min_size=2, batch_idx=0,
    )
    trace = trace_segmentation_frame(
        points, formal, segment_mode="depth", conf_map=confidence,
        top_conf_percentile=.5, seg_scale=2, seg_sigma=0, seg_min_size=2,
    )
    assert trace["metrics"]["final"]["valid"] is True
    assert "split" not in trace["metrics"]


def test_layer_atomic_split_trace_matches_staged_formal_labels_and_emits_evidence():
    height, width = 8, 10
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((xx, yy, 1.0 + xx / width), axis=-1)
    rgb = np.stack((xx / width, yy / height, np.zeros_like(xx)), axis=-1)
    confidence = np.linspace(0, 1, height * width).reshape(height, width)

    stages = segment_point_map_layer_atomic_split_stages(
        points, .1, rgb_images=rgb, conf_map=confidence[None],
        top_conf_percentile=.5, seg_scale=2, seg_sigma=0, seg_min_size=2,
        batch_idx=0,
    )
    trace = trace_segmentation_frame(
        points,
        stages.final_labels,
        segment_mode="layer_atomic_split",
        rgb_image=rgb,
        conf_map=confidence,
        top_conf_percentile=.5,
        seg_scale=2,
        seg_sigma=0,
        seg_min_size=2,
        normal_method="cross",
        split_score_thresh=.10,
        split_aux_confirmation=True,
    )

    np.testing.assert_array_equal(trace["arrays"]["final_labels"], stages.final_labels)
    np.testing.assert_array_equal(trace["arrays"]["pre_split_labels"], stages.pre_split_labels)
    np.testing.assert_array_equal(trace["arrays"]["changed_mask"], stages.split_trace.changed_mask)
    assert trace["metrics"]["split"]["split_aux_confirmation"] is True
    changed_mask = trace["arrays"]["changed_mask"]
    ratio = trace["metrics"]["split"]["split_changed_pixel_ratio"]
    assert isinstance(ratio, float)
    assert np.isfinite(ratio)
    assert 0.0 <= ratio <= 1.0
    assert ratio == np.count_nonzero(changed_mask) / changed_mask.size
    assert np.count_nonzero(changed_mask) == 0
    assert ratio == 0.0


def test_layer_atomic_split_trace_reports_observed_changed_pixel_ratio(monkeypatch):
    height, width = 24, 32
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[:, width // 2:] = 1.0
    normals = np.zeros_like(points)
    normals[:, :width // 2, 2] = 1.0
    normals[:, width // 2:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones((height, width), dtype=bool)),
    )
    stages = segment_point_map_layer_atomic_split_stages(
        points,
        .1,
        rgb_images=rgb,
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=.10,
    )

    trace = trace_segmentation_frame(
        points,
        stages.final_labels,
        segment_mode="layer_atomic_split",
        rgb_image=rgb,
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=.10,
    )

    changed_mask = trace["arrays"]["changed_mask"]
    ratio = trace["metrics"]["split"]["split_changed_pixel_ratio"]
    assert isinstance(ratio, float)
    assert np.isfinite(ratio)
    assert 0.0 < ratio <= 1.0
    assert ratio == np.count_nonzero(changed_mask) / changed_mask.size


def test_layer_atomic_split_trace_rejects_partition_equivalent_relabeling():
    height, width = 8, 10
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((xx, yy, 1.0 + xx / width), axis=-1)
    rgb = np.stack((xx / width, yy / height, np.zeros_like(xx)), axis=-1)
    confidence = np.linspace(0, 1, height * width).reshape(height, width)
    stages = segment_point_map_layer_atomic_split_stages(
        points, .1, rgb_images=rgb, conf_map=confidence[None],
        top_conf_percentile=.5, seg_scale=2, seg_sigma=0, seg_min_size=2,
        batch_idx=0,
    )

    with pytest.raises(DiagnosticParityError, match="differs from formal labels"):
        trace_segmentation_frame(
            points,
            stages.final_labels + 1,
            segment_mode="layer_atomic_split",
            rgb_image=rgb,
            conf_map=confidence,
            top_conf_percentile=.5,
            seg_scale=2,
            seg_sigma=0,
            seg_min_size=2,
            normal_method="cross",
            split_score_thresh=.10,
            split_aux_confirmation=True,
        )
