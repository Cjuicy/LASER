import math

import numpy as np

from inference_engine.diagnostics.segmentation import (
    compare_labelings,
    summarize_labels,
    trace_segmentation_frame,
)
from inference_engine.utils.depth import segment_depth_felzenszwalb_rag


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
