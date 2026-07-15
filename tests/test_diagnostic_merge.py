import numpy as np
import pytest

from inference_engine.diagnostics.merge import (
    DiagnosticParityError,
    analyze_layer_atomic_merge,
)
from inference_engine.utils.layer_atomic_geometry import merge_layer_atoms


def _scene():
    top = np.asarray([(x, 0.0, 1.0) for x in (0.0, 1.0, 2.05, 3.05, 7.0, 8.0)])
    points = np.stack([top + (0.0, y, 0.0) for y in range(4)])
    atoms = np.tile(np.asarray([10, 10, 20, 20, 30, 30]), (4, 1))
    coarse = np.tile(np.asarray([4, 4, 4, 4, 9, 9]), (4, 1))
    return points, atoms, coarse


def test_atomic_analyzer_matches_formal_labels_and_records_decisions():
    points, atoms, coarse = _scene()
    formal = merge_layer_atoms(points, atoms, coarse, 0.1)

    trace = analyze_layer_atomic_merge(points, atoms, coarse, 0.1, formal)

    np.testing.assert_array_equal(trace.final_labels, formal)
    assert trace.metrics["candidate_count"] == 2
    assert trace.metrics["accepted_count"] == 1
    assert trace.metrics["same_coarse_candidate_count"] == 1
    assert trace.metrics["same_coarse_accepted_count"] == 1
    assert trace.metrics["cross_coarse_candidate_count"] == 1
    assert trace.metrics["cross_coarse_accepted_count"] == 0
    assert trace.metrics["final_component_count"] == 2
    assert trace.metrics["longest_merge_chain"] == 1
    assert trace.metrics["max_atoms_per_component"] == 2
    assert len(trace.events) == 1
    assert trace.events[0]["component_atoms_after"] == 2
    assert trace.pair_table[0]["normalized_gap"] == pytest.approx(1.05)
    assert trace.pair_table[0]["threshold_margin"] == pytest.approx(0.05)
    assert trace.metrics["normalized_gap_quantiles"]["p50"] is not None
    assert trace.metrics["boundary_gap_quantiles"]["p95"] is not None
    assert trace.metrics["normal_angle_quantiles_deg"]["p50"] is not None
    assert set(np.unique(trace.decision_map)) >= {0, 1, 4}
    assert trace.component_growth_map.max() == 2


def test_atomic_analyzer_identifies_degenerate_scale_pairs():
    points = np.zeros((2, 4, 3), dtype=float)
    atoms = np.tile(np.asarray([0, 0, 1, 1]), (2, 1))
    coarse = np.zeros_like(atoms)
    formal = merge_layer_atoms(points, atoms, coarse, 0.1)

    trace = analyze_layer_atomic_merge(points, atoms, coarse, 0.1, formal)

    assert trace.metrics["accepted_count"] == 0
    assert trace.pair_table[0]["invalid_reason"] == "invalid_atom_scale"


def test_atomic_analyzer_fails_loudly_when_formal_output_does_not_match():
    points, atoms, coarse = _scene()
    wrong = np.zeros(atoms.shape, dtype=np.int64)
    with pytest.raises(DiagnosticParityError, match="formal layer-atomic labels"):
        analyze_layer_atomic_merge(points, atoms, coarse, 0.1, wrong)
