import copy

import numpy as np

from inference_engine.diagnostics.selection import robust_zscore, select_intervals


def _records():
    records = []
    for seq in ("00", "02", "04", "05", "09", "10"):
        for window in range(8):
            spike = 10.0 if window == (2 if seq in {"02", "04", "10"} else 5) else 0.1 * window
            records.append({
                "config_id": "layer_atomic", "sequence_id": seq,
                "window_id": window, "frame_start": window * 5, "frame_end": window * 5 + 9,
                "trajectory_regret": spike,
                "merge_anomaly": spike * .8,
                "atom_anomaly": spike * .7,
                "scale_dispersion": spike * .6,
                "temporal_churn": spike * .4,
                "gt_speed": window + 1, "gt_turn": window * .1, "confidence": .9 - window * .01,
            })
    return records


def test_robust_zscore_is_finite_for_constant_and_outlier_inputs():
    np.testing.assert_array_equal(robust_zscore([1, 1, 1]), np.zeros(3))
    values = robust_zscore([0, 0, 10])
    assert np.isfinite(values).all()
    assert values[-1] > values[0]


def test_selector_is_deterministic_bounded_expanded_and_reason_preserving():
    records = _records()
    first = select_intervals(records, limit=12, context_windows=2)
    second = select_intervals(copy.deepcopy(records), limit=12, context_windows=2)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) <= 12
    assert {item.sequence_id for item in first} >= {"00", "02", "04", "05", "09", "10"}
    assert all(item.reasons for item in first)
    # Original spike at frame 10 is expanded two five-frame strides to frame 0.
    assert any(item.sequence_id == "02" and item.start_frame == 0 for item in first)


def test_selector_unions_configs_and_keeps_family_diversity():
    records = _records()
    duplicate = [{**row, "config_id": "geometry_baseline"} for row in records]
    selected = select_intervals(records + duplicate, limit=48)
    reasons = {reason for item in selected for reason in item.reasons}
    assert {"trajectory", "merge", "immutable_atom", "scale", "temporal", "control"} <= reasons
    assert len(selected) <= 48
