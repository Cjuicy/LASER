import numpy as np

from scripts import evaluate_post_merge_split as evaluation


def test_load_trace_uses_only_split_inputs(tmp_path):
    path = tmp_path / "trace.npz"
    np.savez(
        path,
        layer_atomic__inputs__rgb=np.zeros((3, 8, 10), dtype=np.float32),
        layer_atomic__inputs__point_map=np.zeros((8, 10, 3), dtype=np.float32),
        layer_atomic__segmentation__final_labels=np.zeros((8, 10), dtype=np.intp),
        layer_atomic__segmentation__initial_labels=np.zeros((8, 10), dtype=np.intp),
        layer_atomic__segmentation__atom_scales=np.ones(1, dtype=np.float64),
    )

    trace = evaluation.load_trace(path)

    assert set(trace) == {
        "rgb",
        "point_map",
        "auto_labels",
        "atom_labels",
        "atom_scales",
        "geometry_labels",
    }
    assert trace["geometry_labels"] is None


def test_summary_enforces_region_and_runtime_budgets():
    summary = evaluation.summarize(
        [
            {
                "auto_regions": 10,
                "split_regions": 12,
                "auto_ms": 100.0,
                "split_ms": 118.0,
            },
            {
                "auto_regions": 20,
                "split_regions": 25,
                "auto_ms": 200.0,
                "split_ms": 235.0,
            },
        ]
    )

    assert summary["median_region_growth"] <= 1.30
    assert summary["median_runtime_overhead"] <= 0.20
    assert summary["passes_region_budget"] is True
    assert summary["passes_runtime_budget"] is True


def test_summary_rejects_excessive_single_trace_fragmentation():
    summary = evaluation.summarize(
        [
            {
                "auto_regions": 10,
                "split_regions": 16,
                "auto_ms": 100.0,
                "split_ms": 110.0,
            }
        ]
    )

    assert summary["passes_region_budget"] is False


def test_runtime_measurement_alternates_auto_and_split_order():
    events = []

    evaluation._paired_runtimes(
        lambda: events.append("auto"),
        lambda: events.append("split"),
        repeats=2,
    )

    assert events == ["auto", "split", "auto", "split", "split", "auto"]
