from __future__ import annotations

import csv
import json

import pytest

from evaluation.pointcloud.geometry_metrics import PrimaryMetrics
from evaluation.pointcloud.results import (
    SequencePointCloudResult,
    aggregate_pointcloud_results,
    write_pointcloud_results,
)


def _primary(value):
    return PrimaryMetrics(
        accuracy_mean_m=value,
        accuracy_median_m=value,
        completion_mean_m=value,
        completion_median_m=value,
        normal_consistency_mean=1.0 - value,
        normal_consistency_median=1.0 - value,
    )


def test_pointcloud_results_use_exact_sequence_macro_average(tmp_path):
    results = (
        SequencePointCloudResult("fixture", "a", 3, _primary(0.01), {}),
        SequencePointCloudResult("fixture", "b", 4, _primary(0.03), {}),
    )

    summary = aggregate_pointcloud_results(
        "fixture", results, expected_sequences=("a", "b")
    )
    output = write_pointcloud_results(summary, results, tmp_path / "output")

    assert summary.primary.accuracy_mean_m == pytest.approx(0.02)
    payload = json.loads((output / "pointcloud_metrics.json").read_text())
    assert payload["completed_sequences"] == 2
    with (output / "pointcloud_sequences.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["sequence"] for row in rows] == ["a", "b"]


def test_pointcloud_aggregation_rejects_incomplete_coverage():
    with pytest.raises(ValueError, match="missing.*b"):
        aggregate_pointcloud_results(
            "fixture",
            (SequencePointCloudResult("fixture", "a", 3, _primary(0.01), {}),),
            expected_sequences=("a", "b"),
        )
