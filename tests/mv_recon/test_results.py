import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mv_recon.geometry_metrics import (
    DirectionalNormalMetrics,
    GeometryDiagnostics,
    PrimaryMetrics,
    ThresholdMetrics,
)
from mv_recon.protocol import DatasetReference
from mv_recon.results import (
    METRIC_SCHEMA_VERSION,
    FailureRecord,
    ResultStore,
    RunIdentity,
    SequenceResult,
    aggregate_dataset,
)


def _primary(**overrides) -> PrimaryMetrics:
    values = {
        "accuracy_mean_m": 0.01,
        "accuracy_median_m": 0.005,
        "completion_mean_m": 0.02,
        "completion_median_m": 0.006,
        "normal_consistency_mean": 0.7,
        "normal_consistency_median": 0.8,
    }
    values.update(overrides)
    return PrimaryMetrics(**values)


def _diagnostics() -> GeometryDiagnostics:
    return GeometryDiagnostics(
        umeyama_scale=1.25,
        icp_transformation=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        icp_fitness=0.9,
        icp_inlier_rmse=0.01,
        predicted_point_count=100,
        ground_truth_point_count=110,
        directional_normals=DirectionalNormalMetrics(
            nc1_mean=0.6,
            nc1_median=0.7,
            nc2_mean=0.8,
            nc2_median=0.9,
        ),
        chamfer_l1_m=0.015,
        thresholds=(
            ThresholdMetrics(0.01, 0.5, 0.4, 4.0 / 9.0),
            ThresholdMetrics(0.02, 0.7, 0.6, 0.6461538461538462),
            ThresholdMetrics(0.05, 1.0, 1.0, 1.0),
        ),
    )


def _sequence(
    name: str,
    *,
    dataset: str = "7scenes-dense",
    **metric_overrides,
) -> SequenceResult:
    return SequenceResult(
        dataset=dataset,
        sequence=name,
        frame_count=100,
        input_manifest_sha256=(name[0] if name else "a") * 64,
        ground_truth_sha256="e" * 64,
        ordinary_prediction_key="b" * 64,
        primary=_primary(**metric_overrides),
        diagnostics=_diagnostics(),
        cache_diagnostics={"ordinary_hits": 1, "ordinary_misses": 0},
    )


def _identity(**overrides) -> RunIdentity:
    values = {
        "protocol_identity_sha256": "a" * 64,
        "pipeline_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "sequence_map_sha256": {"7scenes-dense": "d" * 64},
        "metric_version": METRIC_SCHEMA_VERSION,
    }
    values.update(overrides)
    return RunIdentity(**values)


def _reference() -> DatasetReference:
    return DatasetReference(
        accuracy_mean_m=0.013,
        accuracy_median_m=0.005,
        completion_mean_m=0.017,
        completion_median_m=0.006,
        normal_consistency_mean=0.607,
        normal_consistency_median=0.665,
    )


def _initialize(
    store: ResultStore,
    *,
    expected=("a", "b"),
    full_count=2,
    subset=False,
):
    return store.initialize(
        expected_sequences={"7scenes-dense": expected},
        full_sequence_counts={"7scenes-dense": full_count},
        paper_reference={"7scenes-dense": _reference()},
        subset=subset,
    )


def test_dataset_summary_is_exact_sequence_macro_average():
    first = _sequence(
        "a",
        accuracy_mean_m=0.01,
        completion_mean_m=0.03,
    )
    second = _sequence(
        "b",
        accuracy_mean_m=0.03,
        completion_mean_m=0.01,
    )

    summary = aggregate_dataset(
        "7scenes-dense",
        (first, second),
        expected_sequences=("a", "b"),
        paper_reference=_reference(),
    )

    assert summary.primary.accuracy_mean_m == pytest.approx(0.02)
    assert summary.primary.completion_mean_m == pytest.approx(0.02)
    assert summary.completed_sequences == 2
    assert summary.delta_to_paper.accuracy_mean_m == pytest.approx(0.007)


def test_dataset_summary_rejects_missing_expected_sequence():
    with pytest.raises(ValueError, match="missing.*b"):
        aggregate_dataset(
            "7scenes-dense",
            (_sequence("a"),),
            expected_sequences=("a", "b"),
            paper_reference=_reference(),
        )


def test_nonempty_output_requires_explicit_resume(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="resume"):
        ResultStore(output, _identity(), resume=False)


def test_successful_limited_run_is_subset_and_writes_canonical_artifacts(
    tmp_path,
):
    output = tmp_path / "results"
    store = ResultStore(output, _identity(), resume=False)
    _initialize(store, expected=("a",), full_count=18, subset=True)
    store.record_sequence(_sequence("a"))

    result = store.finalize()

    assert result.state == "subset"
    assert result.datasets[0].status == "subset"
    assert {
        "results.json",
        "summary.csv",
        "sequences.csv",
        "failures.jsonl",
    } <= {path.name for path in output.iterdir()}
    canonical = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert canonical["schema_version"] == METRIC_SCHEMA_VERSION
    assert canonical["state"] == "subset"
    assert canonical["datasets"][0]["delta_to_paper"]["accuracy_mean_m"] == pytest.approx(
        -0.003
    )
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["accuracy_mean_m"]) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "field",
    (
        "protocol_identity_sha256",
        "pipeline_sha256",
        "checkpoint_sha256",
        "metric_version",
    ),
)
def test_resume_rejects_changed_scalar_identity(tmp_path, field):
    output = tmp_path / "results"
    first = ResultStore(output, _identity(), resume=False)
    _initialize(first)
    changed = replace(_identity(), **{field: "f" * 64})

    with pytest.raises(ValueError, match="resume identity mismatch"):
        ResultStore(output, changed, resume=True)


def test_resume_rejects_changed_sequence_map_identity(tmp_path):
    output = tmp_path / "results"
    first = ResultStore(output, _identity(), resume=False)
    _initialize(first)
    changed = replace(
        _identity(),
        sequence_map_sha256={"7scenes-dense": "f" * 64},
    )

    with pytest.raises(ValueError, match="resume identity mismatch"):
        ResultStore(output, changed, resume=True)


def test_resume_identity_includes_auxiliary_checkpoint_hashes(tmp_path):
    identity = _identity(
        auxiliary_checkpoint_sha256={
            "salad": "a" * 64,
            "dino": "b" * 64,
        }
    )
    output = tmp_path / "results"
    store = ResultStore(output, identity, resume=False)
    _initialize(store)
    changed = replace(
        identity,
        auxiliary_checkpoint_sha256={
            "salad": "c" * 64,
            "dino": "b" * 64,
        },
    )

    with pytest.raises(ValueError, match="resume identity mismatch"):
        ResultStore(output, changed, resume=True)


def test_resume_reuses_only_exact_input_manifest(tmp_path):
    output = tmp_path / "results"
    first = ResultStore(output, _identity(), resume=False)
    _initialize(first)
    sequence = _sequence("a")
    first.record_sequence(sequence)

    resumed = ResultStore(output, _identity(), resume=True)

    assert resumed.reusable_sequence(
        "7scenes-dense",
        "a",
        sequence.input_manifest_sha256,
        sequence.ground_truth_sha256,
    ) == sequence
    assert resumed.reusable_sequence(
        "7scenes-dense",
        "a",
        "f" * 64,
        sequence.ground_truth_sha256,
    ) is None
    assert resumed.reusable_sequence(
        "7scenes-dense",
        "a",
        sequence.input_manifest_sha256,
        "f" * 64,
    ) is None


def test_nonfinite_metric_is_rejected_before_json_write(tmp_path):
    output = tmp_path / "results"
    store = ResultStore(output, _identity(), resume=False)
    _initialize(store)

    with pytest.raises(ValueError, match="finite"):
        store.record_sequence(
            _sequence("a", accuracy_mean_m=float("nan"))
        )


def test_resume_rejects_nonfinite_metric_in_stored_results(tmp_path):
    output = tmp_path / "results"
    first = ResultStore(output, _identity(), resume=False)
    _initialize(first)
    first.record_sequence(_sequence("a"))
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    payload["sequences"][0]["primary"]["accuracy_mean_m"] = float("nan")
    (output / "results.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        ResultStore(output, _identity(), resume=True)


def test_failure_after_success_preserves_sequence_and_is_incomplete(tmp_path):
    output = tmp_path / "results"
    store = ResultStore(output, _identity(), resume=False)
    _initialize(store)
    store.record_sequence(_sequence("a"))
    store.record_failure(
        FailureRecord(
            dataset="7scenes-dense",
            sequence="b",
            category="RuntimeError",
            message="GPU failed",
        )
    )

    result = store.finalize()

    assert result.state == "incomplete"
    assert [item.sequence for item in result.sequences] == ["a"]
    failure_lines = (output / "failures.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert json.loads(failure_lines[0])["sequence"] == "b"


def test_successful_resume_clears_prior_failure_for_same_sequence(tmp_path):
    output = tmp_path / "results"
    first = ResultStore(output, _identity(), resume=False)
    _initialize(first)
    first.record_sequence(_sequence("a"))
    first.record_failure(
        FailureRecord(
            dataset="7scenes-dense",
            sequence="b",
            category="RuntimeError",
            message="temporary failure",
        )
    )
    first.finalize()

    resumed = ResultStore(output, _identity(), resume=True)
    _initialize(resumed)
    resumed.record_sequence(_sequence("b"))
    result = resumed.finalize()

    assert result.state == "complete"
    assert result.failures == ()
    assert (output / "failures.jsonl").read_text(encoding="utf-8") == ""


def test_resume_replaces_stale_sequence_when_ground_truth_changes(tmp_path):
    output = tmp_path / "results"
    first = ResultStore(output, _identity(), resume=False)
    _initialize(first, expected=("a",), full_count=1)
    original = _sequence("a")
    first.record_sequence(original)
    first.finalize()

    resumed = ResultStore(output, _identity(), resume=True)
    _initialize(resumed, expected=("a",), full_count=1)
    changed = replace(original, ground_truth_sha256="f" * 64)
    assert resumed.reusable_sequence(
        "7scenes-dense",
        "a",
        changed.input_manifest_sha256,
        changed.ground_truth_sha256,
    ) is None
    resumed.record_sequence(changed)

    result = resumed.finalize()
    assert result.state == "complete"
    assert len(result.sequences) == 1
    assert result.sequences[0].ground_truth_sha256 == "f" * 64
