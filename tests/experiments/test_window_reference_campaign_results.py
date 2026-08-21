from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.window_reference_campaign.diagnostics import (
    DiagnosticsSummary,
    DistributionSummary,
    aggregate_diagnostics,
    finite_distribution,
)
from experiments.window_reference_campaign.matrix import (
    RunIdentitySeed,
    complete_identity,
)
from experiments.window_reference_campaign.results import (
    CacheStats,
    FailureStage,
    RunError,
    RunRecord,
    RunStatus,
    RunTimings,
    load_valid_completed_run,
    read_run_record,
    redact_argv,
    write_campaign_metadata,
    write_run_record,
    write_summaries,
)


def _artifact_diagnostics(observations):
    return {
        "stage_timings_ms": {"reconstruction": 250.0},
        "segmentation_summaries": observations,
        "candidate_count": 0,
        "constraint_count": 0,
        "mode_scalars": {"window_count": 2},
    }


def _enabled_observation(
    *,
    window_index: int,
    frame_index: int,
    regions_before: int = 3,
    regions_after: int = 2,
    applied: bool = True,
    keyframes: str = "0",
    keyframe_count: int = 1,
    is_keyframe: bool = False,
    coverage: float = 1.0,
    candidate_edges: int = 1,
    accepted_edges: int = 1,
    conflict_edges: int = 0,
    projected_samples: int = 8,
    occluded_samples: int = 2,
    depth_rejected_samples: int = 1,
    fallback: str = "none",
):
    return {
        "window_index": window_index,
        "frame_index": frame_index,
        "region_count": regions_after,
        "window_reference_applied": applied,
        "window_reference_keyframes": keyframes,
        "window_reference_keyframe_count": keyframe_count,
        "window_reference_is_keyframe": is_keyframe,
        "window_reference_coverage_ratio": coverage,
        "window_reference_regions_before": regions_before,
        "window_reference_regions_after": regions_after,
        "window_reference_candidate_edges": candidate_edges,
        "window_reference_accepted_edges": accepted_edges,
        "window_reference_conflict_edges": conflict_edges,
        "window_reference_projected_samples": projected_samples,
        "window_reference_occluded_samples": occluded_samples,
        "window_reference_depth_rejected_samples": depth_rejected_samples,
        "window_reference_fallback": fallback,
    }


def test_overlap_observations_are_not_silently_deduplicated():
    summary = aggregate_diagnostics(
        _artifact_diagnostics([
            {
                "window_index": 0, "frame_index": 0, "region_count": 3,
                "window_reference_applied": False,
                "window_reference_keyframes": "0",
                "window_reference_keyframe_count": 1,
                "window_reference_is_keyframe": True,
                "window_reference_coverage_ratio": 0.5,
                "window_reference_regions_before": 3,
                "window_reference_regions_after": 3,
                "window_reference_candidate_edges": 0,
                "window_reference_accepted_edges": 0,
                "window_reference_conflict_edges": 0,
                "window_reference_projected_samples": 8,
                "window_reference_occluded_samples": 2,
                "window_reference_depth_rejected_samples": 1,
                "window_reference_fallback": "none",
            },
            {
                "window_index": 0, "frame_index": 1, "region_count": 2,
                "window_reference_applied": True,
                "window_reference_keyframes": "0",
                "window_reference_keyframe_count": 1,
                "window_reference_is_keyframe": False,
                "window_reference_coverage_ratio": 1.0,
                "window_reference_regions_before": 3,
                "window_reference_regions_after": 2,
                "window_reference_candidate_edges": 1,
                "window_reference_accepted_edges": 1,
                "window_reference_conflict_edges": 0,
                "window_reference_projected_samples": 9,
                "window_reference_occluded_samples": 1,
                "window_reference_depth_rejected_samples": 0,
                "window_reference_fallback": "none",
            },
            {
                "window_index": 1, "frame_index": 1, "region_count": 3,
                "window_reference_applied": False,
                "window_reference_keyframes": "1",
                "window_reference_keyframe_count": 1,
                "window_reference_is_keyframe": True,
                "window_reference_coverage_ratio": 0.5,
                "window_reference_regions_before": 3,
                "window_reference_regions_after": 3,
                "window_reference_candidate_edges": 0,
                "window_reference_accepted_edges": 0,
                "window_reference_conflict_edges": 0,
                "window_reference_projected_samples": 7,
                "window_reference_occluded_samples": 3,
                "window_reference_depth_rejected_samples": 2,
                "window_reference_fallback": "none",
            },
        ]),
        refinement_enabled=True,
    )
    assert summary.unique_frame_count == 2
    assert summary.window_frame_observation_count == 3
    assert summary.window_count == 2
    assert summary.applied_frame_count == 1
    assert summary.applied_frame_rate == pytest.approx(1 / 3)
    assert summary.region_reduction_absolute_total == 1
    assert summary.projected_sample_total == 24
    assert summary.keyframe_index_histogram == {"0": 2, "1": 1}
    assert summary.keyframe_index_records == ("0", "0", "1")


def test_disabled_refinement_is_not_fallback_or_zero_keyframe_performance():
    summary = aggregate_diagnostics(
        _artifact_diagnostics([
            {"window_index": 0, "frame_index": 0, "region_count": 4},
            {"window_index": 0, "frame_index": 1, "region_count": 3},
        ]),
        refinement_enabled=False,
    )
    assert summary.refinement_state == "not_applicable"
    assert summary.keyframe_count is None
    assert summary.coverage_ratio is None
    assert summary.applied_frame_count is None
    assert summary.applied_frame_rate is None
    assert summary.fallback_reason_histogram is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("keyframe_index_histogram", {"0": 1}),
        ("keyframe_index_records", ("0",)),
    ],
)
def test_disabled_summary_rejects_refinement_only_fields(field, value):
    values = dict(_summary(False).__dict__)
    values[field] = value
    with pytest.raises(ValueError, match="disabled refinement"):
        DiagnosticsSummary(**values)


def test_disabled_summary_round_trip_writes_null_refinement_fields(tmp_path):
    record = make_success_record()
    path = write_run_record(tmp_path / "run.json", record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = payload["diagnostics"]
    assert diagnostics["keyframe_index_histogram"] is None
    assert diagnostics["keyframe_index_records"] is None
    restored = read_run_record(path)
    assert restored == record

    paths = write_summaries(
        (record,),
        {
            (
                record.identity_seed.dataset,
                record.identity_seed.scene,
                "f000000-000002-s1",
                record.run_id,
            ): record.identity_seed,
        },
        tmp_path / "summary",
    )
    row = next(csv.DictReader(paths.diagnostics_csv.open(newline="", encoding="utf-8")))
    assert row["keyframe_index_histogram"] == ""
    assert row["keyframe_index_records"] == ""


@pytest.mark.parametrize(
    "field, value",
    [
        ("keyframe_index_histogram", {"0": 1}),
        ("keyframe_index_records", ["0"]),
    ],
)
def test_disabled_persisted_refinement_fields_are_rejected_on_read_and_resume(
    tmp_path, field, value
):
    record = make_success_record()
    path = write_run_record(tmp_path / "run.json", record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["diagnostics"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disabled refinement"):
        read_run_record(path)
    with pytest.raises(ValueError, match="disabled refinement"):
        load_valid_completed_run(path, record.identity_seed)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda item: item["region_count"].update(minimum=-1), "region_count"),
        (lambda item: item["keyframe_count"].update(minimum=-1), "keyframe_count"),
        (lambda item: item["coverage_ratio"].update(maximum=2), "coverage_ratio"),
        (lambda item: item["region_reduction_relative"].update(maximum=2), "region_reduction_relative"),
        (lambda item: item.update(candidate_edge_total=-1), "candidate_edge_total"),
        (lambda item: item.update(applied_frame_count=3), "applied_frame_count"),
        (lambda item: item.update(keyframe_index_histogram={"bad": 1}), "keyframe_index_histogram"),
        (lambda item: item.update(keyframe_index_records="0"), "keyframe_index_records"),
    ],
)
def test_read_and_resume_reject_corrupt_persisted_diagnostics(tmp_path, mutate, match):
    record = make_success_record(enabled=True)
    path = write_run_record(tmp_path / "run.json", record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload["diagnostics"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_run_record(path)
    with pytest.raises(ValueError, match=match):
        load_valid_completed_run(path, record.identity_seed)


def test_enabled_diagnostics_count_fallbacks_only_for_non_keyframes():
    summary = aggregate_diagnostics(
        _artifact_diagnostics([
            _enabled_observation(
                window_index=0, frame_index=0, applied=False,
                keyframes="", keyframe_count=0, is_keyframe=True,
                regions_before=2, regions_after=2, fallback="keyframe",
            ),
            _enabled_observation(
                window_index=0, frame_index=1, applied=False,
                keyframes="0", keyframe_count=1, is_keyframe=False,
                regions_before=2, regions_after=2,
                candidate_edges=0, accepted_edges=0, projected_samples=0,
                occluded_samples=0, depth_rejected_samples=0,
                fallback="insufficient_support",
            ),
        ]),
        refinement_enabled=True,
    )
    assert summary.refinement_state == "fallback"
    assert summary.fallback_reason_histogram == {"insufficient_support": 1}
    assert summary.keyframe_count == DistributionSummary(2, 0.0, 1.0, 0.5, 0.5)


@pytest.mark.parametrize("values", [(), (float("nan"),), (float("inf"),)])
def test_finite_distribution_is_empty_or_rejects_non_finite_values(values):
    if not values:
        assert finite_distribution(values) is None
    else:
        with pytest.raises(ValueError, match="finite"):
            finite_distribution(values)


def make_identity_seed(
    *,
    dataset: str = "synthetic",
    scene: str = "fixture",
    segmentation_method: str = "depth",
    window_reference_enabled: bool = False,
    frame_stop: int = 2,
):
    return RunIdentitySeed(
        schema_version=1,
        campaign_config_sha256="1" * 64,
        source_commit="2" * 40,
        source_dirty=False,
        dataset=dataset,
        scene=scene,
        frame_start=0,
        frame_stop=frame_stop,
        frame_stride=1,
        staged_manifest_sha256="3" * 64,
        segmentation_method=segmentation_method,
        atomic_split_mode="conservative",
        window_reference_enabled=window_reference_enabled,
        window_reference_config={
            "sampling_stride": 4,
            "max_keyframes": 4,
            "relative_depth_tolerance": 0.05,
            "min_reference_score": 0.30,
            "stop_coverage_ratio": 0.90,
            "min_coverage_gain": 0.03,
            "min_region_correspondences": 8,
            "min_region_coverage": 0.10,
            "min_region_purity": 0.80,
            "merge_vote_threshold": 0.80,
        },
        reconstruction_mode="no_loop",
        window_size=75,
        overlap=30,
        model_name="pi3",
        model_dtype="bfloat16",
        checkpoint_sha256="4" * 64,
    )


def _summary(enabled: bool = False) -> DiagnosticsSummary:
    if not enabled:
        return DiagnosticsSummary(
            unique_frame_count=2,
            window_frame_observation_count=2,
            window_count=1,
            region_count=DistributionSummary(2, 2.0, 3.0, 2.5, 2.5),
            refinement_enabled=False,
            refinement_state="not_applicable",
            keyframe_count=None,
            keyframe_index_histogram=None,
            keyframe_index_records=None,
            coverage_ratio=None,
            regions_before=None,
            regions_after=None,
            region_reduction_absolute_total=None,
            region_reduction_relative=None,
            candidate_edge_total=None,
            accepted_edge_total=None,
            conflict_edge_total=None,
            projected_sample_total=None,
            occluded_sample_total=None,
            depth_rejected_sample_total=None,
            applied_frame_count=None,
            applied_frame_rate=None,
            fallback_reason_histogram=None,
        )
    return DiagnosticsSummary(
        unique_frame_count=2,
        window_frame_observation_count=2,
        window_count=1,
        region_count=DistributionSummary(2, 2.0, 3.0, 2.5, 2.5),
        refinement_enabled=True,
        refinement_state="applied",
        keyframe_count=DistributionSummary(2, 1.0, 1.0, 1.0, 1.0),
        keyframe_index_histogram={"0": 2},
        keyframe_index_records=("0", "0"),
        coverage_ratio=DistributionSummary(2, 0.5, 1.0, 0.75, 0.75),
        regions_before=DistributionSummary(2, 3.0, 3.0, 3.0, 3.0),
        regions_after=DistributionSummary(2, 2.0, 3.0, 2.5, 2.5),
        region_reduction_absolute_total=1,
        region_reduction_relative=DistributionSummary(2, 0.0, 1 / 3, 1 / 6, 1 / 6),
        candidate_edge_total=1,
        accepted_edge_total=1,
        conflict_edge_total=0,
        projected_sample_total=16,
        occluded_sample_total=2,
        depth_rejected_sample_total=1,
        applied_frame_count=1,
        applied_frame_rate=0.5,
        fallback_reason_histogram={"none": 1},
    )


def make_success_record(
    *,
    seed: RunIdentitySeed | None = None,
    run_id: str | None = None,
    prediction_key: str = "a" * 64,
    evaluation_kind: str = "none",
    evaluation_metrics: dict[str, object] | None = None,
    enabled: bool = False,
):
    seed = seed or make_identity_seed(window_reference_enabled=enabled)
    identity = complete_identity(seed, prediction_key)
    return RunRecord(
        schema_version=1,
        run_id=run_id or f"{seed.segmentation_method}__wr-{'on' if enabled else 'off'}",
        identity_seed=seed,
        identity=identity,
        status=RunStatus.SUCCEEDED,
        attempt=1,
        failure_stage=None,
        started_at="2026-08-21T00:00:00Z",
        finished_at="2026-08-21T00:00:01Z",
        frame_count=2,
        window_count=1,
        cache_policy="auto",
        cache_stats=CacheStats(1, 0, 0, 0.1, 0.2, 1, 1024, ()),
        timings=RunTimings(0.25, 0.5),
        diagnostics=_summary(enabled),
        evaluation_kind=evaluation_kind,
        evaluation_metrics=evaluation_metrics,
        artifact_manifest_sha256="b" * 64,
        error=None,
    )


def make_six_success_records():
    records = []
    for method in ("depth", "geometry", "atomic"):
        for enabled in (False, True):
            seed = make_identity_seed(
                segmentation_method=method,
                window_reference_enabled=enabled,
            )
            records.append(make_success_record(seed=seed, enabled=enabled))
    return records


def make_six_expected_seeds():
    records = make_six_success_records()
    return {
        (
            record.identity_seed.dataset,
            record.identity_seed.scene,
            "f000000-000002-s1",
            record.run_id,
        ): record.identity_seed
        for record in records
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_completed_record_rejects_non_finite_required_metric(tmp_path, bad):
    record = make_success_record(
        evaluation_kind="internal_trajectory",
        evaluation_metrics={
            "internal_ate_rmse_m": bad,
            "internal_rpe_translation_rmse_m": 0.1,
            "internal_rpe_rotation_rmse_deg": 0.2,
            "internal_matched_frame_count": 80,
        },
    )
    with pytest.raises(ValueError, match="finite"):
        write_run_record(tmp_path / "run.json", record)
    assert not (tmp_path / "run.json").exists()


def test_resume_requires_exact_seed_and_complete_prediction_key(tmp_path):
    record = make_success_record()
    path = write_run_record(tmp_path / "run.json", record)
    assert load_valid_completed_run(path, record.identity_seed) == record
    wrong = replace(record.identity_seed, campaign_config_sha256="9" * 64)
    with pytest.raises(ValueError, match="identity"):
        load_valid_completed_run(path, wrong)


def test_failed_record_requires_failure_protocol(tmp_path):
    seed = make_identity_seed()
    failed = RunRecord(
        schema_version=1,
        run_id="depth__wr-off",
        identity_seed=seed,
        identity=None,
        status=RunStatus.FAILED,
        attempt=1,
        failure_stage=FailureStage.RECONSTRUCTION,
        started_at="2026-08-21T00:00:00Z",
        finished_at="2026-08-21T00:00:01Z",
        frame_count=0,
        window_count=1,
        cache_policy="auto",
        cache_stats=CacheStats(0, 0, 0, 0.0, 0.0, 0, 0, ()),
        timings=RunTimings(None, None),
        diagnostics=None,
        evaluation_kind="none",
        evaluation_metrics=None,
        artifact_manifest_sha256=None,
        error=RunError("RuntimeError", "bad reconstruction"),
    )
    assert write_run_record(tmp_path / "failed.json", failed).is_file()


def test_summary_writes_required_atomic_csv_and_json(tmp_path):
    records = tuple(make_six_success_records())
    paths = write_summaries(records, make_six_expected_seeds(), tmp_path / "summary")
    assert all(path.is_file() for path in paths.__dict__.values())
    with paths.runs_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["run_id"] == "depth__wr-off"
    assert rows[1]["run_id"] == "depth__wr-on"
    assert json.loads(paths.summary_json.read_text())[
        "completed_runs"
    ] == 6
    assert json.loads(paths.failures_json.read_text()) == {"failures": []}
    diagnostics = {
        row["run_id"]: row
        for row in csv.DictReader(paths.diagnostics_csv.open(newline="", encoding="utf-8"))
    }
    assert diagnostics["depth__wr-off"]["unique_frame_count"] == "2"
    assert diagnostics["depth__wr-off"]["window_frame_observation_count"] == "2"
    assert "official" not in paths.trajectory_csv.read_text(encoding="utf-8").lower()


def test_summary_rejects_prediction_key_disagreement_within_scene(tmp_path):
    records = list(make_six_success_records())
    records[1] = replace(
        records[1],
        identity=complete_identity(records[1].identity_seed, "c" * 64),
    )
    with pytest.raises(ValueError, match="prediction key"):
        write_summaries(records, make_six_expected_seeds(), tmp_path / "summary")


def test_redact_argv_masks_secret_values_and_url_userinfo():
    assert redact_argv((
        "prog",
        "--password", "hunter2",
        "--token=abc",
        "https://user:secret@example.test/path",
    )) == (
        "prog",
        "--password", "<redacted>",
        "--token=<redacted>",
        "https://<redacted>@example.test/path",
    )


def test_campaign_metadata_redacts_argv_before_atomic_json(tmp_path):
    path = write_campaign_metadata(
        tmp_path / "campaign.json",
        {"argv": ["prog", "--cookie", "abc", "https://u:p@example.test"]},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["argv"] == ["prog", "--cookie", "<redacted>", "https://<redacted>@example.test"]
