import json
from pathlib import Path

import pytest

from inference_engine.diagnostics.schema import (
    SCHEMA_VERSION,
    DiagnosticContext,
    RunManifest,
    SelectedInterval,
)
from inference_engine.diagnostics.storage import (
    FreeSpaceReserveExceeded,
    RunLock,
    RunLockError,
    StorageBudget,
    StorageLimitExceeded,
    append_jsonl,
    atomic_write_json,
    cleanup_owned_directory,
    owned_temp_directory,
)


def test_diagnostic_context_round_trip_and_frame_ids():
    context = DiagnosticContext(
        run_id="run-1",
        config_id="layer_atomic",
        sequence_id="02",
        pass_id=2,
        window_id=3,
        frame_start=45,
    )

    restored = DiagnosticContext.from_dict(context.to_dict())

    assert restored == context
    assert restored.frame_id(4) == 49
    assert restored.to_dict()["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("pass_id", [0, 3])
def test_diagnostic_context_rejects_unknown_pass(pass_id):
    with pytest.raises(ValueError, match="pass_id"):
        DiagnosticContext("run", "depth", "00", pass_id, 0, 0)


def test_selected_interval_is_inclusive_and_validated():
    interval = SelectedInterval(
        sequence_id="10",
        start_frame=100,
        end_frame=120,
        reasons=("trajectory_regret", "scale_incoherence"),
        score=3.5,
    )

    assert interval.contains(100)
    assert interval.contains(120)
    assert not interval.contains(99)
    assert SelectedInterval.from_dict(interval.to_dict()) == interval

    with pytest.raises(ValueError, match="end_frame"):
        SelectedInterval("10", 20, 19, ("bad",), 1.0)


def test_run_manifest_round_trip_and_status_updates():
    manifest = RunManifest(
        run_id="run-1",
        git_commit="abc123",
        checkpoint_sha256="f" * 64,
        config_hash="c" * 64,
        dataset_fingerprint="d" * 64,
        seed=0,
        environment={"python": "3.11"},
        budget={"max_temp_gib": 50},
    )
    manifest.mark("pass1", "layer_atomic", "02", "complete")

    restored = RunManifest.from_dict(manifest.to_dict())

    assert restored.to_dict() == manifest.to_dict()
    assert restored.state["pass1"]["layer_atomic"]["02"] == "complete"


def test_atomic_json_and_jsonl_are_readable(tmp_path):
    destination = tmp_path / "manifest.json"
    atomic_write_json(destination, {"status": "complete", "count": 2})

    assert json.loads(destination.read_text()) == {"count": 2, "status": "complete"}
    assert not (tmp_path / "manifest.json.partial").exists()

    records = tmp_path / "metrics.jsonl"
    append_jsonl(records, {"frame": 1, "value": 2.5})
    append_jsonl(records, {"frame": 2, "value": None})
    assert [json.loads(line) for line in records.read_text().splitlines()] == [
        {"frame": 1, "value": 2.5},
        {"frame": 2, "value": None},
    ]


def test_run_lock_is_exclusive_and_reusable(tmp_path):
    lock_path = tmp_path / "run.lock"
    first = RunLock(lock_path, run_id="run-1")
    first.acquire()

    with pytest.raises(RunLockError, match="already locked"):
        RunLock(lock_path, run_id="run-2").acquire()

    first.release()
    with RunLock(lock_path, run_id="run-2"):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_owned_temp_cleanup_requires_matching_marker(tmp_path):
    with owned_temp_directory(tmp_path, run_id="run-1", cleanup=False) as owned:
        (owned / "payload.bin").write_bytes(b"abc")

    cleanup_owned_directory(owned, run_id="run-1")
    assert not owned.exists()

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "payload.bin").write_bytes(b"do-not-delete")
    with pytest.raises(PermissionError, match="ownership"):
        cleanup_owned_directory(foreign, run_id="run-1")
    assert foreign.exists()


def test_storage_budget_ok_warning_and_hard_limit(tmp_path):
    budget = StorageBudget(
        root=tmp_path,
        max_bytes=100,
        warn_bytes=80,
        min_free_bytes=0,
    )

    assert budget.state(used_bytes=79, free_bytes=1000).level == "ok"
    assert budget.state(used_bytes=80, free_bytes=1000).level == "warning"
    assert budget.enforce(used_bytes=80, free_bytes=1000).level == "warning"
    with pytest.raises(StorageLimitExceeded, match="100"):
        budget.enforce(used_bytes=95, free_bytes=1000, estimated_bytes=6)


def test_storage_budget_enforces_free_space_reserve(tmp_path):
    budget = StorageBudget(
        root=tmp_path,
        max_bytes=1000,
        warn_bytes=800,
        min_free_bytes=50,
    )

    with pytest.raises(FreeSpaceReserveExceeded, match="50"):
        budget.enforce(used_bytes=10, free_bytes=55, estimated_bytes=6)


def test_storage_budget_rejects_inconsistent_thresholds(tmp_path):
    with pytest.raises(ValueError, match="warn"):
        StorageBudget(tmp_path, max_bytes=100, warn_bytes=101, min_free_bytes=0)
