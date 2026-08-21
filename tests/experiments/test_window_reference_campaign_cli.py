"""Subprocess contract tests for the complete campaign command line."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "run_window_reference_campaign.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_synthetic_campaign(
    tmp_path: Path,
    *extra: str,
    check: bool = True,
) -> Path:
    output = tmp_path / "campaign-output"
    missing = tmp_path / "missing"
    command = [
        sys.executable,
        str(ENTRYPOINT),
        "run",
        "--preset",
        "synthetic-smoke",
        "--data-root",
        str(missing / "data"),
        "--checkpoint",
        str(missing / "model.safetensors"),
        "--output-root",
        str(output),
        *extra,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if check:
        assert completed.returncode == 0, completed.stderr
    return output / "window-reference-v1"


def test_synthetic_cli_runs_exact_six_without_checkpoint_gpu_or_pi3(tmp_path: Path):
    campaign = run_synthetic_campaign(tmp_path, "--resume", "--fail-fast")
    rows = read_csv(campaign / "summary/runs.csv")
    assert [row["run_id"] for row in rows] == [
        "depth__wr-off",
        "depth__wr-on",
        "geometry__wr-off",
        "geometry__wr-on",
        "atomic__wr-off",
        "atomic__wr-on",
    ]
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in campaign.glob("runs/synthetic/*/*/*/run.json")
    ]
    assert len(records) == 6
    assert {record["identity"]["prediction_key"] for record in records} == {
        records[0]["identity"]["prediction_key"]
    }
    assert all(record["identity"]["reconstruction_mode"] == "no_loop" for record in records)
    assert all(record["identity"]["window_size"] == 75 for record in records)
    assert all(record["identity"]["overlap"] == 30 for record in records)
    assert all(record["identity"]["atomic_split_mode"] == "conservative" for record in records)
    assert rows[0]["cache_policy"] == "auto"
    assert all(row["cache_policy"] == "readonly" for row in rows[1:])


def test_synthetic_diagnostics_cover_merge_fallback_occlusion_and_depth_rejection(
    tmp_path: Path,
):
    campaign = run_synthetic_campaign(tmp_path)
    rows = {row["run_id"]: row for row in read_csv(campaign / "summary/diagnostics.csv")}
    for run_id in ("depth__wr-off", "geometry__wr-off", "atomic__wr-off"):
        assert rows[run_id]["refinement_state"] == "not_applicable"
        assert rows[run_id]["applied_frame_count"] == ""
    for run_id in ("depth__wr-on", "geometry__wr-on", "atomic__wr-on"):
        assert int(rows[run_id]["applied_frame_count"]) >= 1
        assert int(rows[run_id]["accepted_edge_total"]) >= 1
        assert int(rows[run_id]["occluded_sample_total"]) >= 1
        assert int(rows[run_id]["depth_rejected_sample_total"]) >= 1
        histogram = json.loads(rows[run_id]["fallback_reason_histogram"])
        assert histogram["insufficient_support"] >= 1


def test_cli_subset_preserves_canonical_order_and_resume_skips_exact_records(
    tmp_path: Path,
):
    campaign = run_synthetic_campaign(
        tmp_path,
        "--methods",
        "atomic,depth",
        "--refinement",
        "on",
    )
    first = read_csv(campaign / "summary/runs.csv")
    assert [row["run_id"] for row in first] == ["depth__wr-on", "atomic__wr-on"]
    attempts_before = sorted(campaign.glob("runs/synthetic/*/*/attempts/*"))
    run_synthetic_campaign(
        tmp_path,
        "--methods",
        "atomic,depth",
        "--refinement",
        "on",
        "--resume",
    )
    assert sorted(campaign.glob("runs/synthetic/*/*/attempts/*")) == attempts_before


def test_campaign_metadata_redacts_secrets_and_has_runtime_provenance(tmp_path: Path):
    campaign = run_synthetic_campaign(tmp_path)
    payload = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    assert payload["source_commit"]
    assert type(payload["source_dirty"]) is bool
    assert payload["config_sha256"]
    assert payload["status"] == "succeeded"
    encoded = json.dumps(payload).lower()
    assert "password" not in encoded
    assert "cookie" not in encoded
    assert "hunter2" not in encoded


def test_run_dry_run_does_not_write_campaign_files(tmp_path: Path):
    output = tmp_path / "campaign-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "run",
            "--preset",
            "synthetic-smoke",
            "--output-root",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert not output.exists()


def test_synthetic_preflight_ignores_missing_data_checkpoint_and_gpu(tmp_path: Path):
    output = tmp_path / "campaign-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "preflight",
            "--allow-no-gpu",
            "--preset",
            "synthetic-smoke",
            "--data-root",
            str(tmp_path / "missing-data"),
            "--checkpoint",
            str(tmp_path / "missing-model.safetensors"),
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (output / "window-reference-v1/preflight.json").read_text(encoding="utf-8")
    )
    assert report["errors"] == []
    assert report["checks"][-1]["detail"]["not_applicable"] is True


def test_summarize_missing_campaign_is_nonzero_and_read_only(tmp_path: Path):
    output = tmp_path / "campaign-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "summarize",
            "--preset",
            "synthetic-smoke",
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    campaign = output / "window-reference-v1"
    assert (campaign / "summary/runs.csv").is_file()
    assert not (campaign / "work").exists()
    assert not (campaign / "runs").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--dry-run", "--no-resume"),
        ("--dry-run", "--keep-artifacts"),
        ("--dry-run", "--fail-fast"),
        ("--dry-run", "--keep-going"),
    ],
)
def test_dry_run_rejects_execution_only_options(tmp_path: Path, arguments: tuple[str, ...]):
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "run",
            "--preset",
            "synthetic-smoke",
            "--output-root",
            str(tmp_path / "out"),
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
