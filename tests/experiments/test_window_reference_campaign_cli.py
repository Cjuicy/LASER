"""Subprocess contract tests for the complete campaign command line."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    assert payload["source_commit"] == payload["preflight"]["git"]["commit"]
    assert payload["source_dirty"] == payload["preflight"]["git"]["dirty"]
    assert payload["config_sha256"]
    assert payload["status"] == "succeeded"
    assert payload["preflight"]["status"] in {"ok", "ok_with_warnings"}
    check_names = {item["name"] for item in payload["preflight"]["checks"]}
    assert {"storage", "plan:run-directories", "identity:synthetic-fixture"} <= check_names
    assert len(payload["preflight"]["identity_seed_sha256"]) == 6
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_synthetic_run_preflight_error_aborts_before_any_campaign_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The run handler must honor a controlled synthetic preflight failure."""

    from experiments.window_reference_campaign import cli, preflight

    output = tmp_path / "campaign-output"
    calls: list[tuple[object, object]] = []

    def controlled_failure(loaded, plan, *, allow_no_gpu):
        calls.append((loaded, plan))
        assert allow_no_gpu is True
        return SimpleNamespace(
            status="error",
            errors=("controlled synthetic preflight failure",),
            warnings=(),
            to_payload=lambda: {
                "schema_version": 1,
                "status": "error",
                "allow_no_gpu": True,
                "checks": [],
                "warnings": [],
                "errors": ["controlled synthetic preflight failure"],
                "git": {"commit": "unknown", "dirty": False},
                "checkpoint_sha256": "",
                "free_disk_bytes": 0,
                "identity_seed_sha256": [],
            },
        )

    monkeypatch.setattr(preflight, "preflight_campaign", controlled_failure)
    code = cli.main(
        [
            "run",
            "--preset",
            "synthetic-smoke",
            "--data-root",
            str(tmp_path / "missing-data"),
            "--checkpoint",
            str(tmp_path / "missing-model.safetensors"),
            "--output-root",
            str(output),
        ]
    )
    assert code != 0
    assert calls
    assert not output.exists()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "blocked_module",
    (
        "numpy",
        "torch",
        "inference_engine.segmentation",
        "inference_engine.prediction_cache.fingerprint",
        "pipeline.config",
    ),
)
def test_synthetic_run_checks_runtime_imports_before_any_campaign_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_module: str,
):
    from experiments.window_reference_campaign import cli, preflight

    output = tmp_path / "campaign-output"
    base = replace(
        preflight.default_preflight_dependencies(),
        python_version=(3, 11, 15),
    )
    calls: list[str] = []

    def blocked_import(name: str):
        calls.append(name)
        if name == blocked_module:
            raise ImportError(f"controlled synthetic {blocked_module} failure")
        return base.import_module(name)

    monkeypatch.setattr(
        preflight,
        "default_preflight_dependencies",
        lambda: replace(base, import_module=blocked_import),
    )
    code = cli.main(
        [
            "run",
            "--preset",
            "synthetic-smoke",
            "--data-root",
            str(tmp_path / "missing-data"),
            "--checkpoint",
            str(tmp_path / "missing-model.safetensors"),
            "--output-root",
            str(output),
        ]
    )
    assert code != 0
    assert blocked_module in calls
    assert not {"open3d", "evo", "lietorch", "pi3"} & set(calls)
    assert not output.exists()


def test_synthetic_frame_override_is_rejected_without_writes(tmp_path: Path):
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
            "--max-frames",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert not output.exists()


def test_default_campaign_root_is_ignored_without_changing_git_source_state():
    default_root = Path("outputs/window_reference_campaign")
    check = subprocess.run(
        ["git", "check-ignore", "--no-index", f"{default_root.as_posix()}/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr

    def source_state() -> str:
        return subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    before = source_state()
    created = not default_root.exists()
    if created:
        marker = default_root / "window-reference-v1" / "campaign.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}\n", encoding="utf-8")
    try:
        assert source_state() == before
    finally:
        if created:
            marker.unlink()
            marker.parent.rmdir()
            default_root.rmdir()


def _synthetic_cache_entry(campaign: Path) -> Path:
    entries = sorted((campaign / "work/cache/synthetic/f000000-000004-s1/v2").iterdir())
    assert len(entries) == 1
    return entries[0]


def _synthetic_prediction_key(campaign: Path) -> str:
    record = json.loads(
        (campaign / "runs/synthetic/synthetic-fixture/f000000-000004-s1/depth__wr-on/run.json")
        .read_text(encoding="utf-8")
    )
    return record["identity"]["prediction_key"]


def _write_complete_synthetic_cache(campaign: Path, prediction_key: str) -> Path:
    entry = campaign / "work/cache/synthetic/f000000-000004-s1/v2" / prediction_key
    (entry / "windows").mkdir(parents=True, exist_ok=True)
    (entry / "manifest.json").write_text(
        json.dumps({"synthetic": True, "key": prediction_key}) + "\n",
        encoding="utf-8",
    )
    (entry / "sequence.json").write_text(
        json.dumps({"synthetic": True, "key": prediction_key}) + "\n",
        encoding="utf-8",
    )
    for index in range(2):
        (entry / "windows" / f"{index:06d}.pt").write_bytes(b"synthetic-window-v1")
    (entry / "complete.json").write_text(
        json.dumps({"schema_version": 2, "key": prediction_key, "window_count": 2})
        + "\n",
        encoding="utf-8",
    )
    return entry


def _remove_first_synthetic_record(campaign: Path) -> None:
    path = campaign / (
        "runs/synthetic/synthetic-fixture/f000000-000004-s1/depth__wr-off/run.json"
    )
    path.unlink()


def test_synthetic_partial_cache_is_rebuilt_by_auto_and_refresh(tmp_path: Path):
    campaign = run_synthetic_campaign(tmp_path, "--keep-artifacts")
    entry = _synthetic_cache_entry(campaign)
    (entry / "manifest.json").unlink()
    _remove_first_synthetic_record(campaign)

    resumed = run_synthetic_campaign(tmp_path, "--resume", "--keep-artifacts")
    assert resumed == campaign
    assert (entry / "manifest.json").is_file()
    record = json.loads(
        (campaign / "runs/synthetic/synthetic-fixture/f000000-000004-s1/depth__wr-off/run.json")
        .read_text(encoding="utf-8")
    )
    assert record["cache_stats"]["ordinary_hits"] == 0
    assert record["cache_stats"]["ordinary_misses"] >= 1

    refresh_campaign = run_synthetic_campaign(
        tmp_path / "refresh",
        "--keep-artifacts",
        "--cache-policy",
        "refresh",
    )
    refresh_entry = _synthetic_cache_entry(refresh_campaign)
    (refresh_entry / "sequence.json").unlink()
    _remove_first_synthetic_record(refresh_campaign)
    refreshed = run_synthetic_campaign(
        tmp_path / "refresh",
        "--resume",
        "--keep-artifacts",
        "--cache-policy",
        "refresh",
    )
    assert refreshed == refresh_campaign
    assert (refresh_entry / "sequence.json").is_file()


def test_synthetic_readonly_partial_cache_fails_without_reporting_hit(tmp_path: Path):
    campaign = run_synthetic_campaign(
        tmp_path,
        "--keep-artifacts",
        "--cache-policy",
        "readonly",
        check=False,
    )
    failed_record = campaign / (
        "runs/synthetic/synthetic-fixture/f000000-000004-s1/depth__wr-off/run.json"
    )
    assert json.loads(failed_record.read_text(encoding="utf-8"))["status"] == "failed"
    prediction_key = json.loads(
        (campaign / (
            "runs/synthetic/synthetic-fixture/f000000-000004-s1/depth__wr-off/artifact/synthetic.json"
        )).read_text(encoding="utf-8")
    )["prediction_key"]
    entry = _write_complete_synthetic_cache(campaign, prediction_key)
    completed = run_synthetic_campaign(
        tmp_path,
        "--resume",
        "--keep-artifacts",
        "--cache-policy",
        "readonly",
    )
    assert completed == campaign
    _remove_first_synthetic_record(campaign)
    (entry / "windows/000001.pt").unlink()

    failed = run_synthetic_campaign(
        tmp_path,
        "--resume",
        "--keep-artifacts",
        "--cache-policy",
        "readonly",
        check=False,
    )
    assert failed == campaign
    record = json.loads(
        (campaign / "runs/synthetic/synthetic-fixture/f000000-000004-s1/depth__wr-off/run.json")
        .read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["cache_stats"]["ordinary_hits"] == 0


def test_synthetic_cache_off_does_not_read_or_repair_partial_entry(tmp_path: Path):
    campaign = run_synthetic_campaign(
        tmp_path,
        "--keep-artifacts",
        "--cache-policy",
        "off",
    )
    entry = _write_complete_synthetic_cache(campaign, _synthetic_prediction_key(campaign))
    (entry / "manifest.json").unlink()
    _remove_first_synthetic_record(campaign)

    resumed = run_synthetic_campaign(
        tmp_path,
        "--resume",
        "--keep-artifacts",
        "--cache-policy",
        "off",
    )
    assert resumed == campaign
    assert not (entry / "manifest.json").exists()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_synthetic_cache_probe_rebuilds_wrong_complete_metadata(tmp_path: Path):
    from experiments.window_reference_campaign.runner import cache_entry_complete

    campaign = run_synthetic_campaign(tmp_path, "--keep-artifacts")
    entry = _synthetic_cache_entry(campaign)
    prediction_key = entry.name
    complete_path = entry / "complete.json"
    mutations = (
        ("key", "0" * 64),
        ("schema_version", 1),
        ("window_count", 1),
    )
    for field, value in mutations:
        _write_complete_synthetic_cache(campaign, prediction_key)
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        payload[field] = value
        complete_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        _remove_first_synthetic_record(campaign)
        run_synthetic_campaign(tmp_path, "--resume", "--keep-artifacts")
        assert cache_entry_complete(campaign / "work/cache/synthetic/f000000-000004-s1", prediction_key, 2)


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
