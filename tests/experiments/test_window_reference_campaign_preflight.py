from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.window_reference_campaign.config import (
    CampaignOverrides,
    load_campaign_config,
)
from experiments.window_reference_campaign.matrix import build_plan
from experiments.window_reference_campaign.preflight import (
    CudaState,
    DiskUsage,
    GitState,
    PreflightDependencies,
    build_bootstrap_actions,
    preflight_campaign,
    run_bootstrap,
    write_preflight_report,
)
from experiments.window_reference_campaign.results import redact_argv


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/window_reference_campaign.yaml"


def make_preflight_fixture(tmp_path: Path, *, preset: str = "kitti-smoke"):
    data_root = tmp_path / "data"
    image_dir = data_root / "KITTI" / "04" / "image_2"
    image_dir.mkdir(parents=True)
    for index in range(80):
        Image.new("RGB", (8, 6), color=(index % 255, 0, 0)).save(
            image_dir / f"{index:06d}.png"
        )
    np.savetxt(data_root / "KITTI" / "04" / "poses.txt", np.ones((80, 12)))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture-checkpoint")
    loaded = load_campaign_config(
        CONFIG,
        CampaignOverrides(
            preset=preset,
            data_root=data_root,
            checkpoint=checkpoint,
            output_root=tmp_path / "outputs",
        ),
    )
    return loaded, build_plan(loaded)


def _dependencies(*, cuda_available: bool) -> PreflightDependencies:
    return PreflightDependencies.for_tests(
        python_version=(3, 11, 15),
        git=GitState("1" * 40, False),
        cuda=CudaState(
            available=cuda_available,
            device_count=1 if cuda_available else 0,
            selected_device=0 if cuda_available else None,
            device_name="fixture-gpu" if cuda_available else None,
            bfloat16_supported=True if cuda_available else None,
        ),
        free_bytes=100 * 1024**3,
    )


def test_bootstrap_dry_run_works_when_torch_and_omegaconf_imports_are_blocked(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname in {'torch', 'omegaconf'} or fullname.startswith(('torch.', 'omegaconf.')):\n"
        "      raise RuntimeError('heavy import forbidden during bootstrap')\n"
        "sys.meta_path.insert(0, Block())\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_window_reference_campaign.py"),
            "bootstrap",
            "--repository",
            str(ROOT),
            "--dry-run",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "git submodule update --init --recursive" in completed.stdout
    assert "setup.py build_ext --inplace" in completed.stdout


def test_bootstrap_actions_never_download_protected_datasets(tmp_path):
    actions = build_bootstrap_actions(
        ROOT,
        python_executable=sys.executable,
        external_checkpoint=None,
    )
    commands = "\n".join(shlex.join(action.argv) for action in actions)
    assert "scripts/download_weights.sh" in commands
    lowered = commands.lower()
    for forbidden in ("kitti", "7-scenes", "7scenes", "neuralrgbd", "cookie", "password", "token"):
        assert forbidden not in lowered


def test_run_bootstrap_prints_shell_quoted_commands_and_executes_in_order(tmp_path, capsys):
    actions = build_bootstrap_actions(
        tmp_path,
        python_executable="python with space",
        external_checkpoint=tmp_path / "checkpoint.safetensors",
    )
    assert run_bootstrap(actions, execute=False) == 0
    output = capsys.readouterr().out
    assert "cd " in output
    assert "'python with space'" in output

    calls = []

    def run(argv, *, cwd, check):
        calls.append((argv, cwd, check))
        return subprocess.CompletedProcess(argv, 0)

    assert run_bootstrap(actions, execute=True, run=run) == 0
    assert [item[0] for item in calls] == [action.argv for action in actions]
    assert all(item[1] == tmp_path.resolve() and item[2] is True for item in calls)


def test_allow_no_gpu_succeeds_but_records_not_gpu_ready(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "ok_with_warnings"
    assert report.errors == ()
    assert any("CUDA" in warning for warning in report.warnings)
    cuda = next(check for check in report.checks if check.name == "cuda")
    assert cuda.status == "warning"
    assert cuda.detail["gpu_ready"] is False


def test_strict_preflight_rejects_same_no_gpu_environment(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=False,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "error"
    assert any("CUDA" in error for error in report.errors)


def test_allow_no_gpu_does_not_downgrade_import_or_bfloat16_errors(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    base = _dependencies(cuda_available=True)

    def imports(name):
        if name in {"torch", "open3d"}:
            raise RuntimeError(f"missing {name}")
        return base.import_module(name)

    dependencies = replace(
        base,
        import_module=imports,
        cuda_state=lambda _: CudaState(True, 1, 0, "fixture-gpu", False),
    )
    report = preflight_campaign(
        loaded, plan, allow_no_gpu=True, dependencies=dependencies
    )
    assert report.status == "error"
    assert any("required import torch" in error for error in report.errors)
    assert any("required import open3d" in error for error in report.errors)
    assert any("bfloat16" in error for error in report.errors)
    assert next(check for check in report.checks if check.name == "cuda").status == "error"


def test_preflight_collects_all_required_import_failures(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    base = _dependencies(cuda_available=False)

    def imports(name):
        raise RuntimeError(f"blocked {name}")

    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=replace(base, import_module=imports),
    )
    assert report.status == "error"
    assert len([item for item in report.errors if item.startswith("required import")]) == 9
    assert any(check.status == "warning" for check in report.checks if check.name == "cuda")


def test_allow_no_gpu_does_not_hide_missing_checkpoint_or_dataset(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    loaded = replace(
        loaded,
        config=replace(
            loaded.config,
            storage=replace(
                loaded.config.storage,
                checkpoint=tmp_path / "missing.safetensors",
                data_root=tmp_path / "missing-data",
            ),
        ),
    )
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "error"
    assert any("checkpoint" in error for error in report.errors)
    assert any("KITTI" in error or "data" in error for error in report.errors)


def test_preflight_writes_only_atomic_report_and_warns_for_space_dirty_source(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    dependencies = replace(
        _dependencies(cuda_available=False),
        git_state=lambda _: GitState("2" * 40, True),
        disk_usage=lambda _: DiskUsage(
            100 * 1024**3, 90 * 1024**3, 10 * 1024**3
        ),
    )
    report = preflight_campaign(
        loaded, plan, allow_no_gpu=True, dependencies=dependencies
    )
    target = write_preflight_report(report, loaded.config.campaign_root)
    assert target == loaded.config.campaign_root / "preflight.json"
    created = sorted(
        path.relative_to(loaded.config.campaign_root).as_posix()
        for path in loaded.config.campaign_root.rglob("*") if path.is_file()
    )
    assert created == ["preflight.json"]
    assert any("20.0 GiB" in warning for warning in report.warnings)
    assert any("dirty" in warning for warning in report.warnings)
    assert len(report.identity_seed_sha256) == 6
    assert len(set(report.identity_seed_sha256)) == 6


def test_persisted_argv_redacts_options_and_url_userinfo():
    assert redact_argv((
        "--token", "abc",
        "--password=hunter2",
        "https://alice:secret@example.invalid/repo.git",
        "--preset", "kitti-small",
    )) == (
        "--token", "<redacted>",
        "--password=<redacted>",
        "https://<redacted>@example.invalid/repo.git",
        "--preset", "kitti-small",
    )
