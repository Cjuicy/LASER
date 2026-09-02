from __future__ import annotations

from pathlib import Path

import pytest

import run_reconstruction
from pipeline.config import ReconstructionMode, load_pipeline_config
from tests.test_pipeline_runner import _artifact


def test_reconstruction_cli_accepts_only_config_and_repeatable_set(monkeypatch, capsys):
    calls = []
    loaded = load_pipeline_config(
        "configs/reconstruction/pi3_laser_no_loop.yaml",
        ("window.size=20", "window.overlap=5"),
    )

    class Runner:
        artifact_dir = "results/artifact"

        def __init__(self, value):
            calls.append(value)

        def run(self):
            return _artifact(ReconstructionMode.NO_LOOP)

    monkeypatch.setattr(run_reconstruction, "load_pipeline_config", lambda *a: loaded)
    monkeypatch.setattr(run_reconstruction, "PipelineRunner", Runner)

    exit_code = run_reconstruction.main(
        [
            "--config",
            "config.yaml",
            "--set",
            "segmentation.method=geometry",
            "--set",
            "window.size=20",
        ]
    )

    assert exit_code == 0
    assert calls == [loaded]
    output = capsys.readouterr().out
    assert "mode=no_loop" in output
    assert "window=20" in output
    assert "overlap=5" in output
    assert f"config_hash={loaded.sha256}" in output
    assert "artifact_dir=results/artifact" in output


def test_reconstruction_cli_rejects_legacy_flags():
    with pytest.raises(SystemExit):
        run_reconstruction.build_parser().parse_args(
            ["--config", "config.yaml", "--loop-method", "corrected"]
        )


def test_reconstruction_cli_reports_both_staged_artifact_paths(
    monkeypatch,
    capsys,
):
    loaded = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml",
        (
            "reconstruction.mode=traditional_second_global",
            "loop.optimizer.implementation=python",
        ),
    )

    class Runner:
        artifact_dir = Path("results/artifact")
        stage_artifact_dirs = {
            "stage1": artifact_dir / "stage1",
            "stage2": artifact_dir,
        }

        def __init__(self, value):
            assert value is loaded

        def run(self):
            return _artifact(
                ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
            )

    monkeypatch.setattr(
        run_reconstruction,
        "load_pipeline_config",
        lambda *arguments: loaded,
    )
    monkeypatch.setattr(run_reconstruction, "PipelineRunner", Runner)

    assert run_reconstruction.main(["--config", "config.yaml"]) == 0
    output = capsys.readouterr().out
    assert "stage1_artifact_dir=results/artifact/stage1" in output
    assert "stage2_artifact_dir=results/artifact" in output
