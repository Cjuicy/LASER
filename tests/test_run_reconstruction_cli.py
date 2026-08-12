from __future__ import annotations

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
