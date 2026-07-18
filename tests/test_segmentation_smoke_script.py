import subprocess
import sys
from pathlib import Path


def test_smoke_script_verifies_all_segmentation_modes():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/verify_segmentation_modes.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for mode in ("depth", "geometry", "layer_atomic", "layer_atomic_split"):
        assert f"[PASS] mode={mode} frames=2" in result.stdout


def test_diagnostics_cli_and_cloud_workflow_advertise_only_three_methods():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/run_segmentation_diagnostics.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    description = " ".join(result.stdout.split())
    assert "depth, geometry_baseline, and layer_atomic_split" in description

    cloud_workflow = (repo_root / "docs" / "segmentation-diagnostics-cloud.md").read_text(
        encoding="utf-8"
    )
    assert "[phase pass1 1/3] depth" in cloud_workflow
    assert "[phase pass1 2/3] geometry_baseline" in cloud_workflow
    assert "[phase pass1 3/3] layer_atomic_split" in cloud_workflow
    assert "[phase pass1 1/4]" not in cloud_workflow
