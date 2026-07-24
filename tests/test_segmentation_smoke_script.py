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
    for method in ("depth", "geometry", "atomic"):
        assert f"[PASS] method={method} frames=2" in result.stdout
