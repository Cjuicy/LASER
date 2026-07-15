import subprocess
import sys


def test_cpu_only_diagnostic_verifier(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/verify_segmentation_diagnostics.py", "--output-dir", str(tmp_path / "verify")],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("schema", "parity", "storage", "selection", "rendering", "report"):
        assert f"[PASS] {name}" in result.stdout
    assert (tmp_path / "verify" / "report" / "index.html").is_file()
