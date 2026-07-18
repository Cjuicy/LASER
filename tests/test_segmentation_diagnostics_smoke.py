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
    assert "schema 2.0" in result.stdout
    assert "three methods: depth, geometry_baseline, layer_atomic_split" in result.stdout
    assert "four profiles" not in result.stdout
    assert (tmp_path / "verify" / "report" / "index.html").is_file()
    assert (tmp_path / "verify" / "cases" / "02" / "000000-000020" / "comparison-rendering.json").is_file()
