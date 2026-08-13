from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/reconstruction-evaluation-cloud-validation.md"
SMOKE_BLOCK = re.compile(
    r"```bash\s*\n# smoke-test\s*\n(?P<body>.*?)```",
    re.DOTALL,
)


def _smoke_commands() -> tuple[str, ...]:
    text = DOCUMENT.read_text(encoding="utf-8")
    commands = []
    for match in SMOKE_BLOCK.finditer(text):
        for line in match.group("body").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line)
    return tuple(commands)


def test_documented_smoke_commands_execute(tmp_path):
    commands = _smoke_commands()
    assert commands
    allowed = {
        "run_reconstruction.py",
        "evaluate_ate.py",
        "evaluate_pointcloud.py",
        "run_experiment_matrix.py",
    }
    environment = {**os.environ, "LASER_SMOKE_TMP": str(tmp_path)}
    for command in commands:
        arguments = shlex.split(command)
        assert arguments[0] == "python"
        assert arguments[1] in allowed
        assert arguments[2:] == ["--help"]
        completed = subprocess.run(
            arguments,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, (
            f"command failed: {command}\n{completed.stdout}\n{completed.stderr}"
        )
