from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrajectoryMetrics:
    ate_rmse_m: float
    rpe_translation_rmse_m: float
    rpe_rotation_rmse_deg: float
    matched_frame_count: int

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.ate_rmse_m,
                self.rpe_translation_rmse_m,
                self.rpe_rotation_rmse_deg,
            )
        ):
            raise ValueError("trajectory metrics must be finite and non-negative")
        if self.matched_frame_count < 2:
            raise ValueError("matched_frame_count must be at least two")


def write_trajectory_metrics(
    metrics: TrajectoryMetrics,
    output_dir: str | Path,
    *,
    artifact_manifest_sha256: str,
) -> Path:
    if not isinstance(metrics, TrajectoryMetrics):
        raise ValueError("metrics must be TrajectoryMetrics")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "trajectory_metrics.json"
    payload = {
        **asdict(metrics),
        "artifact_manifest_sha256": artifact_manifest_sha256,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output,
            prefix=".trajectory_metrics.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target
