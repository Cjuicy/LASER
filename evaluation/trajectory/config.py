from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class TrajectoryEvaluationConfig:
    version: int = 1
    alignment: str = "sim3"
    rpe_delta_frames: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("trajectory evaluation version must be 1")
        if self.alignment != "sim3":
            raise ValueError("trajectory alignment must be sim3")
        if (
            isinstance(self.rpe_delta_frames, bool)
            or not isinstance(self.rpe_delta_frames, int)
            or self.rpe_delta_frames < 1
        ):
            raise ValueError("rpe_delta_frames must be a positive integer")


def load_trajectory_evaluation_config(
    path: str | Path,
) -> TrajectoryEvaluationConfig:
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise ValueError("trajectory evaluation config is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("trajectory evaluation config must be a mapping")
    allowed = {"version", "alignment", "rpe_delta_frames"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            "unknown trajectory evaluation field: " + ", ".join(unknown)
        )
    try:
        return TrajectoryEvaluationConfig(**payload)
    except TypeError as exc:
        raise ValueError("trajectory evaluation config is incomplete") from exc
