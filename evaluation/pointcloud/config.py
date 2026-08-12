from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class PointCloudEvaluationConfig:
    version: int = 1
    center_crop_size: int = 224
    alignment: str = "umeyama_sim3_then_icp"
    icp_type: str = "point_to_point"
    icp_threshold_m: float = 0.1
    normal_estimation: str = "open3d_default"
    fscore_thresholds_m: tuple[float, ...] = (0.01, 0.02, 0.05)

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("point-cloud evaluation version must be 1")
        if (
            isinstance(self.center_crop_size, bool)
            or not isinstance(self.center_crop_size, int)
            or self.center_crop_size < 1
        ):
            raise ValueError("center_crop_size must be a positive integer")
        if self.alignment != "umeyama_sim3_then_icp":
            raise ValueError("point-cloud alignment must be umeyama_sim3_then_icp")
        if self.icp_type != "point_to_point":
            raise ValueError("point-cloud icp_type must be point_to_point")
        if (
            not math.isfinite(self.icp_threshold_m)
            or self.icp_threshold_m <= 0
        ):
            raise ValueError("icp_threshold_m must be finite and positive")
        if self.normal_estimation != "open3d_default":
            raise ValueError("normal_estimation must be open3d_default")
        thresholds = tuple(float(item) for item in self.fscore_thresholds_m)
        if not thresholds or any(
            not math.isfinite(item) or item <= 0 for item in thresholds
        ):
            raise ValueError("fscore_thresholds_m must be finite and positive")
        object.__setattr__(self, "fscore_thresholds_m", thresholds)


def load_pointcloud_evaluation_config(
    path: str | Path,
) -> PointCloudEvaluationConfig:
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise ValueError("point-cloud evaluation config is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("point-cloud evaluation config must be a mapping")
    allowed = {
        "version",
        "center_crop_size",
        "alignment",
        "icp_type",
        "icp_threshold_m",
        "normal_estimation",
        "fscore_thresholds_m",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            "unknown point-cloud evaluation field: " + ", ".join(unknown)
        )
    if "fscore_thresholds_m" in payload:
        payload["fscore_thresholds_m"] = tuple(payload["fscore_thresholds_m"])
    try:
        return PointCloudEvaluationConfig(**payload)
    except TypeError as exc:
        raise ValueError("point-cloud evaluation config is incomplete") from exc
