from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from inference_engine.prediction_cache.types import WindowSpec
from inference_engine.segmentation.base import SegmentationResult
from pipeline.config import SegmentationMethod


PREDICTION_KEY = "characterization-prediction-key"
CHECKPOINT_DIGEST = "a" * 64
SPECS = (
    WindowSpec(index=0, frame_start=0, frame_end=2),
    WindowSpec(index=1, frame_start=1, frame_end=3),
    WindowSpec(index=2, frame_start=2, frame_end=4),
)


def identity_sim3(scale: float = 1.0):
    return scale, torch.eye(3), torch.zeros(3)


class LiteralProvider:
    prediction_key = PREDICTION_KEY

    def get(self, spec: WindowSpec, images: torch.Tensor):
        if images.shape[0] != spec.frame_count:
            raise AssertionError("fixture images do not match WindowSpec")
        count = spec.frame_count
        points = torch.zeros((1, count, 1, 1, 3), dtype=torch.float32)
        points[..., 2] = 1.0
        return {
            "local_points": points,
            "camera_poses": torch.eye(4).repeat(1, count, 1, 1),
            "conf": torch.arange(count, dtype=torch.float32).view(
                1, count, 1, 1
            ),
            "images": images.unsqueeze(0),
        }


class OneRegionSegmenter:
    name = SegmentationMethod.DEPTH

    def segment(self, points, confidence, images):
        if not points.shape[0] == confidence.shape[0] == images.shape[0]:
            raise AssertionError("fixture segmentation frame axes differ")
        return [
            SegmentationResult(
                labels=np.zeros(points.shape[1:3], dtype=np.intp),
                diagnostics={"method": "depth", "region_count": 1},
            )
            for _ in points
        ]


@dataclass
class SequencedAnchor:
    scales: tuple[float, ...] = (5.0, 7.0)
    calls: list[tuple[float, float]] = field(default_factory=list)

    def propagate(
        self,
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap,
    ):
        del source_graphs, target_graphs, overlap
        self.calls.append(
            (
                float(source_points[-1, 0, 0, 2]),
                float(target_points[0, 0, 0, 2]),
            )
        )
        scale = self.scales[len(self.calls) - 1]
        return torch.full((*target_points.shape[:-1], 1), scale)


def literal_window(depth: float = 1.0) -> dict[str, torch.Tensor]:
    points = torch.zeros((1, 2, 1, 1, 3), dtype=torch.float32)
    points[..., 2] = depth
    return {
        "local_points": points,
        "camera_poses": torch.eye(4).repeat(1, 2, 1, 1),
        "conf": torch.ones((1, 2, 1, 1)),
        "images": torch.zeros((1, 2, 3, 1, 1)),
    }
