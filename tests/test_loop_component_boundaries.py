from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from loop_closure.detection import SaladLoopDetector
from loop_closure.evidence import LoopWindow
from loop_closure.types import LoopCandidate, validate_sim3
from pipeline.config import load_pipeline_config
from pipeline.manifest import ImageManifest


class CompleteSaladBackend:
    def __init__(self):
        self.descriptors = np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            dtype=np.float32,
        )
        self.frame_indices = (0, 2, 12)
        self.similarities = (0.8,)
        self.calls = []

    def detect(self, config, manifest, images, output_path):
        self.calls.append((config, manifest, images, output_path))
        return ((12, 2, self.similarities[0]),)


def test_loop_detector_returns_typed_candidates_from_manifest(tmp_path):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml"
    ).config.loop.detection
    manifest = ImageManifest(
        paths=tuple(Path(f"frame-{index}.png") for index in range(13))
    )
    images = torch.zeros((13, 3, 2, 2))
    backend = CompleteSaladBackend()
    detector = SaladLoopDetector(
        config,
        output_path=tmp_path / "candidates.txt",
        backend=backend,
    )

    result = detector.detect(manifest, images)

    assert result == (
        LoopCandidate(frame_a=12, frame_b=2, similarity=0.8),
    )
    assert backend.calls == [
        (config, manifest, images, tmp_path / "candidates.txt")
    ]


@dataclass(frozen=True)
class CompleteLoopWindow:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor


def test_loop_window_protocol_accepts_mode_neutral_state():
    window = CompleteLoopWindow(
        window_index=0,
        frame_start=0,
        frame_end=2,
        local_points=torch.ones((2, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
    )

    assert isinstance(window, LoopWindow)
    transform = (1.0, torch.eye(3), torch.zeros(3))
    validate_sim3(transform)


def test_loop_detector_rejects_image_manifest_length_mismatch(tmp_path):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml"
    ).config.loop.detection
    manifest = ImageManifest(paths=(Path("frame-0.png"), Path("frame-1.png")))
    detector = SaladLoopDetector(
        config,
        output_path=tmp_path / "candidates.txt",
        backend=CompleteSaladBackend(),
    )

    try:
        detector.detect(manifest, torch.zeros((1, 3, 2, 2)))
    except ValueError as error:
        assert "image and manifest lengths" in str(error)
    else:
        raise AssertionError("detector accepted mismatched images")
