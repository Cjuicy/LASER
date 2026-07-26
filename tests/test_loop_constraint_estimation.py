from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_closure.constraint_estimation import (
    JointPi3AlignmentEstimator,
    centered_frame_range,
)
from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopCandidate,
    WindowCache,
)
from pipeline.config import LoopMethod
from pipeline.manifest import ImageManifest


@pytest.mark.parametrize(
    ("bounds", "center", "chunk_size", "expected"),
    (
        ((4, 10), 8, 4, (6, 10)),
        ((4, 10), 4, 4, (4, 8)),
        ((4, 10), 6, 5, (4, 9)),
        ((4, 10), 7, 1, (7, 8)),
    ),
)
def test_centered_frame_range_returns_exact_bounded_neighborhood(
    bounds,
    center,
    chunk_size,
    expected,
):
    assert centered_frame_range(*bounds, center, chunk_size) == expected


class RecordingPi3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images.detach().cpu().clone())
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, frames, _, height, width = images.shape
        frame_ids = images[:, :, :1].permute(0, 1, 3, 4, 2)
        local_points = frame_ids.repeat(1, 1, 1, 1, 3)
        camera_poses = torch.eye(4).repeat(batch, frames, 1, 1)
        confidence = torch.ones((batch, frames, height, width))
        return {
            "points": local_points,
            "local_points": local_points,
            "camera_poses": camera_poses,
            "conf": confidence,
        }


class RegistrationCall:
    def __init__(self, source, target):
        self.source_frame_count = source.shape[0]
        self.target_frame_ids = target[:, 0, 0, 0].tolist()


def _cache(frame_start, frame_end, window_index):
    frames = frame_end - frame_start
    return WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=LoopMethod.TRADITIONAL,
        window_index=window_index,
        frame_start=frame_start,
        frame_end=frame_end,
        local_points=torch.zeros((frames, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(frames, 1, 1),
        confidence=torch.ones((frames, 1, 1)),
        segmentation_labels=tuple(
            np.zeros((1, 1), dtype=np.intp) for _ in range(frames)
        ),
        anchor_scale_mask=None,
        loop_state={"tag": LoopMethod.TRADITIONAL.value},
    )


def _images(frame_count=10):
    return torch.arange(frame_count, dtype=torch.float32).view(
        frame_count, 1, 1, 1
    ).repeat(1, 3, 1, 1)


def _manifest(frame_count=10):
    return ImageManifest(
        paths=tuple(Path(f"frame-{frame}.png") for frame in range(frame_count))
    )


def _estimator(model, images=None, manifest=None):
    return JointPi3AlignmentEstimator(
        model=model,
        images=_images() if images is None else images,
        manifest=_manifest() if manifest is None else manifest,
        inference_device="cpu",
        dtype=torch.float32,
        chunk_size=4,
        confidence_keep_ratio=0.5,
    )


def test_joint_estimator_infers_once_and_registers_each_cached_side(
    monkeypatch,
):
    model = RecordingPi3()
    registration_calls = []

    def record_registration(source, target, *_args):
        registration_calls.append(RegistrationCall(source, target))
        return 1.0, torch.eye(3), torch.zeros(3)

    monkeypatch.setattr(
        "loop_closure.constraint_estimation.register_adjacent_windows",
        record_registration,
    )
    estimator = _estimator(model)

    alignment_a, alignment_b = estimator(
        _cache(4, 10, 1),
        _cache(0, 6, 0),
        LoopCandidate(frame_a=8, frame_b=1, similarity=0.9),
        keep_ratio=0.5,
    )

    assert model.inputs[0][:, 0, 0, 0].tolist() == [
        6.0,
        7.0,
        8.0,
        9.0,
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    assert len(model.inputs) == 1
    assert len(registration_calls) == 2
    assert registration_calls[0].source_frame_count == 4
    assert registration_calls[0].target_frame_ids == [6.0, 7.0, 8.0, 9.0]
    assert registration_calls[1].source_frame_count == 4
    assert registration_calls[1].target_frame_ids == [0.0, 1.0, 2.0, 3.0]
    assert alignment_a[0] == 1.0
    assert alignment_b[0] == 1.0


def test_joint_estimator_rejects_image_and_manifest_length_mismatch():
    with pytest.raises(ValueError, match="image.*manifest.*length"):
        _estimator(RecordingPi3(), images=_images(9))


def test_joint_estimator_rejects_legacy_image_manifest_keyword():
    with pytest.raises(TypeError, match="image_manifest"):
        JointPi3AlignmentEstimator(
            model=RecordingPi3(),
            images=_images(),
            image_manifest=_manifest(),
            inference_device="cpu",
            dtype=torch.float32,
            chunk_size=4,
            confidence_keep_ratio=0.5,
        )


def test_joint_estimator_rejects_call_time_keep_ratio_mismatch():
    estimator = _estimator(RecordingPi3())

    with pytest.raises(ValueError, match="does not match"):
        estimator(
            _cache(4, 10, 1),
            _cache(0, 6, 0),
            LoopCandidate(frame_a=8, frame_b=1, similarity=0.9),
            keep_ratio=0.3,
        )
