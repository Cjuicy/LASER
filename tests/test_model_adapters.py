from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from inference_engine.models.adapters import (
    REQUIRED_PREDICTION_KEYS,
    Pi3Adapter,
)
from pipeline.config import ModelName


def _complete_prediction(
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch, frames, _, height, width = images.shape
    local_points = torch.ones((batch, frames, height, width, 3))
    camera_poses = torch.eye(4).repeat(batch, frames, 1, 1)
    confidence = torch.arange(
        batch * frames * height * width,
        dtype=torch.float32,
    ).reshape(batch, frames, height, width)
    return {
        "points": local_points + 1,
        "local_points": local_points,
        "camera_poses": camera_poses,
        "conf": confidence,
    }


class RecordingPi3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor):
        self.inputs.append(images)
        return _complete_prediction(images)


def test_pi3_adapter_normalizes_four_dimensional_images():
    backend = RecordingPi3()
    images = torch.zeros((2, 3, 3, 4))

    prediction = Pi3Adapter(backend)(images)

    assert backend.inputs[0].shape == (1, 2, 3, 3, 4)
    assert tuple(prediction) == REQUIRED_PREDICTION_KEYS
    assert prediction["local_points"].shape == (1, 2, 3, 4, 3)
    assert prediction["camera_poses"].shape == (1, 2, 4, 4)
    assert prediction["conf"].shape == (1, 2, 3, 4)
    assert prediction["images"].shape == (1, 2, 3, 3, 4)
    assert "points" not in prediction


def test_pi3_adapter_preserves_five_dimensional_images():
    backend = RecordingPi3()
    images = torch.zeros((2, 3, 3, 4, 5))

    prediction = Pi3Adapter(backend)(images)

    assert backend.inputs[0] is images
    assert prediction["images"] is images
    assert prediction["conf"].shape == (2, 3, 4, 5)


@pytest.mark.parametrize(
    "images",
    (
        torch.zeros((3, 4, 5)),
        torch.zeros((1, 2, 3, 4, 5, 6)),
        torch.zeros((2, 1, 3, 4)),
    ),
)
def test_adapter_rejects_invalid_image_shapes_before_backend_call(images):
    backend = RecordingPi3()

    with pytest.raises(ValueError, match=r"\(B,N,3,H,W\)"):
        Pi3Adapter(backend)(images)

    assert backend.inputs == []


def _mutate_prediction(
    mutation: Callable[[dict[str, torch.Tensor]], object],
) -> RecordingPi3:
    class InvalidPi3(RecordingPi3):
        def forward(self, images: torch.Tensor):
            prediction = _complete_prediction(images)
            mutation(prediction)
            return prediction

    return InvalidPi3()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda prediction: prediction.__setitem__(
                "local_points",
                prediction["local_points"][:, :-1],
            ),
            "local_points",
        ),
        (
            lambda prediction: prediction.__setitem__(
                "camera_poses",
                prediction["camera_poses"][..., :3],
            ),
            "camera_poses",
        ),
        (
            lambda prediction: prediction.__setitem__(
                "conf",
                prediction["conf"][..., :-1],
            ),
            "conf",
        ),
        (
            lambda prediction: prediction.__setitem__("conf", "invalid"),
            "conf",
        ),
        (
            lambda prediction: prediction["local_points"].fill_(
                float("nan")
            ),
            "local_points",
        ),
        (
            lambda prediction: prediction["camera_poses"].fill_(
                float("inf")
            ),
            "camera_poses",
        ),
    ),
)
def test_adapter_rejects_malformed_or_non_finite_predictions(
    mutation,
    message,
):
    adapter = Pi3Adapter(_mutate_prediction(mutation))

    with pytest.raises(ValueError, match=message):
        adapter(torch.zeros((2, 3, 3, 4)))


def test_adapter_rejects_non_mapping_prediction():
    class InvalidPi3(torch.nn.Module):
        def forward(self, images):
            return images

    with pytest.raises(ValueError, match="mapping"):
        Pi3Adapter(InvalidPi3())(torch.zeros((2, 3, 3, 4)))


def test_adapter_exposes_pi3_model_name():
    assert Pi3Adapter(RecordingPi3()).model_name is ModelName.PI3
