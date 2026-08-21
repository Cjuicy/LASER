from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from inference_engine.prediction_cache.types import WindowSpec
from reconstruction.prediction_stream import (
    WindowPrediction,
    iter_window_predictions,
)


SPECS = (
    WindowSpec(0, 0, 2),
    WindowSpec(1, 1, 3),
    WindowSpec(2, 2, 4),
)


def _mapping():
    return {
        "local_points": torch.zeros((1, 2, 2, 2, 3)),
        "camera_poses": torch.eye(4).repeat(1, 2, 1, 1),
        "conf": torch.ones((1, 2, 2, 2)),
        "images": torch.zeros((1, 2, 3, 2, 2)),
    }


def test_window_prediction_allows_absent_reference_intrinsic():
    prediction = WindowPrediction.from_mapping(
        _mapping(),
        SPECS[0],
        "key",
        "cpu",
    )

    assert prediction.reference_intrinsic is None


def test_window_prediction_moves_reference_intrinsic_to_process_device():
    intrinsic = torch.tensor(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    )
    mapping = {**_mapping(), "reference_intrinsic": intrinsic}

    prediction = WindowPrediction.from_mapping(
        mapping,
        SPECS[0],
        "key",
        "cpu",
    )

    assert torch.equal(prediction.reference_intrinsic, intrinsic)
    assert prediction.reference_intrinsic.device.type == "cpu"
    intrinsic[0, 0] = -123.0
    assert prediction.reference_intrinsic[0, 0].item() > 0.0


@pytest.mark.parametrize(
    ("name", "intrinsic", "match"),
    [
        ("shape", torch.ones((2, 3)), "shape"),
        ("dtype", torch.ones((3, 3), dtype=torch.int64), "floating dtype"),
        (
            "finiteness",
            torch.tensor(
                [
                    [2.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 0.0, float("nan")],
                ],
            ),
            "finite",
        ),
        (
            "focal lengths",
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            "focal lengths",
        ),
        (
            "last row",
            torch.tensor(
                [
                    [2.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 0.0, 2.0],
                ],
            ),
            "last row",
        ),
    ],
)
def test_window_prediction_rejects_invalid_reference_intrinsic(
    name,
    intrinsic,
    match,
):
    del name

    with pytest.raises(ValueError, match=match):
        WindowPrediction.from_mapping(
            {**_mapping(), "reference_intrinsic": intrinsic},
            SPECS[0],
            "key",
            "cpu",
        )


class CompleteRecordingProvider:
    def __init__(self, prediction_key: str):
        self.prediction_key = prediction_key
        self.specs = []
        self.image_slices = []

    def get(self, spec, images):
        self.specs.append(spec)
        self.image_slices.append(images.clone())
        frames, _, height, width = images.shape
        points = torch.zeros((1, frames, height, width, 3))
        points[..., 2] = spec.index + 1.0
        return {
            "local_points": points,
            "camera_poses": torch.eye(4).repeat(1, frames, 1, 1),
            "conf": torch.ones((1, frames, height, width)),
            "images": images.unsqueeze(0),
        }


class NonfiniteCompleteProvider(CompleteRecordingProvider):
    def get(self, spec, images):
        prediction = super().get(spec, images)
        if spec.index == 1:
            prediction["local_points"][0, 0, 0, 0, 0] = torch.nan
        return prediction


def test_stream_yields_canonical_typed_predictions_once():
    provider = CompleteRecordingProvider(prediction_key="p" * 64)
    images = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(
        4, 3, 2, 2
    )

    predictions = tuple(
        iter_window_predictions(
            provider,
            SPECS,
            images,
            process_device="cpu",
        )
    )

    assert [item.spec for item in predictions] == list(SPECS)
    assert provider.specs == list(SPECS)
    assert all(isinstance(item, WindowPrediction) for item in predictions)
    assert all(item.prediction_key == "p" * 64 for item in predictions)
    assert predictions[0].local_points.shape == (2, 2, 2, 3)
    assert torch.equal(provider.image_slices[1], images[1:3])


def test_stream_rejects_nonfinite_prediction_with_window_context():
    provider = NonfiniteCompleteProvider(prediction_key="p" * 64)
    images = torch.zeros((4, 3, 2, 2))

    with pytest.raises(
        ValueError,
        match=r"window 1 frames 1:3.*local_points.*finite",
    ):
        tuple(iter_window_predictions(provider, SPECS, images, "cpu"))


def test_stream_rejects_noncanonical_specs_before_provider_call():
    provider = CompleteRecordingProvider(prediction_key="p" * 64)
    invalid = (WindowSpec(0, 0, 2), WindowSpec(2, 1, 3))

    with pytest.raises(ValueError, match="canonical order"):
        tuple(
            iter_window_predictions(
                provider,
                invalid,
                torch.zeros((3, 3, 2, 2)),
                "cpu",
            )
        )
    assert provider.specs == []


def test_stream_rejects_partial_provider_mapping():
    class PartialProvider(CompleteRecordingProvider):
        def get(self, spec, images):
            complete = super().get(spec, images)
            del complete["camera_poses"]
            return complete

    with pytest.raises(ValueError, match="camera_poses"):
        tuple(
            iter_window_predictions(
                PartialProvider("p" * 64),
                SPECS,
                torch.zeros((4, 3, 2, 2)),
                "cpu",
            )
        )


def test_provider_exposes_read_only_prediction_key():
    from inference_engine.prediction_cache.provider import (
        OrdinaryPredictionProvider,
    )

    provider = object.__new__(OrdinaryPredictionProvider)
    provider.store = SimpleNamespace(
        fingerprint=SimpleNamespace(key="k" * 64)
    )
    assert provider.prediction_key == "k" * 64
    with pytest.raises(AttributeError):
        provider.prediction_key = "changed"
