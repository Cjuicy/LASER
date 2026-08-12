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
