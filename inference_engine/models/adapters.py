from __future__ import annotations

from collections.abc import Mapping

import torch

from pipeline.config import ModelName


MODEL_ADAPTER_CONTRACT_VERSION = 1
REQUIRED_PREDICTION_KEYS = (
    "local_points",
    "camera_poses",
    "conf",
    "images",
)


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise ValueError("model images must be a tensor")
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("model images must have shape (B,N,3,H,W)")
    if not torch.isfinite(images).all():
        raise ValueError("model images must contain only finite values")
    return images


def _require_tensor(
    prediction: Mapping[str, object],
    key: str,
) -> torch.Tensor:
    value = prediction.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"model prediction {key!r} must be a tensor")
    return value


class ReconstructionModelAdapter(torch.nn.Module):
    model_name: ModelName

    def __init__(self, backend: torch.nn.Module) -> None:
        super().__init__()
        if not isinstance(backend, torch.nn.Module):
            raise ValueError("model adapter backend must be a torch.nn.Module")
        self.backend = backend

    def _call_backend(
        self,
        images: torch.Tensor,
    ) -> Mapping[str, object]:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized_images = _normalize_images(images)
        prediction = self._call_backend(normalized_images)
        if not isinstance(prediction, Mapping):
            raise ValueError("model prediction must be a mapping")

        local_points = _require_tensor(prediction, "local_points")
        camera_poses = _require_tensor(prediction, "camera_poses")
        confidence = _require_tensor(prediction, "conf")

        batch, frames, _, height, width = normalized_images.shape
        expected_shapes = {
            "local_points": (batch, frames, height, width, 3),
            "camera_poses": (batch, frames, 4, 4),
            "conf": (batch, frames, height, width),
        }
        tensors = {
            "local_points": local_points,
            "camera_poses": camera_poses,
            "conf": confidence,
        }
        for key, value in tensors.items():
            if tuple(value.shape) != expected_shapes[key]:
                raise ValueError(
                    f"model prediction {key!r} has shape "
                    f"{tuple(value.shape)}, expected {expected_shapes[key]}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"model prediction {key!r} must contain only finite values"
                )

        return {
            "local_points": local_points,
            "camera_poses": camera_poses,
            "conf": confidence,
            "images": normalized_images,
        }


class Pi3Adapter(ReconstructionModelAdapter):
    model_name = ModelName.PI3

    def _call_backend(
        self,
        images: torch.Tensor,
    ) -> Mapping[str, object]:
        return self.backend(images)
