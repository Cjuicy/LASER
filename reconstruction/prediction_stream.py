from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from inference_engine.prediction_cache.types import WindowSpec


def _context(spec: WindowSpec) -> str:
    return f"window {spec.index} frames {spec.frame_start}:{spec.frame_end}"


@dataclass(frozen=True)
class WindowPrediction:
    spec: WindowSpec
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    images: torch.Tensor
    prediction_key: str

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        spec: WindowSpec,
        prediction_key: str,
        process_device: str,
    ) -> "WindowPrediction":
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{_context(spec)} prediction must be a mapping")
        if not isinstance(prediction_key, str) or not prediction_key:
            raise ValueError(f"{_context(spec)} prediction key must be non-empty")
        values = {}
        for public_name, source_name in (
            ("local_points", "local_points"),
            ("camera_poses", "camera_poses"),
            ("confidence", "conf"),
            ("images", "images"),
        ):
            value = mapping.get(source_name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"{_context(spec)} prediction is missing tensor {source_name}"
                )
            if value.ndim < 2 or value.shape[0] != 1:
                raise ValueError(
                    f"{_context(spec)} {public_name} must have one model batch"
                )
            value = value.squeeze(0).to(process_device)
            if value.shape[0] != spec.frame_count:
                raise ValueError(
                    f"{_context(spec)} {public_name} frame count mismatch"
                )
            if not value.is_floating_point():
                raise ValueError(
                    f"{_context(spec)} {public_name} must use floating dtype"
                )
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"{_context(spec)} {public_name} must contain finite values"
                )
            values[public_name] = value

        local_points = values["local_points"]
        camera_poses = values["camera_poses"]
        confidence = values["confidence"]
        window_images = values["images"]
        if local_points.ndim != 4 or local_points.shape[-1] != 3:
            raise ValueError(
                f"{_context(spec)} local_points must have shape (N,H,W,3)"
            )
        if camera_poses.shape != (spec.frame_count, 4, 4):
            raise ValueError(
                f"{_context(spec)} camera_poses must have shape (N,4,4)"
            )
        if confidence.shape != local_points.shape[:-1]:
            raise ValueError(
                f"{_context(spec)} confidence shape must match local_points"
            )
        if (
            window_images.ndim != 4
            or window_images.shape[1] != 3
            or window_images.shape[2:] != local_points.shape[1:3]
        ):
            raise ValueError(
                f"{_context(spec)} images must have shape (N,3,H,W)"
            )
        return cls(
            spec=spec,
            local_points=local_points,
            camera_poses=camera_poses,
            confidence=confidence,
            images=window_images,
            prediction_key=prediction_key,
        )


def _validate_specs(specs: Sequence[WindowSpec], frame_count: int):
    normalized = tuple(specs)
    if not normalized:
        raise ValueError("prediction stream requires at least one WindowSpec")
    for expected_index, spec in enumerate(normalized):
        if not isinstance(spec, WindowSpec) or spec.index != expected_index:
            raise ValueError("WindowSpecs must be in canonical order")
        if expected_index == 0 and spec.frame_start != 0:
            raise ValueError("WindowSpecs must be in canonical order")
        if expected_index > 0:
            previous = normalized[expected_index - 1]
            if not previous.frame_start < spec.frame_start < previous.frame_end:
                raise ValueError("WindowSpecs must be in canonical order")
        if spec.frame_end > frame_count:
            raise ValueError("WindowSpec exceeds prediction stream images")
    if normalized[-1].frame_end != frame_count:
        raise ValueError("WindowSpecs must cover all prediction stream images")
    return normalized


def iter_window_predictions(
    provider: object,
    specs: Sequence[WindowSpec],
    images: torch.Tensor,
    process_device: str,
) -> Iterable[WindowPrediction]:
    if (
        not isinstance(images, torch.Tensor)
        or images.ndim != 4
        or images.shape[1] != 3
        or not images.is_floating_point()
        or not torch.isfinite(images).all()
    ):
        raise ValueError("images must have finite shape (N,3,H,W)")
    normalized = _validate_specs(specs, int(images.shape[0]))
    get_prediction = getattr(provider, "get", None)
    if not callable(get_prediction):
        raise ValueError("prediction provider must define get(spec, images)")
    prediction_key = getattr(provider, "prediction_key", None)
    if not isinstance(prediction_key, str) or not prediction_key:
        raise ValueError("prediction provider must expose prediction_key")

    for spec in normalized:
        window_images = images[spec.frame_start : spec.frame_end]
        mapping = get_prediction(spec, window_images)
        yield WindowPrediction.from_mapping(
            mapping,
            spec,
            prediction_key,
            process_device,
        )
