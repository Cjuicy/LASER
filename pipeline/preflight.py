from __future__ import annotations

import re
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.manifest import ImageManifest


SUPPORTED_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_DEVICE_PATTERN = re.compile(r"^(?:cpu|mps|cuda(?::\d+)?)$")


def _validate_device(field: str, value: str) -> None:
    if _DEVICE_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field} is unsupported: {value!r}; "
            "expected cpu, mps, cuda, or cuda:<index>"
        )


def validate_preflight(
    config: PipelineConfig,
    manifest: ImageManifest,
    cuda_available: bool,
) -> None:
    """Reject invalid runtime inputs before constructing any model."""

    if len(manifest) <= config.window.overlap:
        raise ValueError(
            "image manifest length must be greater than window.overlap "
            f"({len(manifest)} <= {config.window.overlap})"
        )

    if config.model.dtype not in SUPPORTED_DTYPES:
        supported = ", ".join(sorted(SUPPORTED_DTYPES))
        raise ValueError(
            f"model.dtype is unsupported: {config.model.dtype!r}; "
            f"expected one of {supported}"
        )
    _validate_device("model.inference_device", config.model.inference_device)
    _validate_device("model.process_device", config.model.process_device)

    requested_devices = (
        config.model.inference_device,
        config.model.process_device,
    )
    if not cuda_available and any(
        device.startswith("cuda") for device in requested_devices
    ):
        raise RuntimeError(
            "CUDA was requested by model device configuration "
            "but CUDA is unavailable"
        )

    required_files = [("model.checkpoint", config.model.checkpoint)]
    if config.loop.enabled:
        required_files.extend(
            [
                (
                    "loop.detection.salad_checkpoint",
                    config.loop.detection.salad_checkpoint,
                ),
                (
                    "loop.detection.dino_checkpoint",
                    config.loop.detection.dino_checkpoint,
                ),
            ]
        )

    for field, raw_path in required_files:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{field} does not exist or is not a file: {path}"
            )
