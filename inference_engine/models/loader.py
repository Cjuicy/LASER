from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from pipeline.config import ModelConfig, ModelName

from .adapters import Pi3Adapter
from .lazy import LazyModelHandle


def _construct_pi3() -> torch.nn.Module:
    from pi3.models.pi3 import Pi3

    return Pi3()


def _load_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    if path.suffix.casefold() == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(path), device="cpu")
    else:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("model checkpoint must contain a state-dict mapping")
    return checkpoint


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype: {name!r}") from exc


def build_model_adapter(config: ModelConfig) -> Pi3Adapter:
    if config.name is not ModelName.PI3:
        raise ValueError(f"unsupported model name: {config.name!r}")
    checkpoint = _load_checkpoint(Path(config.checkpoint))
    try:
        backend = _construct_pi3()
        backend.load_state_dict(checkpoint, strict=True)
        adapter = Pi3Adapter(backend)
    finally:
        del checkpoint
    return adapter.to(config.inference_device).eval()


def build_model_handle(config: ModelConfig) -> LazyModelHandle:
    return LazyModelHandle(
        lambda: build_model_adapter(config),
        inference_device=config.inference_device,
        dtype=_resolve_dtype(config.dtype),
    )
