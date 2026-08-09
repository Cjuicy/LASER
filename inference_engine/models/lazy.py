from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch


class ModelForwardKind(str, Enum):
    ORDINARY = "ordinary"
    JOINT = "joint"


@dataclass
class ModelExecutionStats:
    model_constructed: bool = False
    ordinary_forward_count: int = 0
    joint_forward_count: int = 0


class LazyModelHandle:
    def __init__(
        self,
        factory: Callable[[], torch.nn.Module],
        *,
        inference_device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not callable(factory):
            raise ValueError("lazy model factory must be callable")
        if not isinstance(dtype, torch.dtype):
            raise ValueError("lazy model dtype must be a torch.dtype")
        self._factory = factory
        self.inference_device = torch.device(inference_device)
        self.dtype = dtype
        self.stats = ModelExecutionStats()
        self._model: torch.nn.Module | None = None

    def get(self) -> torch.nn.Module:
        if self._model is None:
            model = self._factory()
            if not isinstance(model, torch.nn.Module):
                raise ValueError(
                    "lazy model factory must return a torch.nn.Module"
                )
            self._model = model
            self.stats.model_constructed = True
        return self._model

    def predict(
        self,
        images: torch.Tensor,
        *,
        kind: ModelForwardKind,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(kind, ModelForwardKind):
            raise ValueError("model forward kind is invalid")
        if not isinstance(images, torch.Tensor):
            raise ValueError("model prediction images must be a tensor")

        model = self.get()
        if kind is ModelForwardKind.ORDINARY:
            self.stats.ordinary_forward_count += 1
        else:
            self.stats.joint_forward_count += 1

        device_type = self.inference_device.type
        autocast_context = (
            torch.autocast(device_type, dtype=self.dtype)
            if self.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            prediction = model(images.to(self.inference_device))
        if not isinstance(prediction, dict):
            raise ValueError("model prediction must be a dictionary")
        return prediction
