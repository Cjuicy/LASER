from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from inference_engine.models.adapters import Pi3Adapter
from inference_engine.models.lazy import (
    LazyModelHandle,
    ModelForwardKind,
)
from inference_engine.models.loader import (
    build_model_adapter,
    build_model_handle,
)
from pipeline.config import load_pipeline_config


class RecordingBackend(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.load_calls: list[tuple[dict[str, torch.Tensor], bool]] = []

    def load_state_dict(self, state_dict, strict=True):
        self.load_calls.append((state_dict, strict))
        return super().load_state_dict(
            {"marker": state_dict["marker"]},
            strict=strict,
        )

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, frames, _, height, width = images.shape
        return {
            "local_points": torch.ones(
                (batch, frames, height, width, 3)
            ),
            "camera_poses": torch.eye(4).repeat(
                batch,
                frames,
                1,
                1,
            ),
            "conf": torch.ones((batch, frames, height, width)),
        }


def _model_config(tmp_path: Path, suffix: str = ".pt"):
    loaded = load_pipeline_config("configs/pipeline/test.yaml")
    checkpoint = tmp_path / f"model{suffix}"
    state = {"marker": torch.tensor([3.0])}
    if suffix == ".safetensors":
        save_file(state, checkpoint)
    else:
        torch.save(state, checkpoint)
    return replace(
        loaded.config.model,
        checkpoint=str(checkpoint),
        inference_device="cpu",
        dtype="float32",
    )


@pytest.mark.parametrize("suffix", (".pt", ".safetensors"))
def test_pi3_loader_uses_strict_checkpoint_loading(
    tmp_path,
    monkeypatch,
    suffix,
):
    backend = RecordingBackend()
    monkeypatch.setattr(
        "inference_engine.models.loader._construct_pi3",
        lambda: backend,
    )

    adapter = build_model_adapter(_model_config(tmp_path, suffix))

    assert isinstance(adapter, Pi3Adapter)
    assert backend.load_calls[0][1] is True
    assert backend.load_calls[0][0]["marker"].item() == pytest.approx(3.0)
    assert adapter.training is False
    assert backend.training is False
    assert backend.marker.device.type == "cpu"


def test_lazy_handle_constructs_once_and_counts_forward_kinds():
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return Pi3Adapter(RecordingBackend())

    handle = LazyModelHandle(
        factory,
        inference_device="cpu",
        dtype=torch.float32,
    )
    assert factory_calls == 0
    assert handle.stats.model_constructed is False

    images = torch.zeros((2, 3, 3, 4))
    handle.predict(images, kind=ModelForwardKind.ORDINARY)
    handle.predict(images, kind=ModelForwardKind.JOINT)

    assert factory_calls == 1
    assert handle.stats.model_constructed is True
    assert handle.stats.ordinary_forward_count == 1
    assert handle.stats.joint_forward_count == 1


def test_lazy_handle_rejects_invalid_forward_kind_before_construction():
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return Pi3Adapter(RecordingBackend())

    handle = LazyModelHandle(
        factory,
        inference_device="cpu",
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="forward kind"):
        handle.predict(torch.zeros((1, 3, 2, 2)), kind="ordinary")

    assert factory_calls == 0


def test_build_model_handle_is_lazy(tmp_path, monkeypatch):
    construction_calls = []

    def build(config):
        construction_calls.append(config)
        return Pi3Adapter(RecordingBackend())

    monkeypatch.setattr(
        "inference_engine.models.loader.build_model_adapter",
        build,
    )
    config = _model_config(tmp_path)

    handle = build_model_handle(config)
    assert construction_calls == []

    handle.predict(
        torch.zeros((1, 3, 2, 2)),
        kind=ModelForwardKind.ORDINARY,
    )
    assert construction_calls == [config]
