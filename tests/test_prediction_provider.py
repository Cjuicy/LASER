from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from inference_engine.inference_utils import (
    unproject_depth_to_local_points,
)
from inference_engine.models.lazy import (
    LazyModelHandle,
    ModelForwardKind,
)
from inference_engine.prediction_cache.fingerprint import (
    PredictionFingerprint,
)
from inference_engine.prediction_cache.provider import (
    OrdinaryPredictionProvider,
)
from inference_engine.prediction_cache.store import (
    OrdinaryPredictionStore,
    PredictionCacheCorruptError,
    PredictionCacheMissError,
    _sequence_artifact_digest,
)
from inference_engine.prediction_cache.types import (
    PREDICTION_CACHE_SCHEMA_VERSION,
    WindowSpec,
)
from pipeline.config import PredictionCacheMode


SPECS = (WindowSpec(0, 0, 2), WindowSpec(1, 1, 3))
IMAGES = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(
    3,
    3,
    4,
    4,
)


def _fingerprint():
    return PredictionFingerprint(
        key="e" * 64,
        checkpoint_sha256="f" * 64,
        image_manifest_sha256="1" * 64,
        runtime_source_sha256="2" * 64,
        canonical_payload={
            "model_name": "pi3",
            "image_shape": [3, 3, 4, 4],
            "window_specs": [spec.to_payload() for spec in SPECS],
        },
    )


def _store(tmp_path, mode=PredictionCacheMode.AUTO):
    return OrdinaryPredictionStore(
        root=tmp_path / "predictions",
        fingerprint=_fingerprint(),
        mode=mode,
        expected_specs=SPECS,
    )


class CountingAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs = []
        self.intrinsic = torch.tensor(
            [
                [2.0, 0.0, 2.0],
                [0.0, 2.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        self.inputs.append(images.detach().cpu().clone())
        batch, frames, _, height, width = images.shape
        depth = (
            torch.arange(frames, dtype=torch.float32)
            .view(frames, 1, 1)
            .expand(frames, height, width)
            + 2.0
        )
        local_points = unproject_depth_to_local_points(
            depth,
            self.intrinsic,
        ).unsqueeze(0).expand(batch, -1, -1, -1, -1)
        return {
            "local_points": local_points,
            "camera_poses": torch.eye(4).repeat(
                batch,
                frames,
                1,
                1,
            ),
            "conf": torch.ones((batch, frames, height, width)),
            "images": images,
        }


def _handle(construction_counter):
    def factory():
        construction_counter.append("constructed")
        return CountingAdapter()

    return LazyModelHandle(
        factory,
        inference_device="cpu",
        dtype=torch.float32,
    )


def test_cold_then_warm_windows_construct_model_only_for_cold_misses(
    tmp_path,
):
    constructions = []
    cold_handle = _handle(constructions)
    cold = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=cold_handle,
    )
    cold.get(SPECS[0], IMAGES[0:2])
    cold.get(SPECS[1], IMAGES[1:3])

    assert constructions == ["constructed"]
    assert cold_handle.stats.ordinary_forward_count == 2

    def failing_factory():
        raise AssertionError("warm cache must not construct a model")

    warm_handle = LazyModelHandle(
        failing_factory,
        inference_device="cpu",
        dtype=torch.float32,
    )
    warm = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=warm_handle,
    )
    warm.get(SPECS[0], IMAGES[0:2])
    warm.get(SPECS[1], IMAGES[1:3])

    assert warm_handle.stats.model_constructed is False
    assert warm_handle.stats.ordinary_forward_count == 0
    assert warm.store.stats.ordinary_hits == 2


def test_partial_store_forwards_only_missing_window(tmp_path):
    first_handle = _handle([])
    OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=first_handle,
    ).get(SPECS[0], IMAGES[0:2])

    constructions = []
    resumed_handle = _handle(constructions)
    resumed = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=resumed_handle,
    )
    resumed.get(SPECS[0], IMAGES[0:2])
    resumed.get(SPECS[1], IMAGES[1:3])

    assert constructions == ["constructed"]
    assert resumed_handle.stats.ordinary_forward_count == 1
    assert resumed.store.stats.ordinary_hits == 1
    assert resumed.store.stats.ordinary_misses == 1


def test_provider_persists_depth_and_reconstructs_with_reference_intrinsic(
    tmp_path,
):
    provider = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=_handle([]),
    )

    prediction = provider.get(SPECS[0], IMAGES[0:2])
    artifact = provider.store.read_window(SPECS[0])
    sequence = provider.store.read_sequence()

    assert artifact.depth.shape == (2, 4, 4)
    assert torch.equal(
        artifact.depth,
        prediction["local_points"].squeeze(0)[..., -1],
    )
    assert torch.allclose(
        prediction["local_points"].squeeze(0),
        unproject_depth_to_local_points(
            artifact.depth,
            sequence.reference_intrinsic,
        ),
    )
    payload = torch.load(
        provider.store.entry_path / "windows/000000.pt",
        weights_only=False,
    )
    assert set(payload["artifact"]) == {
        "spec",
        "depth",
        "confidence",
        "camera_poses",
    }


def test_provider_attaches_current_images_and_returns_independent_clones(
    tmp_path,
):
    provider = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=_handle([]),
    )
    first = provider.get(SPECS[0], IMAGES[0:2])
    original_value = first["local_points"][0, 0, 0, 0, 0].item()
    first["local_points"].fill_(-99)
    first["images"].fill_(-99)

    replay = provider.get(SPECS[0], IMAGES[0:2])

    assert replay["local_points"][0, 0, 0, 0, 0].item() == original_value
    assert torch.equal(replay["images"].squeeze(0), IMAGES[0:2])


def test_off_mode_uses_same_normalization_but_forwards_every_window(
    tmp_path,
):
    handle = _handle([])
    provider = OrdinaryPredictionProvider(
        store=_store(tmp_path, PredictionCacheMode.OFF),
        model=handle,
    )

    first = provider.get(SPECS[0], IMAGES[0:2])
    second = provider.get(SPECS[1], IMAGES[1:3])

    assert handle.stats.ordinary_forward_count == 2
    assert first["local_points"].shape == (1, 2, 4, 4, 3)
    assert second["local_points"].shape == (1, 2, 4, 4, 3)
    assert not provider.store.entry_path.exists()


def test_readonly_missing_cache_fails_before_model_construction(tmp_path):
    handle = _handle([])
    provider = OrdinaryPredictionProvider(
        store=_store(tmp_path, PredictionCacheMode.READONLY),
        model=handle,
    )

    try:
        provider.get(SPECS[0], IMAGES[0:2])
    except PredictionCacheMissError as error:
        assert "window 000000 [0,2)" in str(error)
    else:
        raise AssertionError("readonly miss did not fail")

    assert handle.stats.model_constructed is False


def test_readonly_corrupt_cache_fails_before_model_construction(tmp_path):
    cold = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=_handle([]),
    )
    cold.get(SPECS[0], IMAGES[0:2])
    path = cold.store.entry_path / "windows/000000.pt"
    payload = torch.load(path, weights_only=False)
    payload["artifact"]["confidence"][0, 0, 0] += 1.0
    torch.save(payload, path)

    handle = _handle([])
    readonly = OrdinaryPredictionProvider(
        store=_store(tmp_path, PredictionCacheMode.READONLY),
        model=handle,
    )
    with pytest.raises(PredictionCacheCorruptError):
        readonly.get(SPECS[0], IMAGES[0:2])

    assert handle.stats.model_constructed is False


def test_lazy_handle_counts_joint_separately():
    handle = _handle([])

    handle.predict(IMAGES[0:2], kind=ModelForwardKind.JOINT)

    assert handle.stats.ordinary_forward_count == 0
    assert handle.stats.joint_forward_count == 1


def test_provider_rejects_later_window_before_reference_intrinsic(tmp_path):
    handle = _handle([])
    provider = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=handle,
    )

    try:
        provider.get(SPECS[1], IMAGES[1:3])
    except ValueError as exc:
        assert "first window" in str(exc)
    else:
        raise AssertionError("out-of-order first access was accepted")

    assert handle.stats.model_constructed is False


def test_provider_rejects_degenerate_generated_intrinsic(
    tmp_path,
    monkeypatch,
):
    from inference_engine.prediction_cache import provider as provider_module

    monkeypatch.setattr(
        provider_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda points: (
            points[..., -1],
            torch.zeros((points.shape[0], 3, 3)),
        ),
    )
    provider = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=_handle([]),
    )

    with pytest.raises(ValueError, match="intrinsic"):
        provider.get(SPECS[0], IMAGES[0:2])


def test_readonly_provider_rejects_cached_degenerate_intrinsic(tmp_path):
    cold = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=_handle([]),
    )
    cold.get(SPECS[0], IMAGES[0:2])
    intrinsic = torch.zeros((3, 3))
    raw_artifact = SimpleNamespace(reference_intrinsic=intrinsic)
    cold.store._atomic_write_json(
        cold.store.sequence_path,
        {
            "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
            "key": cold.store.fingerprint.key,
            "artifact_sha256": _sequence_artifact_digest(
                cold.store.fingerprint.key,
                raw_artifact,
            ),
            "reference_intrinsic": intrinsic.tolist(),
        },
    )
    constructions = []
    readonly = OrdinaryPredictionProvider(
        store=_store(tmp_path, PredictionCacheMode.READONLY),
        model=_handle(constructions),
    )

    with pytest.raises(PredictionCacheCorruptError, match="000000"):
        readonly.get(SPECS[0], IMAGES[0:2])

    assert constructions == []


def test_provider_rejects_nonfinite_reconstructed_points(
    tmp_path,
    monkeypatch,
):
    from inference_engine.prediction_cache import provider as provider_module

    monkeypatch.setattr(
        provider_module,
        "unproject_depth_to_local_points",
        lambda depth, intrinsic: torch.full(
            (*depth.shape, 3),
            float("inf"),
        ),
    )
    provider = OrdinaryPredictionProvider(
        store=_store(tmp_path),
        model=_handle([]),
    )

    with pytest.raises(ValueError, match="reconstructed local points"):
        provider.get(SPECS[0], IMAGES[0:2])
