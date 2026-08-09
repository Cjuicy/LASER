from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from inference_engine.prediction_cache import fingerprint as fingerprint_module
from inference_engine.prediction_cache.fingerprint import (
    build_prediction_fingerprint,
)
from inference_engine.prediction_cache.types import (
    WindowSpec,
    build_window_specs,
)
from pipeline.config import ModelName, load_pipeline_config
from pipeline.manifest import ImageManifest


def _inputs(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    first = tmp_path / "000.png"
    second = tmp_path / "001.png"
    first.write_bytes(b"image-a")
    second.write_bytes(b"image-b")
    loaded = load_pipeline_config("configs/pipeline/test.yaml")
    model = replace(
        loaded.config.model,
        checkpoint=str(checkpoint),
        inference_device="cuda:0",
        process_device="cpu",
        dtype="float32",
    )
    manifest = ImageManifest((first.resolve(), second.resolve()))
    return {
        "model": model,
        "manifest": manifest,
        "image_shape": (2, 3, 14, 28),
        "sample_stride": 1,
        "window_size": 2,
        "overlap": 1,
        "specs": build_window_specs(2, 2, 1),
    }


def test_semantically_identical_inputs_have_stable_key(tmp_path):
    inputs = _inputs(tmp_path)

    first = build_prediction_fingerprint(**inputs)
    second = build_prediction_fingerprint(**inputs)

    assert first == second
    assert len(first.key) == 64
    assert first.canonical_payload["model_name"] == "pi3"
    assert first.canonical_payload["window_size"] == 2
    assert first.canonical_payload["overlap"] == 1


def test_mtime_and_device_changes_do_not_change_key(tmp_path):
    inputs = _inputs(tmp_path)
    initial = build_prediction_fingerprint(**inputs)
    os.utime(inputs["manifest"].paths[0], (1_000_000, 1_000_000))
    moved_devices = replace(
        inputs["model"],
        inference_device="cuda:7",
        process_device="mps",
    )

    changed = build_prediction_fingerprint(
        **{**inputs, "model": moved_devices}
    )

    assert changed.key == initial.key


def test_checkpoint_content_changes_key(tmp_path):
    inputs = _inputs(tmp_path)
    initial = build_prediction_fingerprint(**inputs)
    Path(inputs["model"].checkpoint).write_bytes(b"checkpoint-b")

    assert build_prediction_fingerprint(**inputs).key != initial.key


def test_image_content_and_order_change_key(tmp_path):
    inputs = _inputs(tmp_path)
    initial = build_prediction_fingerprint(**inputs)
    reversed_manifest = ImageManifest(tuple(reversed(inputs["manifest"].paths)))

    reversed_key = build_prediction_fingerprint(
        **{**inputs, "manifest": reversed_manifest}
    ).key
    inputs["manifest"].paths[0].write_bytes(b"image-c")
    content_key = build_prediction_fingerprint(**inputs).key

    assert reversed_key != initial.key
    assert content_key != initial.key


def test_dtype_preprocessing_and_window_schedule_change_key(tmp_path):
    inputs = _inputs(tmp_path)
    initial = build_prediction_fingerprint(**inputs).key
    variations = (
        {**inputs, "model": replace(inputs["model"], dtype="float16")},
        {**inputs, "image_shape": (2, 3, 28, 28)},
        {**inputs, "sample_stride": 2},
        {
            **inputs,
            "window_size": 3,
            "specs": build_window_specs(2, 3, 1),
        },
    )

    assert all(
        build_prediction_fingerprint(**variation).key != initial
        for variation in variations
    )


def test_schema_and_adapter_contract_versions_change_key(
    tmp_path,
    monkeypatch,
):
    inputs = _inputs(tmp_path)
    initial = build_prediction_fingerprint(**inputs).key

    monkeypatch.setattr(
        fingerprint_module,
        "MODEL_ADAPTER_CONTRACT_VERSION",
        99,
    )
    adapter_key = build_prediction_fingerprint(**inputs).key
    monkeypatch.setattr(
        fingerprint_module,
        "PREDICTION_CACHE_SCHEMA_VERSION",
        99,
    )
    schema_key = build_prediction_fingerprint(**inputs).key

    assert adapter_key != initial
    assert schema_key != adapter_key


def test_runtime_source_content_changes_key(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    source = tmp_path / "runtime.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        fingerprint_module,
        "runtime_source_paths",
        lambda model_name: (source,),
    )
    initial = build_prediction_fingerprint(**inputs).key
    source.write_text("VERSION = 2\n", encoding="utf-8")

    assert build_prediction_fingerprint(**inputs).key != initial


def test_preprocessing_implementation_is_a_runtime_source():
    paths = fingerprint_module.runtime_source_paths(ModelName.PI3)

    assert (
        fingerprint_module._REPOSITORY_ROOT / "utils/load_fn.py"
    ) in paths


def test_non_inference_pipeline_options_cannot_pollute_payload(tmp_path):
    inputs = _inputs(tmp_path)
    fingerprint = build_prediction_fingerprint(**inputs)

    forbidden = {
        "segmentation",
        "atomic",
        "anchor",
        "loop",
        "salad",
        "chunk_size",
        "optimizer",
        "process_device",
        "inference_device",
        "scene_name",
        "result_dir",
        "cache_dir",
    }
    payload_text = repr(fingerprint.canonical_payload).casefold()
    assert all(name not in payload_text for name in forbidden)


def test_invalid_model_name_is_rejected_before_hashing(tmp_path):
    inputs = _inputs(tmp_path)
    invalid_model = replace(inputs["model"], name="pi3x")

    with pytest.raises(ValueError, match="model name"):
        build_prediction_fingerprint(
            **{**inputs, "model": invalid_model}
        )


@pytest.mark.parametrize(
    "shape",
    (
        (1, 3, 14, 28),
        (2, 1, 14, 28),
        (2, 3, 0, 28),
    ),
)
def test_invalid_preprocessed_shape_is_rejected(tmp_path, shape):
    inputs = _inputs(tmp_path)

    with pytest.raises(ValueError, match="shape"):
        build_prediction_fingerprint(
            **{**inputs, "image_shape": shape}
        )


def test_noncanonical_window_schedule_is_rejected(tmp_path):
    inputs = _inputs(tmp_path)
    noncanonical = (
        WindowSpec(0, 0, 2),
        WindowSpec(1, 0, 2),
    )

    with pytest.raises(ValueError, match="canonical sliding schedule"):
        build_prediction_fingerprint(
            **{**inputs, "specs": noncanonical}
        )
