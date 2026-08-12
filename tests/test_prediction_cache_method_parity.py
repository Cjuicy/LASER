from __future__ import annotations

from dataclasses import replace

from inference_engine.prediction_cache.fingerprint import build_prediction_fingerprint
from inference_engine.prediction_cache.types import build_window_specs
from pipeline.config import ReconstructionMode, SegmentationMethod, load_pipeline_config
from pipeline.manifest import ImageManifest


def test_ordinary_prediction_key_is_independent_of_segmentation_and_mode(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for index in range(12):
        path = image_dir / f"frame-{index:04d}.png"
        path.write_bytes(f"frame-{index}".encode("ascii"))
        paths.append(path)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"deterministic-checkpoint")
    loaded = load_pipeline_config("configs/pipeline/test.yaml")
    base = replace(
        loaded.config,
        input=replace(loaded.config.input, image_dir=str(image_dir)),
        model=replace(loaded.config.model, checkpoint=str(checkpoint)),
        window=replace(loaded.config.window, size=10, overlap=5),
    )
    manifest = ImageManifest(paths=tuple(paths))
    specs = build_window_specs(len(manifest), 10, 5)
    keys = set()

    for segmentation in SegmentationMethod:
        for mode in ReconstructionMode:
            config = replace(
                base,
                segmentation=replace(base.segmentation, method=segmentation),
                reconstruction=replace(base.reconstruction, mode=mode),
            )
            fingerprint = build_prediction_fingerprint(
                model=config.model,
                manifest=manifest,
                image_shape=(12, 3, 4, 4),
                sample_stride=config.input.sample_stride,
                window_size=config.window.size,
                overlap=config.window.overlap,
                specs=specs,
            )
            keys.add(fingerprint.key)

    assert len(keys) == 1


def test_window_settings_define_prediction_key(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for index in range(12):
        path = image_dir / f"frame-{index:04d}.png"
        path.write_bytes(f"frame-{index}".encode("ascii"))
        paths.append(path)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"deterministic-checkpoint")
    config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        (f"model.checkpoint={checkpoint}",),
    ).config
    manifest = ImageManifest(paths=tuple(paths))

    def key(size, overlap):
        specs = build_window_specs(len(manifest), size, overlap)
        return build_prediction_fingerprint(
            model=config.model,
            manifest=manifest,
            image_shape=(12, 3, 4, 4),
            sample_stride=1,
            window_size=size,
            overlap=overlap,
            specs=specs,
        ).key

    assert key(10, 5) != key(8, 4)
