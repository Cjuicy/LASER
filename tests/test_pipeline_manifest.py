from pathlib import Path

import pytest

from pipeline.config import load_pipeline_config
from pipeline.manifest import discover_image_manifest
from pipeline.preflight import validate_preflight


def _create_images(directory: Path, count: int = 6) -> None:
    directory.mkdir()
    for index in range(count):
        (directory / f"{index}.png").write_bytes(b"x")


def _load_valid_config(tmp_path: Path, image_dir: Path, *overrides: str):
    model_checkpoint = tmp_path / "model.safetensors"
    salad_checkpoint = tmp_path / "salad.ckpt"
    dino_checkpoint = tmp_path / "dino.pth"
    for checkpoint in (
        model_checkpoint,
        salad_checkpoint,
        dino_checkpoint,
    ):
        checkpoint.write_bytes(b"x")
    return load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            f"input.image_dir={image_dir}",
            f"model.checkpoint={model_checkpoint}",
            f"loop.detection.salad_checkpoint={salad_checkpoint}",
            f"loop.detection.dino_checkpoint={dino_checkpoint}",
            *overrides,
        ),
    ).config


def test_manifest_filters_natural_sorts_then_samples(tmp_path):
    for name in ("frame10.JPG", "frame2.png", "frame1.jpeg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    manifest = discover_image_manifest(tmp_path, sample_stride=2)
    assert [path.name for path in manifest.paths] == [
        "frame1.jpeg",
        "frame10.JPG",
    ]
    assert all(path.is_absolute() for path in manifest.paths)


def test_manifest_rejects_rescanning_mutation(tmp_path):
    (tmp_path / "1.png").write_bytes(b"x")
    manifest = discover_image_manifest(tmp_path, 1)
    (tmp_path / "2.png").write_bytes(b"x")
    assert [path.name for path in manifest.paths] == ["1.png"]


@pytest.mark.parametrize("stride", (0, -1))
def test_manifest_requires_positive_stride(tmp_path, stride):
    with pytest.raises(ValueError, match="sample_stride"):
        discover_image_manifest(tmp_path, stride)


def test_manifest_rejects_nonexistent_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="image_dir"):
        discover_image_manifest(tmp_path / "missing", 1)


def test_manifest_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="image"):
        discover_image_manifest(tmp_path, 1)


def test_preflight_rejects_missing_checkpoint_before_model_load(tmp_path):
    image_dir = tmp_path / "images"
    _create_images(image_dir)
    loaded = load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            f"input.image_dir={image_dir}",
            f"model.checkpoint={tmp_path / 'missing.safetensors'}",
        ),
    )
    manifest = discover_image_manifest(image_dir, 1)
    with pytest.raises(FileNotFoundError, match="model.checkpoint"):
        validate_preflight(loaded.config, manifest, cuda_available=True)


def test_preflight_requires_more_images_than_overlap(tmp_path):
    image_dir = tmp_path / "images"
    _create_images(image_dir, count=5)
    config = _load_valid_config(tmp_path, image_dir)
    manifest = discover_image_manifest(image_dir, 1)
    with pytest.raises(ValueError, match="window.overlap"):
        validate_preflight(config, manifest, cuda_available=True)


@pytest.mark.parametrize(
    "field,filename",
    (
        ("loop.detection.salad_checkpoint", "missing-salad.ckpt"),
        ("loop.detection.dino_checkpoint", "missing-dino.pth"),
    ),
)
def test_preflight_requires_loop_weights_when_enabled(
    tmp_path,
    field,
    filename,
):
    image_dir = tmp_path / "images"
    _create_images(image_dir)
    config = _load_valid_config(
        tmp_path,
        image_dir,
        f"{field}={tmp_path / filename}",
    )
    manifest = discover_image_manifest(image_dir, 1)
    with pytest.raises(FileNotFoundError, match=field):
        validate_preflight(config, manifest, cuda_available=True)


def test_preflight_skips_loop_weights_when_loop_is_disabled(tmp_path):
    image_dir = tmp_path / "images"
    _create_images(image_dir)
    config = _load_valid_config(
        tmp_path,
        image_dir,
        "loop.enabled=false",
        f"loop.detection.salad_checkpoint={tmp_path / 'missing-salad.ckpt'}",
        f"loop.detection.dino_checkpoint={tmp_path / 'missing-dino.pth'}",
        "model.inference_device=cpu",
    )
    manifest = discover_image_manifest(image_dir, 1)
    validate_preflight(config, manifest, cuda_available=False)


def test_preflight_rejects_cuda_when_unavailable(tmp_path):
    image_dir = tmp_path / "images"
    _create_images(image_dir)
    config = _load_valid_config(tmp_path, image_dir)
    manifest = discover_image_manifest(image_dir, 1)
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_preflight(config, manifest, cuda_available=False)


@pytest.mark.parametrize(
    "override,match",
    (
        ("model.dtype=integer", "dtype"),
        ("model.inference_device=tpu", "inference_device"),
        ("model.process_device=tpu", "process_device"),
    ),
)
def test_preflight_rejects_unsupported_model_runtime(
    tmp_path,
    override,
    match,
):
    image_dir = tmp_path / "images"
    _create_images(image_dir)
    config = _load_valid_config(tmp_path, image_dir, override)
    manifest = discover_image_manifest(image_dir, 1)
    with pytest.raises(ValueError, match=match):
        validate_preflight(config, manifest, cuda_available=True)
