from __future__ import annotations

from experiments.config import CANONICAL_RECONSTRUCTION_MODES
from inference_engine.prediction_cache.fingerprint import build_prediction_fingerprint
from inference_engine.prediction_cache.types import build_window_specs
from pipeline.config import SegmentationMethod, load_pipeline_config
from pipeline.manifest import ImageManifest


def test_all_nine_ate_reconstructions_share_one_ordinary_prediction_key(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for index in range(15):
        path = image_dir / f"frame_{index:04d}.png"
        path.write_bytes(f"frame-{index}".encode("ascii"))
        paths.append(path)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fake-checkpoint")
    manifest = ImageManifest(paths=tuple(paths))
    specs = build_window_specs(15, 10, 5)
    keys = []

    for segmentation in SegmentationMethod:
        for mode in CANONICAL_RECONSTRUCTION_MODES:
            loaded = load_pipeline_config(
                "configs/reconstruction/pi3_laser.yaml",
                (
                    f"model.checkpoint={checkpoint}",
                    f"segmentation.method={segmentation.value}",
                    f"reconstruction.mode={mode.value}",
                    "window.size=10",
                    "window.overlap=5",
                ),
            )
            keys.append(
                build_prediction_fingerprint(
                    model=loaded.config.model,
                    manifest=manifest,
                    image_shape=(15, 3, 4, 4),
                    sample_stride=1,
                    window_size=10,
                    overlap=5,
                    specs=specs,
                ).key
            )

    assert len(keys) == 9
    assert len(set(keys)) == 1
