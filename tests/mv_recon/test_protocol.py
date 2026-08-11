from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from mv_recon.protocol import (
    EXPECTED_DATASET_SEQUENCE_COUNTS,
    resolve_evaluation_protocol,
)


ROOT = Path(__file__).resolve().parents[2]


def _root_config(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.2",
    ):
        return compose(
            config_name="eval_mv_recon_dense",
            overrides=[
                "evaluation=mv_recon_laser_paper",
                "device=cpu",
                f"output_dir={tmp_path / 'results'}",
                f"pi3.checkpoint={checkpoint}",
            ],
        )


def test_paper_profile_resolves_all_locked_pipeline_values(tmp_path):
    resolved = resolve_evaluation_protocol(_root_config(tmp_path), ROOT)
    config = resolved.pipeline.config

    assert (config.window.size, config.window.overlap) == (20, 5)
    assert config.segmentation.method.value == "depth"
    assert config.segmentation.confidence_keep_ratio == pytest.approx(0.5)
    assert config.segmentation.depth_merge_threshold == pytest.approx(0.1)
    assert config.segmentation.temporal_iou_threshold == pytest.approx(0.3)
    assert config.segmentation.felzenszwalb.scale == pytest.approx(300)
    assert config.segmentation.felzenszwalb.sigma == pytest.approx(1.1)
    assert config.segmentation.felzenszwalb.min_size == 500
    assert config.anchor_propagation.enabled is True
    assert (
        config.anchor_propagation.correspondence_iou_threshold
        == pytest.approx(0.4)
    )
    assert config.loop.enabled is False
    assert config.loop.method.value == "traditional"
    assert config.loop.registration.confidence_keep_ratio == pytest.approx(0.5)
    assert resolved.protocol.geometry.center_crop_size == 224
    assert EXPECTED_DATASET_SEQUENCE_COUNTS == {
        "7scenes-dense": 18,
        "NRGBD-dense": 9,
    }


@pytest.mark.parametrize(
    "override",
    (
        "window.size=19",
        "segmentation.method=atomic",
        "loop.registration.confidence_keep_ratio=0.3",
        "anchor_propagation.enabled=false",
    ),
)
def test_strict_profile_rejects_pipeline_drift(tmp_path, override):
    config = _root_config(tmp_path)
    config.protocol.pipeline_overrides.append(override)

    with pytest.raises(ValueError, match="laser paper protocol drift"):
        resolve_evaluation_protocol(config, ROOT)


def test_resume_switch_changes_full_hash_but_not_compatibility_identity(
    tmp_path,
):
    initial = _root_config(tmp_path)
    resumed = _root_config(tmp_path)
    resumed.protocol.resume = True

    first = resolve_evaluation_protocol(initial, ROOT)
    second = resolve_evaluation_protocol(resumed, ROOT)

    assert first.sha256 != second.sha256
    assert first.identity_sha256 == second.identity_sha256


def test_protocol_rejects_unknown_fields(tmp_path):
    config = _root_config(tmp_path)
    OmegaConf.set_struct(config.protocol, False)
    config.protocol.unapproved_option = True

    with pytest.raises(ValueError, match="unknown.*unapproved_option"):
        resolve_evaluation_protocol(config, ROOT)


def test_operational_paths_and_device_are_resolved_into_pipeline(tmp_path):
    config = _root_config(tmp_path)
    config.protocol.prediction_cache_root = str(tmp_path / "predictions")
    config.protocol.pipeline_cache_dir = str(tmp_path / "pipeline-cache")

    resolved = resolve_evaluation_protocol(config, ROOT)

    assert resolved.pipeline.config.model.inference_device == "cpu"
    assert resolved.pipeline.config.model.process_device == "cpu"
    assert resolved.pipeline.config.model.dtype == "float32"
    assert resolved.pipeline.config.prediction_cache.root == str(
        (tmp_path / "predictions").resolve()
    )
    assert resolved.pipeline.config.output.cache_dir == str(
        (tmp_path / "pipeline-cache").resolve()
    )


def test_reference_values_are_explicitly_named(tmp_path):
    references = resolve_evaluation_protocol(
        _root_config(tmp_path), ROOT
    ).protocol.paper_reference

    assert references["7scenes-dense"].accuracy_mean_m == pytest.approx(0.013)
    assert references["7scenes-dense"].normal_consistency_median == pytest.approx(
        0.665
    )
    assert references["NRGBD-dense"].completion_mean_m == pytest.approx(0.012)
    assert references["NRGBD-dense"].normal_consistency_median == pytest.approx(
        0.856
    )
