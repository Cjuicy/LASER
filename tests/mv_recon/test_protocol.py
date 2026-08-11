from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest

from mv_recon import protocol as protocol_module
from mv_recon.protocol import (
    EXPECTED_DATASET_SEQUENCE_COUNTS,
    resolve_evaluation_protocol,
)


ROOT = Path(__file__).resolve().parents[2]


def _root_config(tmp_path: Path):
    return _profile_config(tmp_path, "mv_recon_laser_paper")


def _profile_config(tmp_path: Path, profile: str):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.2",
    ):
        return compose(
            config_name="eval_mv_recon_dense",
            overrides=[
                f"evaluation={profile}",
                "device=cpu",
                f"output_dir={tmp_path / 'results'}",
                f"pi3.checkpoint={checkpoint}",
            ],
        )


def _comparison_config(tmp_path: Path, *, method: str = "depth"):
    config = _root_config(tmp_path)
    OmegaConf.set_struct(config.protocol, False)
    config.protocol.mode = "comparison"
    config.protocol.name = "laser_neuralrgbd_pointmap_comparison"
    config.eval_datasets = ["NRGBD-dense"]
    config.protocol.prediction_cache_mode = (
        "auto" if method == "depth" else "readonly"
    )
    config.protocol.pipeline_overrides = [
        (
            f"segmentation.method={method}"
            if item == "segmentation.method=depth"
            else item
        )
        for item in config.protocol.pipeline_overrides
    ]
    return config


@pytest.mark.parametrize(
    ("profile", "method", "cache_mode"),
    (
        ("mv_recon_laser_nrgbd_depth", "depth", "auto"),
        ("mv_recon_laser_nrgbd_geometry", "geometry", "readonly"),
        ("mv_recon_laser_nrgbd_atomic", "atomic", "readonly"),
    ),
)
def test_nrgbd_profiles_are_locked(
    tmp_path, profile, method, cache_mode
):
    config = _profile_config(tmp_path, profile)
    resolved = resolve_evaluation_protocol(config, ROOT)
    pipeline = resolved.pipeline.config

    assert resolved.protocol.mode == "comparison"
    assert resolved.protocol.max_sequences is None
    assert resolved.datasets == ("NRGBD-dense",)
    assert pipeline.segmentation.method.value == method
    assert pipeline.prediction_cache.mode.value == cache_mode
    assert (pipeline.window.size, pipeline.window.overlap) == (20, 5)
    assert pipeline.segmentation.confidence_keep_ratio == pytest.approx(0.5)
    assert pipeline.segmentation.felzenszwalb.scale == pytest.approx(300)
    assert pipeline.segmentation.felzenszwalb.sigma == pytest.approx(1.1)
    assert pipeline.segmentation.felzenszwalb.min_size == 500
    assert pipeline.segmentation.geometry.normal_method == "cross"
    assert (
        pipeline.segmentation.geometry.normal_threshold_degrees
        == pytest.approx(20.0)
    )
    assert pipeline.segmentation.atomic.split_mode.value == "conservative"
    assert (
        pipeline.segmentation.atomic.split_score_threshold
        == pytest.approx(0.10)
    )
    assert pipeline.anchor_propagation.enabled is True
    assert pipeline.loop.enabled is False


def test_nrgbd_profile_uses_all_fixed_kf10_sequences(tmp_path):
    config = _profile_config(tmp_path, "mv_recon_laser_nrgbd_depth")
    resolved = resolve_evaluation_protocol(config, ROOT)

    plans = protocol_module.build_dataset_plans(resolved, config.data, ROOT)

    assert len(plans) == 1
    assert plans[0].name == "NRGBD-dense"
    assert len(plans[0].sequences) == 9
    assert plans[0].sequence_map_sha256 == (
        "f18f2143f8a373727aa4d7043b779b77639354fda80523cdc2a139164ddc33ba"
    )


def test_paper_profile_declares_paper_mode(tmp_path):
    resolved = resolve_evaluation_protocol(_root_config(tmp_path), ROOT)
    resolved_payload = OmegaConf.create(resolved.resolved_yaml)

    assert resolved.protocol.mode == "paper"
    assert resolved.datasets == ("7scenes-dense", "NRGBD-dense")
    assert resolved_payload.pointmap_assembly == (
        "laser-incremental-global-map-v1"
    )


def test_paper_mode_still_rejects_nrgbd_only(tmp_path):
    config = _root_config(tmp_path)
    config.eval_datasets = ["NRGBD-dense"]

    with pytest.raises(ValueError, match="paper protocol.*datasets"):
        resolve_evaluation_protocol(config, ROOT)


def test_comparison_mode_accepts_nrgbd_only_depth(tmp_path):
    resolved = resolve_evaluation_protocol(
        _comparison_config(tmp_path, method="depth"), ROOT
    )

    assert resolved.protocol.mode == "comparison"
    assert resolved.datasets == ("NRGBD-dense",)
    assert resolved.pipeline.config.segmentation.method.value == "depth"


def test_protocol_rejects_unknown_mode(tmp_path):
    config = _root_config(tmp_path)
    OmegaConf.set_struct(config.protocol, False)
    config.protocol.mode = "benchmark"

    with pytest.raises(ValueError, match="protocol.mode"):
        resolve_evaluation_protocol(config, ROOT)


@pytest.mark.parametrize(
    "datasets",
    (
        [],
        ["NRGBD-dense", "NRGBD-dense"],
        ["NRGBD-dense", "7scenes-dense"],
        ["TUM-dense"],
    ),
)
def test_comparison_mode_rejects_invalid_dataset_selection(
    tmp_path, datasets
):
    config = _comparison_config(tmp_path)
    config.eval_datasets = datasets

    with pytest.raises(ValueError, match="comparison protocol.*datasets"):
        resolve_evaluation_protocol(config, ROOT)


def test_comparison_mode_rejects_wrong_prediction_cache_mode(tmp_path):
    config = _comparison_config(tmp_path, method="geometry")
    config.protocol.prediction_cache_mode = "auto"

    with pytest.raises(ValueError, match="prediction_cache.mode"):
        resolve_evaluation_protocol(config, ROOT)


@pytest.mark.parametrize(
    ("method", "override", "field"),
    (
        (
            "geometry",
            "segmentation.geometry.normal_method=sobel",
            "normal_method",
        ),
        (
            "geometry",
            "segmentation.geometry.normal_threshold_degrees=25.0",
            "normal_threshold_degrees",
        ),
        (
            "atomic",
            "segmentation.atomic.split_mode=normal_only",
            "split_mode",
        ),
        (
            "atomic",
            "segmentation.atomic.split_score_threshold=0.2",
            "split_score_threshold",
        ),
    ),
)
def test_comparison_mode_rejects_method_parameter_drift(
    tmp_path, method, override, field
):
    config = _comparison_config(tmp_path, method=method)
    config.protocol.pipeline_overrides.append(override)

    with pytest.raises(ValueError, match=field):
        resolve_evaluation_protocol(config, ROOT)


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


def test_strict_profile_rejects_changed_paper_reference(tmp_path):
    config = _root_config(tmp_path)
    config.protocol.paper_reference["7scenes-dense"].accuracy_mean_m = 0.5

    with pytest.raises(ValueError, match="paper_reference.*accuracy_mean_m"):
        resolve_evaluation_protocol(config, ROOT)


def test_shipped_maps_have_exact_counts_order_and_kf10(tmp_path):
    config = _root_config(tmp_path)
    resolved = resolve_evaluation_protocol(config, ROOT)

    plans = protocol_module.build_dataset_plans(resolved, config.data, ROOT)

    assert [plan.expected_sequence_count for plan in plans] == [18, 9]
    assert [len(plan.sequences) for plan in plans] == [18, 9]
    assert plans[0].sequences[0].name == "chess/seq-03"
    assert plans[1].sequences[0].name == "breakfast_room"
    assert all(
        right - left == 10
        for plan in plans
        for sequence in plan.sequences
        for left, right in zip(sequence.frame_ids, sequence.frame_ids[1:])
    )
    assert all(len(plan.sequence_map_sha256) == 64 for plan in plans)


def test_max_sequences_limits_each_dataset_after_full_map_validation(tmp_path):
    config = _root_config(tmp_path)
    config.protocol.max_sequences = 1
    resolved = resolve_evaluation_protocol(config, ROOT)

    plans = protocol_module.build_dataset_plans(resolved, config.data, ROOT)

    assert [len(plan.sequences) for plan in plans] == [1, 1]
    assert [plan.expected_sequence_count for plan in plans] == [18, 9]
    assert [plan.sequences[0].name for plan in plans] == [
        "chess/seq-03",
        "breakfast_room",
    ]


def test_paper_profile_rejects_an_alternate_sequence_map_path(tmp_path):
    config = _root_config(tmp_path)
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(
        (
            ROOT
            / "datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json"
        ).read_bytes()
    )
    config.data["7scenes-dense"].seq_id_map = str(alternate)
    resolved = resolve_evaluation_protocol(config, ROOT)

    with pytest.raises(ValueError, match="fixed sequence map path"):
        protocol_module.build_dataset_plans(resolved, config.data, ROOT)


def test_sequence_map_rejects_non_kf10_interval(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"scene": [0, 10, 21]}', encoding="utf-8")

    with pytest.raises(ValueError, match="interval 10"):
        protocol_module.load_sequence_map(path, expected_count=1)


def test_dataset_plan_rejects_requested_frame_outside_sequence(tmp_path):
    path = tmp_path / "map.json"
    path.write_text('{"scene": [0, 10, 20]}', encoding="utf-8")
    sequences = protocol_module.load_sequence_map(path, expected_count=1)
    plan = protocol_module.DatasetPlan(
        name="test",
        sequence_map_path=path,
        sequence_map_sha256="a" * 64,
        expected_sequence_count=1,
        sequences=sequences,
    )

    class Dataset:
        sequence_list = ["scene"]

        def get_seq_framenum(self, sequence_name):
            assert sequence_name == "scene"
            return 20

    with pytest.raises(ValueError, match="requests frame 20.*contains 20"):
        protocol_module.validate_dataset_plan(Dataset(), plan)


def test_dataset_preflight_rejects_missing_requested_indoor_files(tmp_path):
    map_path = tmp_path / "map.json"
    map_path.write_text('{"scene": [0, 10]}', encoding="utf-8")
    plan = protocol_module.DatasetPlan(
        name="7scenes-dense",
        sequence_map_path=map_path,
        sequence_map_sha256="a" * 64,
        expected_sequence_count=1,
        sequences=protocol_module.load_sequence_map(
            map_path,
            expected_count=1,
        ),
    )

    class Dataset:
        sequence_list = ["scene"]
        SEVENSCENES_DIR = str(tmp_path / "7scenes")
        load_img_size = 518

        def get_seq_framenum(self, sequence_name):
            return 20

    with pytest.raises(FileNotFoundError, match="requested frame file"):
        protocol_module.validate_dataset_plan(Dataset(), plan)


def test_dataset_preflight_rejects_crop_larger_than_resized_height(tmp_path):
    map_path = tmp_path / "map.json"
    map_path.write_text('{"scene": [0, 10]}', encoding="utf-8")
    plan = protocol_module.DatasetPlan(
        name="test",
        sequence_map_path=map_path,
        sequence_map_sha256="a" * 64,
        expected_sequence_count=1,
        sequences=protocol_module.load_sequence_map(
            map_path,
            expected_count=1,
        ),
    )

    class Dataset:
        sequence_list = ["scene"]
        load_img_size = 200

        def get_seq_framenum(self, sequence_name):
            return 20

    with pytest.raises(ValueError, match="224.*center crop"):
        protocol_module.validate_dataset_plan(Dataset(), plan)


def test_manifest_digest_uses_image_contents(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    initial = protocol_module.manifest_digest_for_paths([first, second])
    second.write_bytes(b"changed")
    changed = protocol_module.manifest_digest_for_paths([first, second])

    assert len(initial) == 64
    assert initial != changed


def test_ground_truth_digest_uses_point_maps_and_valid_mask():
    points = np.zeros((1, 2, 2, 3), dtype=np.float32)
    mask = np.ones((1, 2, 2), dtype=bool)

    initial = protocol_module.digest_ground_truth(points, mask)
    changed_points = points.copy()
    changed_points[0, 0, 0, 2] = 1.0
    changed_mask = mask.copy()
    changed_mask[0, 0, 0] = False

    assert len(initial) == 64
    assert protocol_module.digest_ground_truth(changed_points, mask) != initial
    assert protocol_module.digest_ground_truth(points, changed_mask) != initial
