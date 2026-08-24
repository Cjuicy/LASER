from __future__ import annotations

from types import MappingProxyType

import pytest

import experiments.config as experiment_config
import experiments.matrix as experiment_matrix
from experiments.config import EvaluationKind, load_experiment_config
from pipeline.config import ReconstructionMode


DUAL_CAPABILITY_YAML = """\
version: 2
reconstruction_config: configs/reconstruction/pi3_laser.yaml
output_root: outputs/experiments/keyframe-metrics
dataset:
  name: fixture
  sequence: sequence-0
  trajectory_ground_truth:
    path: data/groundtruth.txt
    format: tum
  pointcloud_ground_truth:
    path: data/groundtruth_pointmaps.npz
evaluation:
  trajectory_config: configs/evaluation/ate.yaml
  pointcloud_config: configs/evaluation/pointcloud.yaml
matrix:
  segmentation: [depth, geometry, atomic]
  refinement: [false, true]
"""


def _load_yaml(tmp_path, text):
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return load_experiment_config(path)


def _capability_config(*kinds):
    inputs = {}
    for kind in kinds:
        inputs[kind] = experiment_config.EvaluationInputConfig(
            kind=kind,
            config_path=f"configs/{kind.value}.yaml",
            ground_truth_path=f"data/{kind.value}.gt",
            ground_truth_format="tum" if kind is EvaluationKind.ATE else None,
        )
    return experiment_config.CapabilityExperimentConfig(
        version=2,
        reconstruction_config="configs/reconstruction/pi3_laser.yaml",
        output_root="outputs/keyframe-metrics",
        dataset_name="fixture",
        sequence="sequence-0",
        evaluator_inputs=inputs,
        segmentation_methods=("depth", "geometry", "atomic"),
        refinement_states=(False, True),
    )


def test_version_two_dual_capability_config_normalizes_inputs(tmp_path):
    config = _load_yaml(tmp_path, DUAL_CAPABILITY_YAML)

    assert isinstance(config, experiment_config.CapabilityExperimentConfig)
    assert config.version == 2
    assert config.dataset_name == "fixture"
    assert config.sequence == "sequence-0"
    assert isinstance(config.evaluator_inputs, MappingProxyType)
    assert tuple(config.evaluator_inputs) == (
        EvaluationKind.ATE,
        EvaluationKind.POINTCLOUD,
    )
    trajectory = config.evaluator_inputs[EvaluationKind.ATE]
    assert trajectory.config_path == "configs/evaluation/ate.yaml"
    assert trajectory.ground_truth_path == "data/groundtruth.txt"
    assert trajectory.ground_truth_format == "tum"
    pointcloud = config.evaluator_inputs[EvaluationKind.POINTCLOUD]
    assert pointcloud.config_path == "configs/evaluation/pointcloud.yaml"
    assert pointcloud.ground_truth_path == "data/groundtruth_pointmaps.npz"
    assert pointcloud.ground_truth_format is None
    assert config.segmentation_methods == ("depth", "geometry", "atomic")
    assert config.refinement_states == (False, True)


def test_version_two_requires_at_least_one_declared_capability(tmp_path):
    text = DUAL_CAPABILITY_YAML.replace(
        "  trajectory_ground_truth:\n"
        "    path: data/groundtruth.txt\n"
        "    format: tum\n"
        "  pointcloud_ground_truth:\n"
        "    path: data/groundtruth_pointmaps.npz\n",
        "",
    ).replace(
        "  trajectory_config: configs/evaluation/ate.yaml\n"
        "  pointcloud_config: configs/evaluation/pointcloud.yaml\n",
        "",
    )

    with pytest.raises(ValueError, match="at least one.*ground truth"):
        _load_yaml(tmp_path, text)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "    format: tum\n",
            "",
            "trajectory.*format",
        ),
        (
            "  trajectory_config: configs/evaluation/ate.yaml\n",
            "",
            "trajectory.*config",
        ),
        (
            "  pointcloud_config: configs/evaluation/pointcloud.yaml\n",
            "",
            "pointcloud.*config",
        ),
        (
            "  refinement: [false, true]\n",
            "  refinement: [false, true]\n  unexpected: true\n",
            "unknown experiment matrix field",
        ),
    ],
    ids=[
        "trajectory-format",
        "trajectory-config",
        "pointcloud-config",
        "unknown-matrix-field",
    ],
)
def test_version_two_rejects_incomplete_or_unknown_fields(
    tmp_path,
    old,
    new,
    message,
):
    with pytest.raises(ValueError, match=message):
        _load_yaml(tmp_path, DUAL_CAPABILITY_YAML.replace(old, new))


@pytest.mark.parametrize(
    ("kinds", "expected_count"),
    [
        ((EvaluationKind.ATE, EvaluationKind.POINTCLOUD), 15),
        ((EvaluationKind.ATE,), 15),
        ((EvaluationKind.POINTCLOUD,), 6),
    ],
    ids=["dual", "ate-only", "pointcloud-only"],
)
def test_capability_matrix_has_exact_scene_supported_variants(
    kinds,
    expected_count,
):
    entries = experiment_matrix.build_capability_matrix(
        _capability_config(*kinds)
    )

    assert len(entries) == expected_count
    assert len({entry.name for entry in entries}) == expected_count
    traditional = [
        entry
        for entry in entries
        if entry.reconstruction_mode is ReconstructionMode.TRADITIONAL
    ]
    if EvaluationKind.ATE in kinds:
        assert len(traditional) == 3
        assert all(not entry.window_reference_enabled for entry in traditional)
    else:
        assert traditional == []


def test_dual_capability_matrix_runs_both_evaluators_only_for_no_loop():
    entries = experiment_matrix.build_capability_matrix(
        _capability_config(EvaluationKind.ATE, EvaluationKind.POINTCLOUD)
    )

    for entry in entries:
        if entry.reconstruction_mode is ReconstructionMode.NO_LOOP:
            assert entry.evaluator_kinds == (
                EvaluationKind.ATE,
                EvaluationKind.POINTCLOUD,
            )
        else:
            assert entry.evaluator_kinds == (EvaluationKind.ATE,)


def test_pointcloud_only_matrix_contains_two_no_loop_states_per_segmentation():
    entries = experiment_matrix.build_capability_matrix(
        _capability_config(EvaluationKind.POINTCLOUD)
    )

    assert [
        (
            entry.segmentation_method.value,
            entry.reconstruction_mode.value,
            entry.window_reference_enabled,
            entry.evaluator_kinds,
        )
        for entry in entries
    ] == [
        (segmentation, "no_loop", refinement, (EvaluationKind.POINTCLOUD,))
        for segmentation in ("depth", "geometry", "atomic")
        for refinement in (False, True)
    ]
