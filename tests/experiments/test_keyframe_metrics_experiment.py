from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path

import json
import numpy as np
import pytest

import experiments.config as experiment_config
import experiments.matrix as experiment_matrix
import experiments.pointcloud as pointcloud_experiment
import experiments.runner as experiment_runner
from evaluation.pointcloud import PrimaryMetrics
from experiments.ate import evaluate_ate_inputs
from experiments.config import EvaluationKind, load_experiment_config
from experiments.evaluation_bundle import (
    EvaluationBundleRunner,
    EvaluatorStatus,
)
from experiments.pointcloud import evaluate_pointcloud_inputs
from pipeline.artifacts import ReconstructionArtifact
from pipeline.artifacts import write_reconstruction_artifact
from pipeline.config import ReconstructionMode, SegmentationMethod
from tests.test_pipeline_runner import _artifact


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


def _capability_runner_setup(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(3):
        (image_dir / f"frame-{index}.png").write_bytes(b"image")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    ate_config = tmp_path / "ate.yaml"
    ate_config.write_text(
        "version: 1\nalignment: sim3\nrpe_delta_frames: 1\n",
        encoding="utf-8",
    )
    ate_truth = tmp_path / "trajectory.txt"
    ate_truth.write_text(
        "0 0 0 0 0 0 0 1\n"
        "1 1 0 0 0 0 0 1\n"
        "2 0 1 0 0 0 0 1\n",
        encoding="utf-8",
    )
    pointcloud_config = tmp_path / "pointcloud.yaml"
    pointcloud_config.write_text(
        "version: 1\n"
        "center_crop_size: 2\n"
        "alignment: umeyama_sim3_then_icp\n"
        "icp_type: point_to_point\n"
        "icp_threshold_m: 0.1\n"
        "normal_estimation: open3d_default\n"
        "fscore_thresholds_m: [0.01, 0.02, 0.05]\n",
        encoding="utf-8",
    )
    pointcloud_truth = tmp_path / "pointcloud.npz"
    np.savez(
        pointcloud_truth,
        point_maps=np.zeros((3, 2, 2, 3), dtype=np.float32),
        valid_mask=np.ones((3, 2, 2), dtype=bool),
    )
    baseline = _capability_config(
        EvaluationKind.ATE,
        EvaluationKind.POINTCLOUD,
    )
    experiment = experiment_config.CapabilityExperimentConfig(
        **{
            **baseline.__dict__,
            "output_root": str(tmp_path / "output"),
            "evaluator_inputs": {
                EvaluationKind.ATE: experiment_config.EvaluationInputConfig(
                    EvaluationKind.ATE,
                    str(ate_config),
                    str(ate_truth),
                    "tum",
                ),
                EvaluationKind.POINTCLOUD: (
                    experiment_config.EvaluationInputConfig(
                        EvaluationKind.POINTCLOUD,
                        str(pointcloud_config),
                        str(pointcloud_truth),
                    )
                ),
            },
        }
    )
    overrides = (
        f"input.image_dir={image_dir}",
        f"model.checkpoint={checkpoint}",
        "model.process_device=cpu",
        "model.inference_device=cpu",
    )
    return experiment, overrides


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


def test_checked_in_keyframe_metrics_config_expands_to_fifteen_variants():
    config = load_experiment_config(
        "configs/experiments/keyframe_metrics_matrix.yaml"
    )

    assert isinstance(config, experiment_config.CapabilityExperimentConfig)
    assert len(config.entries) == 15


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


def test_capability_runner_reconstructs_once_for_two_evaluators(
    tmp_path,
    monkeypatch,
):
    experiment, overrides = _capability_runner_setup(tmp_path)
    entry = experiment_matrix.CapabilityMatrixEntry(
        SegmentationMethod.DEPTH,
        ReconstructionMode.NO_LOOP,
        False,
        (EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
    )
    monkeypatch.setattr(
        experiment_matrix,
        "build_capability_matrix",
        lambda config: (entry,),
    )
    reconstruction_targets = []

    class FakePipelineRunner:
        def __init__(self, loaded, *, artifact_output_dir):
            self.loaded = loaded
            self.target = Path(artifact_output_dir)

        def run(self):
            reconstruction_targets.append(self.target)
            return write_reconstruction_artifact(
                _artifact(self.loaded.config.reconstruction.mode),
                self.target,
                resolved_yaml=self.loaded.resolved_yaml,
                config_sha256=self.loaded.sha256,
                checkpoint_sha256="b" * 64,
                git_commit="c" * 40,
            )

    evaluator_artifacts = []

    def evaluator(artifact_dir, **values):
        evaluator_artifacts.append(Path(artifact_dir))
        output = Path(values["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        result = output / "metrics.json"
        result.write_text("{}\n", encoding="utf-8")
        return result

    records = experiment_runner.run_capability_matrix(
        experiment,
        overrides=overrides,
        runner_factory=FakePipelineRunner,
        bundle_runner=EvaluationBundleRunner(
            ate_evaluator=evaluator,
            pointcloud_evaluator=evaluator,
        ),
    )

    assert len(reconstruction_targets) == 1
    assert evaluator_artifacts == [
        reconstruction_targets[0],
        reconstruction_targets[0],
    ]
    assert records[0].artifact_dir == reconstruction_targets[0]
    assert [item.status for item in records[0].evaluations] == [
        EvaluatorStatus.PASSED,
        EvaluatorStatus.PASSED,
    ]


def test_capability_runner_blocks_failed_reconstruction_and_keeps_going(
    tmp_path,
    monkeypatch,
):
    experiment, overrides = _capability_runner_setup(tmp_path)
    entries = (
        experiment_matrix.CapabilityMatrixEntry(
            SegmentationMethod.DEPTH,
            ReconstructionMode.NO_LOOP,
            False,
            (EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
        ),
        experiment_matrix.CapabilityMatrixEntry(
            SegmentationMethod.DEPTH,
            ReconstructionMode.NO_LOOP,
            True,
            (EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
        ),
    )
    monkeypatch.setattr(
        experiment_matrix,
        "build_capability_matrix",
        lambda config: entries,
    )
    reconstruction_calls = []

    class SometimesFailingPipelineRunner:
        def __init__(self, loaded, *, artifact_output_dir):
            self.loaded = loaded
            self.target = Path(artifact_output_dir)

        def run(self):
            reconstruction_calls.append(self.target)
            if len(reconstruction_calls) == 1:
                raise RuntimeError("synthetic reconstruction failure")
            return write_reconstruction_artifact(
                _artifact(self.loaded.config.reconstruction.mode),
                self.target,
                resolved_yaml=self.loaded.resolved_yaml,
                config_sha256=self.loaded.sha256,
                checkpoint_sha256="b" * 64,
                git_commit="c" * 40,
            )

    def evaluator(artifact_dir, **values):
        del artifact_dir
        output = Path(values["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        result = output / "metrics.json"
        result.write_text("{}\n", encoding="utf-8")
        return result

    records = experiment_runner.run_capability_matrix(
        experiment,
        overrides=overrides,
        runner_factory=SometimesFailingPipelineRunner,
        bundle_runner=EvaluationBundleRunner(
            ate_evaluator=evaluator,
            pointcloud_evaluator=evaluator,
        ),
    )

    assert len(reconstruction_calls) == 2
    assert records[0].reconstruction_identity is None
    assert records[0].artifact_dir is None
    assert [item.status for item in records[0].evaluations] == [
        EvaluatorStatus.BLOCKED,
        EvaluatorStatus.BLOCKED,
    ]
    assert records[0].succeeded is False
    assert [item.status for item in records[1].evaluations] == [
        EvaluatorStatus.PASSED,
        EvaluatorStatus.PASSED,
    ]
    assert records[1].succeeded is True

    summary_path = Path(experiment.output_root) / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["overall_success"] is False
    assert [item["entry"] for item in summary["entries"]] == [
        entries[0].name,
        entries[1].name,
    ]
    assert [
        evaluation["status"]
        for evaluation in summary["entries"][0]["evaluations"]
    ] == ["blocked", "blocked"]


def test_run_matrix_dispatches_version_two_to_capability_runner(
    tmp_path,
    monkeypatch,
):
    experiment, _ = _capability_runner_setup(tmp_path)
    observed = []

    def run_capability(config, **values):
        observed.append((config, values))
        return ("capability-record",)

    monkeypatch.setattr(
        experiment_runner,
        "run_capability_matrix",
        run_capability,
    )
    bundle = EvaluationBundleRunner()

    result = experiment_runner.run_matrix(
        experiment,
        overrides=("model.process_device=cpu",),
        dry_run=True,
        runner_factory=object,
        bundle_runner=bundle,
    )

    assert result == ("capability-record",)
    assert observed == [
        (
            experiment,
            {
                "overrides": ("model.process_device=cpu",),
                "dry_run": True,
                "runner_factory": object,
                "bundle_runner": bundle,
            },
        )
    ]


def test_capability_runner_real_input_functions_resume_passed_outputs(
    tmp_path,
    monkeypatch,
):
    experiment, overrides = _capability_runner_setup(tmp_path)
    entry = experiment_matrix.CapabilityMatrixEntry(
        SegmentationMethod.DEPTH,
        ReconstructionMode.NO_LOOP,
        False,
        (EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
    )
    monkeypatch.setattr(
        experiment_matrix,
        "build_capability_matrix",
        lambda config: (entry,),
    )

    @dataclass(frozen=True)
    class PointcloudDiagnostics:
        frame_count: int

    def evaluate_point_maps(estimate, truth, config):
        assert estimate.global_points.shape == (3, 2, 2, 3)
        assert truth.point_maps.shape == (3, 2, 2, 3)
        assert config.center_crop_size == 2
        return type(
            "Evaluation",
            (),
            {
                "primary": PrimaryMetrics(0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
                "diagnostics": PointcloudDiagnostics(3),
            },
        )()

    monkeypatch.setattr(
        pointcloud_experiment,
        "evaluate_point_maps",
        evaluate_point_maps,
    )
    reconstruction_calls = []

    class CpuArtifactPipelineRunner:
        def __init__(self, loaded, *, artifact_output_dir):
            self.loaded = loaded
            self.target = Path(artifact_output_dir)

        def run(self):
            reconstruction_calls.append(self.target)
            baseline = _artifact(ReconstructionMode.NO_LOOP)
            poses = baseline.camera_poses.clone()
            poses[1, 0, 3] = 1.0
            poses[2, 1, 3] = 1.0
            artifact = ReconstructionArtifact(
                **{
                    **baseline.__dict__,
                    "camera_poses": poses,
                }
            )
            return write_reconstruction_artifact(
                artifact,
                self.target,
                resolved_yaml=self.loaded.resolved_yaml,
                config_sha256=self.loaded.sha256,
                checkpoint_sha256="b" * 64,
                git_commit="c" * 40,
            )

    evaluator_calls = {EvaluationKind.ATE: 0, EvaluationKind.POINTCLOUD: 0}

    def counted_ate(*args, **values):
        evaluator_calls[EvaluationKind.ATE] += 1
        return evaluate_ate_inputs(*args, **values)

    def counted_pointcloud(*args, **values):
        evaluator_calls[EvaluationKind.POINTCLOUD] += 1
        return evaluate_pointcloud_inputs(*args, **values)

    bundle = EvaluationBundleRunner(
        ate_evaluator=counted_ate,
        pointcloud_evaluator=counted_pointcloud,
    )
    first = experiment_runner.run_capability_matrix(
        experiment,
        overrides=overrides,
        runner_factory=CpuArtifactPipelineRunner,
        bundle_runner=bundle,
    )

    assert [item.status for item in first[0].evaluations] == [
        EvaluatorStatus.PASSED,
        EvaluatorStatus.PASSED,
    ]
    assert evaluator_calls == {EvaluationKind.ATE: 1, EvaluationKind.POINTCLOUD: 1}
    summary_path = Path(experiment.output_root) / "evaluation_summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8"))[
        "overall_success"
    ] is True

    second = experiment_runner.run_capability_matrix(
        experiment,
        overrides=overrides,
        runner_factory=CpuArtifactPipelineRunner,
        bundle_runner=bundle,
    )

    assert [item.status for item in second[0].evaluations] == [
        EvaluatorStatus.PASSED,
        EvaluatorStatus.PASSED,
    ]
    assert len(reconstruction_calls) == 1
    assert evaluator_calls == {EvaluationKind.ATE: 1, EvaluationKind.POINTCLOUD: 1}
