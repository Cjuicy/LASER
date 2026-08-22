import hashlib
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.window_reference_campaign.config import (
    CampaignOverrides,
    EvaluationKind,
    FailurePolicy,
    load_campaign_config,
)
from experiments.window_reference_campaign.matrix import (
    RunIdentitySeed,
    build_plan,
    complete_identity,
    expand_matrix,
)
from experiments.window_reference_campaign.preflight import (
    CudaState,
    DiskUsage,
    GitState,
    PreflightDependencies,
    preflight_campaign,
)
from experiments.window_reference_campaign.results import CacheStats
from experiments.window_reference_campaign.runner import (
    EvaluationOutput,
    PipelineExecution,
    RunnerDependencies,
    _source_metadata,
    run_campaign,
)
from experiments.window_reference_campaign.scenes import resolve_scene
from experiments.window_reference_campaign.staging import (
    identity_manifest_sha256,
    stage_scene,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/window_reference_campaign.yaml"


def test_default_matrix_is_the_approved_six_in_order():
    assert [item.run_id for item in expand_matrix()] == [
        "depth__wr-off",
        "depth__wr-on",
        "geometry__wr-off",
        "geometry__wr-on",
        "atomic__wr-off",
        "atomic__wr-on",
    ]


def test_typed_config_locks_primary_protocol_and_presets():
    loaded = load_campaign_config(
        CONFIG,
        CampaignOverrides(preset="pointcloud-small"),
    )
    config = loaded.config
    assert config.matrix.reconstruction_mode.value == "no_loop"
    assert (config.matrix.window_size, config.matrix.overlap) == (75, 30)
    assert config.matrix.atomic_split_mode.value == "conservative"
    assert config.runtime.jobs == 1
    assert [item.scene_id for item in config.selected_scenes] == [
        "seven-chess-seq-03",
        "nrgbd-thin-geometry",
        "nrgbd-complete-kitchen",
    ]
    assert [item.stop for item in load_campaign_config(
        CONFIG, CampaignOverrides(preset="kitti-smoke")
    ).config.selected_scenes] == [80]


@pytest.mark.parametrize("raw", ["off", "on", "yes", "no"])
def test_refinement_axis_rejects_yaml_like_strings(tmp_path, raw):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "refinement: [false, true]",
        f"refinement: [false, {raw!r}]",
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="matrix.refinement.*boolean"):
        load_campaign_config(path, CampaignOverrides(preset="synthetic-smoke"))


def test_unknown_campaign_key_is_rejected(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8") + "surprise: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown campaign field: surprise"):
        load_campaign_config(path, CampaignOverrides(preset="synthetic-smoke"))


def test_runtime_jobs_must_remain_single_scene_serial_protocol(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8").replace("  jobs: 1\n", "  jobs: 2\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime.jobs must be exactly 1"):
        load_campaign_config(path, CampaignOverrides(preset="synthetic-smoke"))


def test_storage_minimum_free_space_is_fixed_to_twenty_gib_protocol(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "  minimum_free_gb: 20.0\n", "  minimum_free_gb: 19.0\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="storage.minimum_free_gb must be exactly 20.0"):
        load_campaign_config(path, CampaignOverrides(preset="synthetic-smoke"))


def test_identity_completion_contains_every_audit_field():
    seed = RunIdentitySeed(
        schema_version=1,
        campaign_config_sha256="1" * 64,
        source_commit="2" * 40,
        source_dirty=False,
        dataset="kitti",
        scene="04",
        frame_start=0,
        frame_stop=80,
        frame_stride=1,
        staged_manifest_sha256="3" * 64,
        segmentation_method="atomic",
        atomic_split_mode="conservative",
        window_reference_enabled=True,
        window_reference_config={
            "sampling_stride": 4,
            "max_keyframes": 4,
            "relative_depth_tolerance": 0.05,
            "min_reference_score": 0.30,
            "stop_coverage_ratio": 0.90,
            "min_coverage_gain": 0.03,
            "min_region_correspondences": 8,
            "min_region_coverage": 0.10,
            "min_region_purity": 0.80,
            "merge_vote_threshold": 0.80,
        },
        reconstruction_mode="no_loop",
        window_size=75,
        overlap=30,
        model_name="pi3",
        model_dtype="bfloat16",
        checkpoint_sha256="4" * 64,
    )
    identity = complete_identity(seed, "5" * 64)
    assert set(identity.to_payload()) == {
        *seed.to_payload(),
        "prediction_key",
    }
    assert identity.prediction_key == "5" * 64


def test_plan_dry_run_does_not_import_torch_or_touch_inputs(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "      raise RuntimeError('torch import forbidden')\n"
        "sys.meta_path.insert(0, Block())\n",
        encoding="utf-8",
    )
    missing = tmp_path / "does-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_window_reference_campaign.py"),
            "plan",
            "--config", str(CONFIG),
            "--preset", "kitti-small",
            "--data-root", str(missing),
            "--checkpoint", str(missing / "model.safetensors"),
            "--output-root", str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [run["run_id"] for run in payload["runs"][:6]] == [
        "depth__wr-off", "depth__wr-on",
        "geometry__wr-off", "geometry__wr-on",
        "atomic__wr-off", "atomic__wr-on",
    ]
    assert not (tmp_path / "output").exists()


def test_plan_without_dry_run_stays_torch_import_light_and_publishes_plan(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "      raise RuntimeError('torch import forbidden')\n"
        "sys.meta_path.insert(0, Block())\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_window_reference_campaign.py"),
            "plan",
            "--config", str(CONFIG),
            "--preset", "kitti-small",
            "--data-root", str(tmp_path / "missing-data"),
            "--checkpoint", str(tmp_path / "missing-model.safetensors"),
            "--output-root", str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    plan_path = output / "window-reference-v1" / "plan.json"
    assert plan_path.is_file()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 18


@pytest.mark.parametrize("preset", ["kitti-small", "kitti-formal-subset"])
def test_multi_scene_presets_have_unique_relative_run_directories(preset):
    loaded = load_campaign_config(CONFIG, CampaignOverrides(preset=preset))
    plan = build_plan(loaded)
    relative_dirs = [run.relative_run_dir for run in plan.runs]
    assert len(relative_dirs) == len(set(relative_dirs))


def test_standalone_summary_uses_concrete_kitti_stop_in_identity_key(tmp_path):
    from PIL import Image

    data_root = tmp_path / "data"
    image_dir = data_root / "KITTI" / "04" / "image_2"
    image_dir.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (8, 6), color=(index, 0, 0)).save(
            image_dir / f"{index:06d}.png"
        )
    import numpy as np

    np.savetxt(data_root / "KITTI" / "04" / "poses.txt", np.ones((2, 12)))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture-checkpoint")
    loaded = load_campaign_config(
        CONFIG,
        CampaignOverrides(
            preset="kitti-small",
            scene_ids=("kitti-04",),
            data_root=data_root,
            checkpoint=checkpoint,
            output_root=tmp_path / "outputs",
        ),
    )
    from experiments.window_reference_campaign.cli import _expected_summary_seeds
    plan = build_plan(loaded)
    expected = _expected_summary_seeds(loaded, plan)
    assert expected
    assert {key[2] for key in expected} == {"f000000-000002-s1"}


def test_kitti_preflight_stage_run_and_standalone_summary_share_identity_seed(tmp_path):
    from PIL import Image
    import numpy as np

    data_root = tmp_path / "data"
    image_dir = data_root / "KITTI" / "04" / "image_2"
    image_dir.mkdir(parents=True)
    for index in range(80):
        Image.new("RGB", (8, 6), color=(index % 255, 0, 0)).save(
            image_dir / f"{index:06d}.png"
        )
    np.savetxt(data_root / "KITTI" / "04" / "poses.txt", np.ones((80, 12)))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture-checkpoint")
    loaded = load_campaign_config(
        CONFIG,
        CampaignOverrides(
            preset="kitti-smoke",
            scene_ids=("kitti-04",),
            data_root=data_root,
            checkpoint=checkpoint,
            output_root=tmp_path / "outputs",
        ),
    )
    plan = build_plan(loaded)
    source_state = _source_metadata(loaded.config.repository_root)
    dependencies = PreflightDependencies(
        python_version=(3, 11, 15),
        import_module=lambda name: object(),
        git_state=lambda _: GitState(source_state[0], source_state[1]),
        cuda_state=lambda _: CudaState(False, 0, None, None, None),
        disk_usage=lambda _: DiskUsage(100 * 1024**3, 0, 100 * 1024**3),
    )

    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=dependencies,
    )
    identity_check = next(
        check for check in report.checks if check.name == "identity:kitti-04"
    )
    assert identity_check.status == "ok"

    resolved = resolve_scene(loaded.config, loaded.config.selected_scenes[0])
    staged = stage_scene(resolved, loaded.config.campaign_root)
    assert staged.identity_manifest_sha256 == identity_manifest_sha256(resolved)
    assert staged.manifest_sha256 != staged.identity_manifest_sha256

    def execute(request, loaded_config):
        request.artifact_dir.mkdir(parents=True, exist_ok=True)
        (request.artifact_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
        return PipelineExecution(
            artifact_dir=request.artifact_dir,
            artifact_manifest_sha256="b" * 64,
            prediction_key="a" * 64,
            frame_count=80,
            diagnostics_payload={
                "stage_timings_ms": {"reconstruction": 1.0},
                "segmentation_summaries": [
                    {
                        "window_index": 0,
                        "frame_index": 0,
                        "region_count": 1,
                        "window_reference_applied": (
                            request.planned.variant.window_reference_enabled
                        ),
                        "window_reference_keyframes": "0",
                        "window_reference_keyframe_count": 1,
                        "window_reference_is_keyframe": True,
                        "window_reference_coverage_ratio": 1.0,
                        "window_reference_regions_before": 1,
                        "window_reference_regions_after": 1,
                        "window_reference_candidate_edges": 0,
                        "window_reference_accepted_edges": 0,
                        "window_reference_conflict_edges": 0,
                        "window_reference_projected_samples": 0,
                        "window_reference_occluded_samples": 0,
                        "window_reference_depth_rejected_samples": 0,
                        "window_reference_fallback": "not_applicable",
                    }
                ],
                "candidate_count": 0,
                "constraint_count": 0,
                "mode_scalars": {"window_count": 1},
            },
            cache_stats=CacheStats(0, 0, 0, 0.0, 0.0, 0, 0, ()),
        )

    evaluation_metrics = {
        "internal_ate_rmse_m": 0.0,
        "internal_rpe_translation_rmse_m": 0.0,
        "internal_rpe_rotation_rmse_deg": 0.0,
        "internal_matched_frame_count": 80,
    }
    runner_dependencies = RunnerDependencies(
        execute_pipeline=execute,
        evaluate_artifact=lambda request, execution, loaded_config: EvaluationOutput(
            EvaluationKind.INTERNAL_TRAJECTORY,
            evaluation_metrics,
            (),
        ),
        stage_scene=stage_scene,
        validate_artifact=lambda path, seed: ("a" * 64, "b" * 64),
        monotonic=itertools.count(1).__next__,
        utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    outcome = run_campaign(
        loaded,
        plan,
        {resolved.scene_id: resolved},
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=False,
        dependencies=runner_dependencies,
        source_state=source_state,
    )
    assert outcome.exit_code == 0

    seed_digests = tuple(
        hashlib.sha256(
            json.dumps(
                record.identity_seed.to_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        for record in outcome.records
    )
    assert seed_digests == report.identity_seed_sha256

    from experiments.window_reference_campaign.cli import main

    assert main(
        [
            "summarize",
            "--config",
            str(CONFIG),
            "--preset",
            "kitti-smoke",
            "--scene",
            "kitti-04",
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    ) == 0


def test_normal_pipeline_enum_import_errors_are_not_swallowed(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname == 'pipeline.config':\n"
        "      raise RuntimeError('pipeline config initialization failed')\n"
        "sys.meta_path.insert(0, Block())\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.pop("LASER_WINDOW_REFERENCE_IMPORT_LIGHT", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.window_reference_campaign.config",
        ],
        cwd=ROOT,
        env={**env, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "pipeline config initialization failed" in completed.stderr
