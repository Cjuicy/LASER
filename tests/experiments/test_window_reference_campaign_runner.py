from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

import experiments.window_reference_campaign.runner as runner_module

from experiments.window_reference_campaign.config import (
    CachePolicy,
    CampaignOverrides,
    DatasetKind,
    EvaluationKind,
    FailurePolicy,
    load_campaign_config,
)
from experiments.window_reference_campaign.matrix import (
    CampaignPlan,
    build_identity_seed,
    build_plan,
)
from experiments.window_reference_campaign.results import (
    CacheStats,
    FailureStage,
    read_run_record,
    RunStatus,
)
from experiments.window_reference_campaign.runner import (
    EvaluationOutput,
    PipelineExecution,
    RunnerDependencies,
    cache_entry_complete,
    next_attempt,
    run_campaign,
    select_cache_mode,
)
from experiments.window_reference_campaign.scenes import (
    ResolvedFrameSelection,
    ResolvedScene,
)
from experiments.window_reference_campaign.staging import StagedScene, guarded_remove
from pipeline.config import PredictionCacheMode


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_CONFIG = ROOT / "configs/experiments/window_reference_campaign.yaml"
WINDOW_REFERENCE = {
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
}


def _disabled_diagnostics_payload():
    return {
        "stage_timings_ms": {"reconstruction": 1.0},
        "segmentation_summaries": [
            {"window_index": 0, "frame_index": 0, "region_count": 1}
        ],
        "candidate_count": 0,
        "constraint_count": 0,
        "mode_scalars": {"window_count": 1},
    }


def _enabled_diagnostics_payload():
    observation = {
        "window_index": 0,
        "frame_index": 0,
        "region_count": 1,
        "window_reference_applied": False,
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
        "window_reference_fallback": "none",
    }
    return {
        "stage_timings_ms": {"reconstruction": 1.0},
        "segmentation_summaries": [observation],
        "candidate_count": 0,
        "constraint_count": 0,
        "mode_scalars": {"window_count": 1},
    }


def _fixture_scene(tmp_path: Path) -> ResolvedScene:
    image = tmp_path / "data" / "synthetic" / "000.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fixture-image")
    return ResolvedScene(
        scene_id="synthetic-fixture",
        dataset=DatasetKind.SYNTHETIC,
        scene="deterministic-pointmaps",
        slice_id="f000000-000004-s1",
        approved_data_root=image.parents[2].resolve(),
        source_images=(image.resolve(),),
        selection=ResolvedFrameSelection(0, 4, 1, (0,)),
        evaluation_kind=EvaluationKind.NONE,
        poses_path=None,
        prepared_gt_path=None,
        frame_index_map=None,
        expected_gt_shape=None,
    )


def _fixture_staged(
    tmp_path: Path,
    scene: ResolvedScene,
    *,
    campaign_root: Path | None = None,
) -> StagedScene:
    stage_root = tmp_path / "campaign" if campaign_root is None else campaign_root
    stage = stage_root / "prepared" / "synthetic" / scene.slice_id
    image_dir = stage / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image = image_dir / "000000.png"
    image.write_bytes(b"fixture-image")
    manifest = stage / "staging.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return StagedScene(
        scene_id=scene.scene_id,
        dataset=scene.dataset,
        scene=scene.scene,
        slice_id=scene.slice_id,
        image_dir=image_dir,
        source_frame_ids=scene.selection.source_frame_ids,
        selection=scene.selection,
        poses_path=None,
        pointcloud_gt_path=None,
        manifest_path=manifest,
        manifest_sha256="3" * 64,
    )


def make_runner_fixture(tmp_path: Path, *, preset: str = "synthetic-smoke"):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture-checkpoint")
    loaded = load_campaign_config(
        CAMPAIGN_CONFIG,
        CampaignOverrides(
            preset=preset,
            output_root=tmp_path / "outputs",
            data_root=tmp_path / "data",
            checkpoint=checkpoint,
        ),
    )
    plan = build_plan(loaded)
    scene = _fixture_scene(tmp_path)
    staged = _fixture_staged(tmp_path, scene)
    return loaded, plan, {scene.scene_id: scene}, staged


def replace_cache_policy(loaded, policy):
    runtime = replace(loaded.config.runtime, cache_policy=policy)
    return replace(loaded, config=replace(loaded.config, runtime=runtime))


def _test_dependencies(execute, evaluate, *, validate_artifact=None, stage_scene=None):
    ticks = itertools.count(1)

    def stage(scene, root):
        return _fixture_staged(Path(root).parent, scene, campaign_root=Path(root))

    return RunnerDependencies(
        execute_pipeline=execute,
        evaluate_artifact=evaluate,
        stage_scene=stage_scene or stage,
        validate_artifact=validate_artifact or (lambda path, seed: ("a" * 64, "b" * 64)),
        monotonic=lambda: float(next(ticks)),
        utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _execution(request, *, prediction_key="a" * 64):
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    (request.artifact_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    if request.cache_mode is not PredictionCacheMode.OFF:
        entry = request.cache_root / "v2" / prediction_key
        (entry / "windows").mkdir(parents=True, exist_ok=True)
        (entry / "manifest.json").write_text("{}\n", encoding="utf-8")
        (entry / "sequence.json").write_text("{}\n", encoding="utf-8")
        (entry / "windows/000000.pt").write_bytes(b"fixture-window")
        (entry / "complete.json").write_text(
            json.dumps({"schema_version": 2, "key": prediction_key, "window_count": 1}),
            encoding="utf-8",
        )
    return PipelineExecution(
        artifact_dir=request.artifact_dir,
        artifact_manifest_sha256="b" * 64,
        prediction_key=prediction_key,
        frame_count=40,
        diagnostics_payload=(
            _enabled_diagnostics_payload()
            if request.planned.variant.window_reference_enabled
            else _disabled_diagnostics_payload()
        ),
        cache_stats=CacheStats(0, 1, 0, 0.0, 1.0, 1, 1024, ()),
    )


def _no_evaluation(request, execution, loaded):
    return EvaluationOutput(EvaluationKind.NONE, None, ())


def test_identity_seed_copies_all_fixed_campaign_axes(tmp_path):
    loaded, plan, _, _ = make_runner_fixture(tmp_path)
    seed = build_identity_seed(
        loaded=loaded,
        planned=plan.runs[0],
        frame_start=0,
        frame_stop=4,
        frame_stride=1,
        staged_manifest_sha256="3" * 64,
        source_commit="2" * 40,
        source_dirty=False,
        checkpoint_sha256="4" * 64,
    )
    assert seed.dataset == "synthetic"
    assert seed.scene == "deterministic-pointmaps"
    assert seed.window_reference_config == WINDOW_REFERENCE
    assert seed.segmentation_method == "depth"
    assert seed.window_reference_enabled is False
    assert seed.reconstruction_mode == "no_loop"
    assert (seed.window_size, seed.overlap, seed.model_name, seed.model_dtype) == (
        75,
        30,
        "pi3",
        "bfloat16",
    )


def test_first_pending_uses_auto_then_all_remaining_use_readonly(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    modes = []

    def execute(request, loaded_config):
        modes.append(request.cache_mode.value)
        return _execution(request)

    outcome = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )

    assert outcome.exit_code == 0
    assert modes == ["auto", "readonly", "readonly", "readonly", "readonly", "readonly"]
    assert {record.identity.prediction_key for record in outcome.records if record.identity} == {"a" * 64}


def test_resume_with_cleaned_cache_returns_first_remaining_run_to_auto(tmp_path):
    loaded, plan, scenes, staged = make_runner_fixture(tmp_path)
    # The runner writes immutable records; this helper is deliberately a real
    # record round-trip rather than a production test-only constructor.
    modes = []

    def execute(request, loaded_config):
        modes.append(request.cache_mode.value)
        return _execution(request)

    first = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )
    assert first.exit_code == 0
    first_two = first.records[:2]
    for record, planned in zip(first_two, plan.runs[:2], strict=True):
        assert record.status is RunStatus.SUCCEEDED
        assert (loaded.config.campaign_root / planned.relative_run_dir / "run.json").is_file()

    modes.clear()
    # Remove the shared cache completion marker to model a cleaned cache while
    # retaining exact completed records.
    cache_root = loaded.config.campaign_root / "work/cache/synthetic/f000000-000004-s1"
    for path in cache_root.rglob("complete.json"):
        path.unlink()
    outcome = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )
    assert outcome.skipped_run_ids == ("depth__wr-off", "depth__wr-on", "geometry__wr-off", "geometry__wr-on", "atomic__wr-off", "atomic__wr-on")
    assert modes == []


def test_no_resume_refuses_to_replace_an_immutable_success(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    first = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=_test_dependencies(lambda request, config: _execution(request), _no_evaluation),
    )
    assert first.exit_code == 0
    with pytest.raises(ValueError, match="new output root or campaign ID"):
        run_campaign(
            loaded,
            plan,
            scenes,
            resume=False,
            failure_policy=FailurePolicy.FAIL_FAST,
            keep_artifacts=True,
            dependencies=_test_dependencies(lambda request, config: _execution(request), _no_evaluation),
        )


def test_refresh_applies_to_first_pending_after_skipped_successes(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    first = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=_test_dependencies(lambda request, config: _execution(request), _no_evaluation),
    )
    assert first.exit_code == 0
    for planned in plan.runs[2:]:
        (loaded.config.campaign_root / planned.relative_run_dir / "run.json").unlink()
    loaded_refresh = replace_cache_policy(loaded, CachePolicy.REFRESH)
    modes = []

    def execute(request, loaded_config):
        modes.append(request.cache_mode.value)
        return _execution(request)

    outcome = run_campaign(
        loaded_refresh,
        build_plan(loaded_refresh),
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )
    assert outcome.exit_code == 0
    assert modes == ["refresh", "readonly", "readonly", "readonly"]


def test_failure_policy_preserves_attempt_and_returns_nonzero(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    calls = []

    def execute(request, loaded_config):
        calls.append(request.planned.run_id)
        request.log_path.write_text("actionable failure\n", encoding="utf-8")
        if request.planned.run_id == "depth__wr-off":
            raise RuntimeError("fixture reconstruction failed")
        return _execution(request)

    outcome = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )
    assert outcome.exit_code == 1
    assert calls == ["depth__wr-off"]
    failed = outcome.failures[0]
    assert failed.attempt == 1
    assert failed.failure_stage is FailureStage.RECONSTRUCTION
    attempt = loaded.config.campaign_root / plan.runs[0].relative_run_dir / "attempts/0001/stdout.log"
    assert attempt.read_text(encoding="utf-8") == "actionable failure\n"


def test_evaluation_failure_resumes_from_valid_artifact_without_reconstruction(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    counts = Counter()

    def execute(request, loaded_config):
        counts["reconstruct"] += 1
        return _execution(request)

    def fail_evaluation(request, execution, loaded_config):
        counts["evaluate"] += 1
        if request.planned.run_id == "depth__wr-off" and counts["evaluate"] == 1:
            raise RuntimeError("evaluation interrupted")
        return _no_evaluation(request, execution, loaded_config)

    first = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, fail_evaluation),
    )
    assert first.exit_code == 1
    assert counts == Counter(reconstruct=1, evaluate=1)

    second = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, fail_evaluation),
    )
    assert second.exit_code == 0
    assert counts["reconstruct"] == 6
    assert counts["evaluate"] == 7


@pytest.mark.parametrize(
    ("requested", "complete", "expected"),
    [
        (CachePolicy.AUTO, False, "auto"),
        (CachePolicy.AUTO, True, "readonly"),
        (CachePolicy.REFRESH, True, "refresh"),
        (CachePolicy.READONLY, False, "readonly"),
        (CachePolicy.OFF, False, "off"),
    ],
)
def test_cache_policy_is_explicit(tmp_path, requested, complete, expected):
    key = "a" * 64
    cache = tmp_path / "cache"
    if complete:
        entry = cache / "v2" / key
        (entry / "windows").mkdir(parents=True)
        (entry / "manifest.json").write_text("{}\n", encoding="utf-8")
        (entry / "sequence.json").write_text("{}\n", encoding="utf-8")
        (entry / "windows/000000.pt").write_bytes(b"fixture")
        (entry / "windows/000001.pt").write_bytes(b"fixture")
        (entry / "complete.json").write_text(
            json.dumps({"schema_version": 2, "key": key, "window_count": 2}),
            encoding="utf-8",
        )
    actual = select_cache_mode(
        requested,
        first_pending=True,
        known_prediction_key=key,
        cache_root=cache,
        window_count=2,
    )
    assert actual.value == expected


def test_cache_probe_rejects_malformed_or_partial_entries(tmp_path):
    key = "a" * 64
    cache = tmp_path / "cache" / "v2" / key
    (cache / "windows").mkdir(parents=True)
    (cache / "complete.json").write_text("not-json", encoding="utf-8")
    assert not cache_entry_complete(tmp_path / "cache", key, 1)


def test_cleanup_rejects_external_data_weights_and_campaign_root(tmp_path):
    campaign = tmp_path / "campaign"
    artifact = campaign / "runs/kitti/04/depth__wr-off/artifact"
    artifact.mkdir(parents=True)
    guarded_remove(artifact, campaign)
    assert not artifact.exists()
    for forbidden in (tmp_path / "data", tmp_path / "weights", campaign):
        forbidden.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="outside campaign-owned root"):
            guarded_remove(forbidden, campaign)


def test_attempt_numbers_are_monotonic_and_never_replaced(tmp_path):
    run_dir = tmp_path / "run"
    first_number, first_path = next_attempt(run_dir)
    second_number, second_path = next_attempt(run_dir)
    assert (first_number, second_number) == (1, 2)
    assert first_path != second_path
    assert first_path.is_dir() and second_path.is_dir()


def test_production_dependencies_have_no_test_builder():
    assert not hasattr(RunnerDependencies, "for_tests")


def _two_scene_plan(loaded, plan, scenes):
    first = plan.runs[0]
    first_scene = scenes[first.scene_id]
    second_scene = replace(
        first_scene,
        scene_id="synthetic-fixture-2",
        scene="deterministic-pointmaps-2",
    )
    second = replace(
        first,
        scene_id=second_scene.scene_id,
        scene=second_scene.scene,
        relative_run_dir=(
            Path("runs")
            / first.dataset.value
            / second_scene.scene_id
            / first.slice_id
            / first.variant.run_id
        ),
    )
    return (
        CampaignPlan(plan.campaign_id, plan.preset, (first, second)),
        {first_scene.scene_id: first_scene, second_scene.scene_id: second_scene},
    )


def test_runner_stages_and_cleans_one_scene_before_staging_the_next(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    two_scene_plan, two_scenes = _two_scene_plan(loaded, plan, scenes)
    stage_calls = []
    executed = []
    first_prepared = (
        loaded.config.campaign_root
        / "prepared"
        / "synthetic"
        / two_scene_plan.runs[0].slice_id
    )

    def stage(scene, root):
        stage_calls.append(scene.scene_id)
        if scene.scene_id == two_scene_plan.runs[1].scene_id:
            if first_prepared.exists():
                raise RuntimeError("prepared scenes coexist")
            raise RuntimeError("second scene staging failed")
        return _fixture_staged(tmp_path, scene, campaign_root=Path(root))

    def execute(request, loaded_config):
        executed.append(request.planned.scene_id)
        return _execution(request)

    with pytest.raises(RuntimeError, match="second scene staging failed"):
        run_campaign(
            loaded,
            two_scene_plan,
            two_scenes,
            resume=True,
            failure_policy=FailurePolicy.KEEP_GOING,
            keep_artifacts=True,
            dependencies=_test_dependencies(
                execute, _no_evaluation, stage_scene=stage
            ),
        )

    summary = json.loads(
        (loaded.config.campaign_root / "summary.json").read_text(encoding="utf-8")
    )
    assert stage_calls == ["synthetic-fixture", "synthetic-fixture-2"]
    assert executed == ["synthetic-fixture"]
    assert summary["completed_runs"] == 1
    assert not first_prepared.exists()


def test_fresh_execution_validates_artifact_before_success_or_cleanup(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    validations = []

    def validate(path, seed):
        validations.append(path)
        return "a" * 64, "b" * 64

    outcome = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=False,
        dependencies=_test_dependencies(
            lambda request, config: _execution(request),
            _no_evaluation,
            validate_artifact=validate,
        ),
    )

    assert outcome.exit_code == 0
    assert len(validations) == 6
    assert all(path.name == "artifact" for path in validations)
    assert all(not path.exists() for path in validations)


@pytest.mark.parametrize("mismatch", ["path", "prediction_key", "digest"])
def test_fresh_execution_artifact_mismatch_is_failed_and_not_cleaned(
    tmp_path, mismatch
):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)

    def execute(request, loaded_config):
        execution = _execution(request)
        if mismatch == "path":
            return replace(execution, artifact_dir=request.run_dir / "wrong-artifact")
        return execution

    def validate(path, seed):
        if mismatch == "prediction_key":
            return "c" * 64, "b" * 64
        if mismatch == "digest":
            return "a" * 64, "c" * 64
        return "a" * 64, "b" * 64

    outcome = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=False,
        dependencies=_test_dependencies(execute, _no_evaluation, validate_artifact=validate),
    )

    record = read_run_record(
        loaded.config.campaign_root / plan.runs[0].relative_run_dir / "run.json"
    )
    assert outcome.exit_code == 1
    assert len(outcome.failures) == 1
    assert record.status is RunStatus.FAILED
    assert (loaded.config.campaign_root / plan.runs[0].relative_run_dir / "artifact").exists()


def test_compaction_failure_retries_validated_artifact_without_reconstruction(tmp_path):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    single_plan = CampaignPlan(plan.campaign_id, plan.preset, (plan.runs[0],))
    counts = Counter()

    def execute(request, loaded_config):
        counts["reconstruct"] += 1
        return _execution(request)

    def evaluate(request, execution, loaded_config):
        counts["evaluate"] += 1
        if counts["evaluate"] == 1:
            return EvaluationOutput(EvaluationKind.NONE, {"unexpected": 1}, ())
        return _no_evaluation(request, execution, loaded_config)

    first = run_campaign(
        loaded,
        single_plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, evaluate),
    )
    assert first.exit_code == 1
    assert first.failures[0].failure_stage is FailureStage.COMPACTION

    second = run_campaign(
        loaded,
        single_plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, evaluate),
    )

    assert second.exit_code == 0
    assert counts == Counter(reconstruct=1, evaluate=2)


def test_cleanup_failure_keeps_success_record_and_resume_does_not_reconstruct(
    tmp_path, monkeypatch
):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)
    counts = Counter()
    original_remove = runner_module.guarded_remove
    failed_once = False

    def remove(path, root):
        nonlocal failed_once
        if Path(path).name == "artifact" and not failed_once:
            failed_once = True
            raise OSError("artifact cleanup interrupted")
        return original_remove(path, root)

    monkeypatch.setattr(runner_module, "guarded_remove", remove)

    def execute(request, loaded_config):
        counts["reconstruct"] += 1
        return _execution(request)

    first = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=False,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )
    first_path = loaded.config.campaign_root / plan.runs[0].relative_run_dir
    first_record = read_run_record(first_path / "run.json")
    assert first.exit_code == 1
    assert first_record.status is RunStatus.SUCCEEDED
    assert (first_path / "artifact").exists()

    monkeypatch.setattr(runner_module, "guarded_remove", original_remove)
    second = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=False,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )

    assert second.exit_code == 0
    assert counts["reconstruct"] == 6
    assert not (first_path / "artifact").exists()
    assert read_run_record(first_path / "run.json").status is RunStatus.SUCCEEDED


@pytest.mark.parametrize(
    "policy", [FailurePolicy.FAIL_FAST, FailurePolicy.KEEP_GOING]
)
def test_malformed_diagnostics_still_publishes_failed_record_and_nonzero(
    tmp_path, policy
):
    loaded, plan, scenes, _ = make_runner_fixture(tmp_path)

    def execute(request, loaded_config):
        execution = _execution(request)
        malformed = dict(execution.diagnostics_payload)
        malformed["mode_scalars"] = {"window_count": "invalid"}
        return replace(execution, diagnostics_payload=malformed)

    outcome = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=policy,
        keep_artifacts=True,
        dependencies=_test_dependencies(execute, _no_evaluation),
    )

    assert outcome.exit_code == 1
    assert outcome.failures
    first_record = read_run_record(
        loaded.config.campaign_root / plan.runs[0].relative_run_dir / "run.json"
    )
    assert first_record.status is RunStatus.FAILED
