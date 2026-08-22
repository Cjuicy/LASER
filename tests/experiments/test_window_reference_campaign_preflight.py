from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import experiments.window_reference_campaign.preflight as preflight

from experiments.window_reference_campaign.config import (
    CampaignOverrides,
    load_campaign_config,
)
from experiments.window_reference_campaign.matrix import build_plan
from experiments.window_reference_campaign.preflight import (
    CudaState,
    DiskUsage,
    GitState,
    PreflightDependencies,
    _default_cuda_state,
    _validate_pointmap,
    bridge_checkpoint_path,
    build_bootstrap_actions,
    preflight_campaign,
    run_bootstrap,
    write_preflight_report,
)
from experiments.window_reference_campaign.results import redact_argv


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/window_reference_campaign.yaml"


def make_preflight_fixture(tmp_path: Path, *, preset: str = "kitti-smoke"):
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
            preset=preset,
            data_root=data_root,
            checkpoint=checkpoint,
            output_root=tmp_path / "outputs",
        ),
    )
    return loaded, build_plan(loaded)


def _dependencies(*, cuda_available: bool) -> PreflightDependencies:
    modules = {
        name: type("SentinelModule", (), {"__version__": "fixture"})()
        for name in (*preflight._REQUIRED_IMPORTS, *preflight._SYNTHETIC_REQUIRED_IMPORTS)
    }
    git = GitState("1" * 40, False)
    cuda = CudaState(
        available=cuda_available,
        device_count=1 if cuda_available else 0,
        selected_device=0 if cuda_available else None,
        device_name="fixture-gpu" if cuda_available else None,
        bfloat16_supported=True if cuda_available else None,
    )
    usage = DiskUsage(100 * 1024**3, 0, 100 * 1024**3)
    return PreflightDependencies(
        python_version=(3, 11, 15),
        import_module=lambda name: modules[name],
        git_state=lambda _: git,
        cuda_state=lambda _: cuda,
        disk_usage=lambda _: usage,
    )


def test_preflight_dependencies_have_no_production_test_builder():
    assert not hasattr(PreflightDependencies, "for_tests")


def test_bootstrap_dry_run_works_when_torch_and_omegaconf_imports_are_blocked(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname in {'torch', 'omegaconf'} or fullname.startswith(('torch.', 'omegaconf.')):\n"
        "      raise RuntimeError('heavy import forbidden during bootstrap')\n"
        "sys.meta_path.insert(0, Block())\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_window_reference_campaign.py"),
            "bootstrap",
            "--repository",
            str(ROOT),
            "--dry-run",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "git submodule update --init --recursive" in completed.stdout
    assert "setup.py build_ext --inplace" in completed.stdout


def test_bootstrap_actions_never_download_protected_datasets(tmp_path):
    actions = build_bootstrap_actions(
        ROOT,
        python_executable=sys.executable,
        external_checkpoint=None,
    )
    commands = "\n".join(shlex.join(action.argv) for action in actions)
    assert "scripts/download_weights.sh" in commands
    lowered = commands.lower()
    for forbidden in ("kitti", "7-scenes", "7scenes", "neuralrgbd", "cookie", "password", "token"):
        assert forbidden not in lowered


def test_bootstrap_default_weight_path_bridges_public_download_to_pi3_contract(tmp_path):
    source = tmp_path / "weights" / "model.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture-weight")
    actions = build_bootstrap_actions(
        tmp_path,
        python_executable=sys.executable,
        external_checkpoint=None,
    )
    bridge = next(action for action in actions if action.label == "pi3-default-path")
    subprocess.run(bridge.argv, cwd=bridge.cwd, check=True)
    target = tmp_path / "weights" / "PI3" / "model.safetensors"
    assert target.is_file()
    assert not target.is_symlink()
    assert target.read_bytes() == source.read_bytes()
    assert not os.path.samefile(source, target)


@pytest.mark.parametrize("bad_layout", ("source", "pi3-parent"))
def test_bootstrap_path_guard_rejects_bad_layout_before_download(tmp_path, bad_layout):
    weights = tmp_path / "weights"
    source = weights / "model.safetensors"
    target = weights / "PI3" / "model.safetensors"
    weights.mkdir()
    if bad_layout == "source":
        external = tmp_path / "external-model.safetensors"
        external.write_bytes(b"outside")
        source.symlink_to(external)
    else:
        source.write_bytes(b"fixture-weight")
        external_parent = tmp_path / "external-PI3"
        external_parent.mkdir()
        (weights / "PI3").symlink_to(external_parent, target_is_directory=True)

    actions = build_bootstrap_actions(
        tmp_path,
        python_executable=sys.executable,
        external_checkpoint=None,
    )
    by_argv = {action.argv: action.label for action in actions}
    labels = []

    def run(argv, *, cwd, check):
        label = by_argv[tuple(argv)]
        labels.append(label)
        if label == "pi3-default-guard":
            return subprocess.run(argv, cwd=cwd, check=check)
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(subprocess.CalledProcessError):
        run_bootstrap(actions, execute=True, run=run)
    assert labels[-1] == "pi3-default-guard"
    assert "public-pi3-weight" not in labels


def test_bootstrap_repeat_guard_detaches_target_before_interrupted_download(tmp_path):
    source = tmp_path / "weights" / "model.safetensors"
    target = tmp_path / "weights" / "PI3" / "model.safetensors"
    target.parent.mkdir(parents=True)
    source.write_bytes(b"stable-checkpoint")
    os.link(source, target)
    actions = build_bootstrap_actions(
        tmp_path,
        python_executable=sys.executable,
        external_checkpoint=None,
    )
    by_argv = {action.argv: action.label for action in actions}
    labels = []

    def run(argv, *, cwd, check):
        label = by_argv[tuple(argv)]
        labels.append(label)
        if label == "pi3-default-guard":
            return subprocess.run(argv, cwd=cwd, check=check)
        if label == "public-pi3-weight":
            source.write_bytes(b"interrupted-download")
            raise RuntimeError("download interrupted")
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(RuntimeError, match="download interrupted"):
        run_bootstrap(actions, execute=True, run=run)
    assert labels.index("pi3-default-guard") < labels.index("public-pi3-weight")
    assert source.read_bytes() == b"interrupted-download"
    assert target.read_bytes() == b"stable-checkpoint"


def test_checkpoint_bridge_reuses_identical_target_without_clobber(tmp_path):
    source = tmp_path / "weights" / "model.safetensors"
    target = tmp_path / "weights" / "PI3" / "model.safetensors"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"fixture-weight")
    target.write_bytes(source.read_bytes())
    before = target.stat().st_ino

    result = bridge_checkpoint_path(source, target, repository=tmp_path)

    assert result == target
    assert target.stat().st_ino == before
    assert target.read_bytes() == source.read_bytes()


def test_checkpoint_bridge_rejects_mismatched_target(tmp_path):
    source = tmp_path / "weights" / "model.safetensors"
    target = tmp_path / "weights" / "PI3" / "model.safetensors"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    target.write_bytes(b"different")

    with pytest.raises(ValueError, match="target checkpoint differs"):
        bridge_checkpoint_path(source, target, repository=tmp_path)


def test_checkpoint_bridge_rejects_source_and_parent_symlinks(tmp_path):
    source = tmp_path / "weights" / "model.safetensors"
    source.parent.mkdir(parents=True)
    real_source = tmp_path / "real-model.safetensors"
    real_source.write_bytes(b"source")
    source.symlink_to(real_source)
    target = tmp_path / "weights" / "PI3" / "model.safetensors"

    with pytest.raises(ValueError, match="source checkpoint must not be a symlink"):
        bridge_checkpoint_path(source, target, repository=tmp_path)

    source.unlink()
    source.write_bytes(b"source")
    real_parent = tmp_path / "real-PI3"
    real_parent.mkdir()
    (tmp_path / "weights" / "PI3").symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        bridge_checkpoint_path(source, target, repository=tmp_path)


def test_checkpoint_bridge_rejects_target_symlink(tmp_path):
    source = tmp_path / "weights" / "model.safetensors"
    target = tmp_path / "weights" / "PI3" / "model.safetensors"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    target_link = tmp_path / "target.safetensors"
    target_link.write_bytes(b"source")
    target.symlink_to(target_link)

    with pytest.raises(ValueError, match="target checkpoint must not be a symlink"):
        bridge_checkpoint_path(source, target, repository=tmp_path)


def test_checkpoint_bridge_rechecks_racing_exdev_target(monkeypatch, tmp_path):
    source = tmp_path / "weights" / "model.safetensors"
    target = tmp_path / "weights" / "PI3" / "model.safetensors"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    calls = 0

    def race_link(source_path, target_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            target.write_bytes(b"raced-different")
            raise FileExistsError(target_path)

    monkeypatch.setattr(preflight.os, "link", race_link)
    with pytest.raises(ValueError, match="target checkpoint differs"):
        bridge_checkpoint_path(source, target, repository=tmp_path)


def test_run_bootstrap_prints_shell_quoted_commands_and_executes_in_order(tmp_path, capsys):
    actions = build_bootstrap_actions(
        tmp_path,
        python_executable="python with space",
        external_checkpoint=tmp_path / "checkpoint.safetensors",
    )
    assert run_bootstrap(actions, execute=False) == 0
    output = capsys.readouterr().out
    assert "cd " in output
    assert "'python with space'" in output

    calls = []

    def run(argv, *, cwd, check):
        calls.append((argv, cwd, check))
        return subprocess.CompletedProcess(argv, 0)

    assert run_bootstrap(actions, execute=True, run=run) == 0
    assert [item[0] for item in calls] == [action.argv for action in actions]
    assert all(item[1] == tmp_path.resolve() and item[2] is True for item in calls)


def test_default_cuda_state_checks_selected_device_bfloat16_and_restores_current(monkeypatch):
    class DeviceContext:
        def __init__(self, cuda, index):
            self.cuda = cuda
            self.index = index
            self.previous = None

        def __enter__(self):
            self.previous = self.cuda.current
            self.cuda.entered.append(self.index)
            self.cuda.current = self.index
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.cuda.current = self.previous
            return False

    class FakeCuda:
        current = 0
        entered = []
        queried = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return f"fake-{index}"

        @classmethod
        def device(cls, index):
            return DeviceContext(cls, index)

        @classmethod
        def is_bf16_supported(cls):
            cls.queried.append(cls.current)
            return cls.current == 1

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    gpu0 = _default_cuda_state(0)
    gpu1 = _default_cuda_state(1)

    assert gpu0.bfloat16_supported is False
    assert gpu1.bfloat16_supported is True
    assert FakeCuda.queried == [0, 1]
    assert FakeCuda.entered == [0, 1]
    assert FakeCuda.current == 0


def test_default_cuda_state_context_or_api_failure_is_conservative(monkeypatch):
    class BrokenCuda:
        current = 0
        entered = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return f"fake-{index}"

        @classmethod
        def device(cls, index):
            cls.entered.append(index)
            raise RuntimeError("device context failed")

        @staticmethod
        def is_bf16_supported():
            raise AssertionError("must not query outside selected device context")

    fake_torch = type("FakeTorch", (), {"cuda": BrokenCuda})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    state = _default_cuda_state(1)

    assert state.bfloat16_supported is False
    assert BrokenCuda.entered == [1]
    assert BrokenCuda.current == 0


def _pointmap_archive(tmp_path: Path, *, include_archive_ids: bool = True) -> Path:
    pointmap = tmp_path / "ground_truth.npz"
    payload = {
        "point_maps": np.zeros((2, 2, 3, 3), dtype=np.float32),
        "valid_mask": np.ones((2, 2, 3), dtype=bool),
    }
    if include_archive_ids:
        payload["frame_ids"] = np.array([10, 20], dtype=np.int64)
    np.savez(pointmap, **payload)
    return pointmap


def test_validate_pointmap_accepts_matching_npz_and_both_sibling_frame_id_sources(tmp_path):
    pointmap = _pointmap_archive(tmp_path)
    np.save(pointmap.parent / "source_frame_ids.npy", np.array([10, 20], dtype=np.int64))
    np.save(pointmap.parent / "frame_ids.npy", np.array([10, 20], dtype=np.int64))

    detail = _validate_pointmap(pointmap, (2, 3), (10, 20))

    assert detail["frame_count"] == 2
    assert detail["spatial_shape"] == [2, 3]


def test_validate_pointmap_rejects_conflicting_npz_and_sibling_frame_id_sources(tmp_path):
    pointmap = _pointmap_archive(tmp_path)
    np.save(pointmap.parent / "source_frame_ids.npy", np.array([10, 20], dtype=np.int64))
    np.save(pointmap.parent / "frame_ids.npy", np.array([10, 21], dtype=np.int64))

    with pytest.raises(ValueError, match="frame ID sources disagree"):
        _validate_pointmap(pointmap, (2, 3), (10, 20))


def test_validate_pointmap_rejects_non_vector_sibling_frame_ids(tmp_path):
    pointmap = _pointmap_archive(tmp_path)
    np.save(pointmap.parent / "source_frame_ids.npy", np.array([[10, 20]], dtype=np.int64))

    with pytest.raises(ValueError, match="one-dimensional integer array"):
        _validate_pointmap(pointmap, (2, 3), (10, 20))


def test_preflight_accepts_raw_pointcloud_partners_and_marks_gt_preparation_pending(tmp_path):
    data_root = tmp_path / "data"
    scene_root = data_root / "NeuralRGBD" / "thin_geometry"
    (scene_root / "images").mkdir(parents=True)
    (scene_root / "depth").mkdir(parents=True)
    frame_ids = tuple(range(0, 391, 10))
    for frame_id in frame_ids:
        Image.new("RGB", (8, 6), color=(frame_id % 255, 0, 0)).save(
            scene_root / "images" / f"img{frame_id}.png"
        )
        Image.new("I;16", (8, 6)).save(
            scene_root / "depth" / f"depth{frame_id}.png"
        )
    np.savetxt(scene_root / "poses.txt", np.tile(np.eye(4), (391, 1)))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture-checkpoint")
    loaded = load_campaign_config(
        CONFIG,
        CampaignOverrides(
            preset="pointcloud-small",
            scene_ids=("nrgbd-thin-geometry",),
            data_root=data_root,
            checkpoint=checkpoint,
            output_root=tmp_path / "outputs",
        ),
    )
    report = preflight_campaign(
        loaded,
        build_plan(loaded),
        allow_no_gpu=True,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.errors == ()
    scene = next(
        check for check in report.checks if check.name == "scene:nrgbd-thin-geometry"
    )
    assert scene.status == "ok"
    assert scene.detail["pointmap"]["preparation"] == "pending"
    assert scene.detail["pointmap"]["source_frame_count"] == len(frame_ids)


def test_allow_no_gpu_succeeds_but_records_not_gpu_ready(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "ok_with_warnings"
    assert report.errors == ()
    assert any("CUDA" in warning for warning in report.warnings)
    cuda = next(check for check in report.checks if check.name == "cuda")
    assert cuda.status == "warning"
    assert cuda.detail["gpu_ready"] is False


def test_strict_preflight_rejects_same_no_gpu_environment(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=False,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "error"
    assert any("CUDA" in error for error in report.errors)


def test_allow_no_gpu_does_not_downgrade_import_or_bfloat16_errors(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    base = _dependencies(cuda_available=True)

    def imports(name):
        if name in {"torch", "open3d"}:
            raise RuntimeError(f"missing {name}")
        return base.import_module(name)

    dependencies = replace(
        base,
        import_module=imports,
        cuda_state=lambda _: CudaState(True, 1, 0, "fixture-gpu", False),
    )
    report = preflight_campaign(
        loaded, plan, allow_no_gpu=True, dependencies=dependencies
    )
    assert report.status == "error"
    assert any("required import torch" in error for error in report.errors)
    assert any("required import open3d" in error for error in report.errors)
    assert any("bfloat16" in error for error in report.errors)
    cuda = next(check for check in report.checks if check.name == "cuda")
    assert cuda.status == "error"
    assert cuda.detail["gpu_ready"] is False


def test_preflight_collects_all_required_import_failures(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    base = _dependencies(cuda_available=False)

    def imports(name):
        raise RuntimeError(f"blocked {name}")

    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=replace(base, import_module=imports),
    )
    assert report.status == "error"
    assert len([item for item in report.errors if item.startswith("required import")]) == 9
    assert any(check.status == "warning" for check in report.checks if check.name == "cuda")


def test_allow_no_gpu_does_not_hide_missing_checkpoint_or_dataset(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    loaded = replace(
        loaded,
        config=replace(
            loaded.config,
            storage=replace(
                loaded.config.storage,
                checkpoint=tmp_path / "missing.safetensors",
                data_root=tmp_path / "missing-data",
            ),
        ),
    )
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "error"
    assert any("checkpoint" in error for error in report.errors)
    assert any("KITTI" in error or "data" in error for error in report.errors)
    scene = next(check for check in report.checks if check.name == "scene:kitti-04")
    assert scene.status == "error"
    identity = next(check for check in report.checks if check.name == "identity:kitti-04")
    assert identity.status == "blocked"
    assert identity.detail["blocked_by"] == "scene:kitti-04"


def test_preflight_writes_only_atomic_report_and_warns_for_space_dirty_source(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path)
    dependencies = replace(
        _dependencies(cuda_available=False),
        git_state=lambda _: GitState("2" * 40, True),
        disk_usage=lambda _: DiskUsage(
            100 * 1024**3, 90 * 1024**3, 10 * 1024**3
        ),
    )
    report = preflight_campaign(
        loaded, plan, allow_no_gpu=True, dependencies=dependencies
    )
    target = write_preflight_report(report, loaded.config.campaign_root)
    assert target == loaded.config.campaign_root / "preflight.json"
    created = sorted(
        path.relative_to(loaded.config.campaign_root).as_posix()
        for path in loaded.config.campaign_root.rglob("*") if path.is_file()
    )
    assert created == ["preflight.json"]
    assert any("20.0 GiB" in warning for warning in report.warnings)
    assert any("dirty" in warning for warning in report.warnings)
    assert len(report.identity_seed_sha256) == 6
    assert len(set(report.identity_seed_sha256)) == 6


def test_synthetic_preflight_checks_matrix_identity_storage_and_skips_gpu(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="synthetic-smoke")
    dependencies = replace(
        _dependencies(cuda_available=True),
        git_state=lambda _: GitState("2" * 40, True),
        disk_usage=lambda _: DiskUsage(
            100 * 1024**3, 92 * 1024**3, 8 * 1024**3
        ),
    )
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=dependencies,
    )
    assert report.status == "ok_with_warnings"
    assert any("20.0 GiB" in warning for warning in report.warnings)
    assert any("dirty" in warning for warning in report.warnings)
    assert [check.name for check in report.checks].count("plan:run-directories") == 1
    assert next(
        check for check in report.checks if check.name == "plan:run-directories"
    ).status == "ok"
    identity = next(
        check for check in report.checks if check.name == "identity:synthetic-fixture"
    )
    assert identity.status == "ok"
    assert identity.detail["unique"] is True
    assert len(report.identity_seed_sha256) == 6
    assert next(check for check in report.checks if check.name == "cuda").detail["not_applicable"] is True


def test_synthetic_preflight_uses_applicable_python_gate_and_targeted_imports(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="synthetic-smoke")
    calls: list[str] = []

    def imports(name: str):
        calls.append(name)
        return importlib.import_module(name)

    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=replace(
            _dependencies(cuda_available=False),
            python_version=(3, 13, 5),
            import_module=imports,
        ),
    )
    assert report.status == "ok"
    assert next(check for check in report.checks if check.name == "python").status == "ok"
    assert {
        "numpy",
        "torch",
        "inference_engine.segmentation",
        "inference_engine.prediction_cache.fingerprint",
        "pipeline.config",
    } <= set(calls)
    assert not {"open3d", "evo", "lietorch", "pi3"} & set(calls)


def test_synthetic_preflight_rejects_unsupported_python_before_campaign_checks(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="synthetic-smoke")
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=replace(_dependencies(cuda_available=False), python_version=(3, 10, 0)),
    )
    assert report.status == "error"
    assert next(check for check in report.checks if check.name == "python").status == "error"
    assert any("supported for synthetic" in error for error in report.errors)


def test_real_preflight_retains_exact_python_311_gate(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="kitti-smoke")
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=replace(_dependencies(cuda_available=False), python_version=(3, 13, 5)),
    )
    assert report.status == "error"
    assert next(check for check in report.checks if check.name == "python").status == "error"
    assert any("Python version must be exactly 3.11" in error for error in report.errors)


def test_persisted_argv_redacts_options_and_url_userinfo():
    assert redact_argv((
        "--token", "abc",
        "--password=hunter2",
        "https://alice:secret@example.invalid/repo.git",
        "--preset", "kitti-small",
    )) == (
        "--token", "<redacted>",
        "--password=<redacted>",
        "https://<redacted>@example.invalid/repo.git",
        "--preset", "kitti-small",
    )
