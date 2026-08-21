"""Standard-library bootstrap actions and read-only campaign preflight.

The module boundary is intentionally import-light.  Bootstrap must be useful
before the runtime dependencies are installed, so every campaign, NumPy,
Torch, and staging import is kept inside the check that needs it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_IMPORTS = (
    "omegaconf",
    "numpy",
    "PIL",
    "torch",
    "scipy",
    "open3d",
    "evo",
    "inference_engine.utils.fast_seg",
    "inference_engine.utils._segmentation_cy",
)
_MIN_FREE_BYTES = 20 * 1024**3
_PROTECTED_WORDS = (
    "kitti",
    "7-scenes",
    "7scenes",
    "neuralrgbd",
    "cookie",
    "password",
    "token",
    "credential",
    "secret",
)


@dataclass(frozen=True)
class BootstrapAction:
    label: str
    argv: tuple[str, ...]
    cwd: Path
    mutates_environment: bool

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("bootstrap action label must be non-empty")
        if not isinstance(self.argv, tuple) or not self.argv:
            raise ValueError("bootstrap action argv must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.argv):
            raise ValueError("bootstrap action argv must contain non-empty strings")
        if not isinstance(self.cwd, Path):
            raise ValueError("bootstrap action cwd must be a path")
        if type(self.mutates_environment) is not bool:
            raise ValueError("bootstrap action mutates_environment must be a boolean")


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def build_bootstrap_actions(
    repository: str | Path,
    *,
    python_executable: str,
    external_checkpoint: str | Path | None,
) -> tuple[BootstrapAction, ...]:
    """Build only the approved, dataset-free bootstrap command sequence."""

    root = Path(repository).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"repository is not a directory: {root}")
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("python_executable must be a non-empty string")

    actions = [
        BootstrapAction(
            "submodules",
            ("git", "submodule", "update", "--init", "--recursive"),
            root,
            True,
        ),
        BootstrapAction(
            "python-3.11",
            (
                python_executable,
                "-c",
                "import sys; assert sys.version_info[:2] == (3, 11), sys.version",
            ),
            root,
            False,
        ),
        BootstrapAction(
            "requirements",
            (python_executable, "-m", "pip", "install", "-r", "requirements.txt"),
            root,
            True,
        ),
        BootstrapAction(
            "cython-extensions",
            (python_executable, "setup.py", "build_ext", "--inplace"),
            root,
            True,
        ),
    ]
    if external_checkpoint is None:
        actions.append(
            BootstrapAction(
                "public-pi3-weight",
                ("bash", "scripts/download_weights.sh"),
                root,
                True,
            )
        )

    protected = tuple(word.casefold() for word in _PROTECTED_WORDS)
    relative_files = ("requirements.txt", "setup.py", "scripts/download_weights.sh")
    for relative in relative_files:
        if not _inside(root, root / relative):
            raise ValueError(f"bootstrap path escapes repository: {relative}")
    for action in actions:
        for token in action.argv:
            lowered = token.casefold()
            if any(word in lowered for word in protected):
                raise ValueError("bootstrap action contains a protected dataset or secret token")
    return tuple(actions)


def run_bootstrap(
    actions: Sequence[BootstrapAction],
    *,
    execute: bool,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Print or execute bootstrap actions sequentially."""

    for action in actions:
        if not isinstance(action, BootstrapAction):
            raise ValueError("bootstrap actions must contain BootstrapAction values")
        rendered = shlex.join(action.argv)
        print(f"cd {shlex.quote(str(action.cwd))} && {rendered}")
        if execute:
            run(action.argv, cwd=action.cwd, check=True)
    return 0


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool

    def __post_init__(self) -> None:
        if not isinstance(self.commit, str) or not self.commit:
            raise ValueError("git commit must be a non-empty string")
        if type(self.dirty) is not bool:
            raise ValueError("git dirty state must be a boolean")


@dataclass(frozen=True)
class CudaState:
    available: bool
    device_count: int
    selected_device: int | None
    device_name: str | None
    bfloat16_supported: bool | None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise ValueError("CUDA availability must be a boolean")
        if type(self.device_count) is not int or self.device_count < 0:
            raise ValueError("CUDA device_count must be a non-negative integer")
        if self.selected_device is not None and (
            type(self.selected_device) is not int or self.selected_device < 0
        ):
            raise ValueError("selected CUDA device must be a non-negative integer")
        if self.device_name is not None and not isinstance(self.device_name, str):
            raise ValueError("CUDA device_name must be a string or None")
        if self.bfloat16_supported is not None and type(self.bfloat16_supported) is not bool:
            raise ValueError("CUDA bfloat16 support must be a boolean or None")


@dataclass(frozen=True)
class DiskUsage:
    total: int
    used: int
    free: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (self.total, self.used, self.free)):
            raise ValueError("disk usage values must be non-negative integers")


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("preflight check name must be non-empty")
        if self.status not in {"ok", "warning", "error", "blocked"}:
            raise ValueError("preflight check status is invalid")
        if not isinstance(self.detail, Mapping):
            raise ValueError("preflight check detail must be a mapping")


@dataclass(frozen=True)
class PreflightReport:
    schema_version: int
    status: str
    allow_no_gpu: bool
    checks: tuple[PreflightCheck, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    git: GitState
    checkpoint_sha256: str
    free_disk_bytes: int
    identity_seed_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("preflight schema_version must be 1")
        if self.status not in {"ok", "ok_with_warnings", "error"}:
            raise ValueError("preflight report status is invalid")
        if type(self.allow_no_gpu) is not bool:
            raise ValueError("allow_no_gpu must be a boolean")
        if not isinstance(self.checks, tuple) or any(
            not isinstance(item, PreflightCheck) for item in self.checks
        ):
            raise ValueError("preflight checks must be a tuple of PreflightCheck")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(item, str) for item in self.warnings
        ):
            raise ValueError("preflight warnings must be a tuple of strings")
        if not isinstance(self.errors, tuple) or any(
            not isinstance(item, str) for item in self.errors
        ):
            raise ValueError("preflight errors must be a tuple of strings")
        if not isinstance(self.git, GitState):
            raise ValueError("preflight git state is invalid")
        if self.checkpoint_sha256 and not _SHA256.fullmatch(self.checkpoint_sha256):
            raise ValueError("preflight checkpoint SHA256 is invalid")
        if type(self.free_disk_bytes) is not int or self.free_disk_bytes < 0:
            raise ValueError("preflight free disk bytes must be non-negative")
        if not isinstance(self.identity_seed_sha256, tuple) or any(
            not _SHA256.fullmatch(item) for item in self.identity_seed_sha256
        ):
            raise ValueError("preflight identity seed hashes are invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "allow_no_gpu": self.allow_no_gpu,
            "checks": [
                {
                    "name": item.name,
                    "status": item.status,
                    "detail": _json_value(item.detail),
                }
                for item in self.checks
            ],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "git": {"commit": self.git.commit, "dirty": self.git.dirty},
            "checkpoint_sha256": self.checkpoint_sha256,
            "free_disk_bytes": self.free_disk_bytes,
            "identity_seed_sha256": list(self.identity_seed_sha256),
        }


@dataclass(frozen=True)
class PreflightDependencies:
    python_version: tuple[int, int, int]
    import_module: Callable[[str], object]
    git_state: Callable[[Path], GitState]
    cuda_state: Callable[[int], CudaState]
    disk_usage: Callable[[Path], DiskUsage]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.python_version, tuple)
            or len(self.python_version) != 3
            or any(type(item) is not int for item in self.python_version)
        ):
            raise ValueError("python_version must be a three-integer tuple")
        for name in ("import_module", "git_state", "cuda_state", "disk_usage"):
            if not callable(getattr(self, name)):
                raise ValueError(f"preflight dependency {name} must be callable")

    @classmethod
    def for_tests(
        cls,
        *,
        python_version: tuple[int, int, int] = (3, 11, 0),
        git: GitState | None = None,
        cuda: CudaState | None = None,
        free_bytes: int = 100 * 1024**3,
    ) -> "PreflightDependencies":
        """Build deterministic boundary doubles without importing heavy modules."""

        modules = {name: type("SentinelModule", (), {"__version__": "fixture"})() for name in _REQUIRED_IMPORTS}
        git_value = git or GitState("1" * 40, False)
        cuda_value = cuda or CudaState(False, 0, None, None, None)
        usage = DiskUsage(100 * 1024**3, 100 * 1024**3 - free_bytes, free_bytes)
        return cls(
            python_version=python_version,
            import_module=lambda name: modules[name],
            git_state=lambda _: git_value,
            cuda_state=lambda _: cuda_value,
            disk_usage=lambda _: usage,
        )


def _default_git_state(repository: Path) -> GitState:
    try:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return GitState(head or "unknown", dirty)
    except (OSError, subprocess.CalledProcessError):
        return GitState("unknown", False)


def _default_cuda_state(index: int) -> CudaState:
    try:
        import torch
    except Exception:
        return CudaState(False, 0, None, None, None)
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        if not available or index < 0 or index >= count:
            return CudaState(available, count, None, None, None)
        try:
            name = str(torch.cuda.get_device_name(index))
        except Exception:
            name = None
        try:
            # PyTorch's public helper reads the current CUDA device.  Enter
            # the selected device explicitly; passing ``device=`` is not
            # portable across supported PyTorch versions.
            with torch.cuda.device(index):
                bfloat16 = bool(torch.cuda.is_bf16_supported())
        except Exception:
            # A context/API failure must never claim dtype compatibility.
            bfloat16 = False
        return CudaState(True, count, index, name, bfloat16)
    except Exception:
        return CudaState(False, 0, None, None, None)


def default_preflight_dependencies() -> PreflightDependencies:
    """Return production dependencies while keeping imports lazy."""

    return PreflightDependencies(
        python_version=(sys.version_info.major, sys.version_info.minor, sys.version_info.micro),
        import_module=importlib.import_module,
        git_state=_default_git_state,
        cuda_state=_default_cuda_state,
        disk_usage=lambda path: DiskUsage(*shutil.disk_usage(path)),
    )


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return _json_value(asdict(value))
    if hasattr(value, "value"):
        return _json_value(getattr(value, "value"))
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _check(
    checks: list[PreflightCheck],
    name: str,
    status: str,
    detail: Mapping[str, object],
) -> None:
    checks.append(PreflightCheck(name, status, dict(detail)))


def _check_python(
    dependencies: PreflightDependencies,
    checks: list[PreflightCheck],
    errors: list[str],
) -> None:
    version = dependencies.python_version
    detail = {"version": ".".join(str(item) for item in version), "required": "3.11"}
    if version[:2] != (3, 11):
        message = f"Python version must be exactly 3.11; found {detail['version']}"
        _check(checks, "python", "error", detail)
        errors.append(message)
    else:
        _check(checks, "python", "ok", detail)


def _check_imports(
    dependencies: PreflightDependencies,
    checks: list[PreflightCheck],
    errors: list[str],
) -> None:
    for module_name in _REQUIRED_IMPORTS:
        try:
            module = dependencies.import_module(module_name)
            version = getattr(module, "__version__", None)
            detail: dict[str, object] = {"module": module_name, "available": True}
            if version is not None:
                detail["version"] = str(version)
            _check(checks, f"import:{module_name}", "ok", detail)
        except Exception as exc:
            message = f"required import {module_name} is unavailable: {exc}"
            _check(
                checks,
                f"import:{module_name}",
                "error",
                {"module": module_name, "available": False, "error": str(exc)},
            )
            errors.append(message)


def _check_git(
    repository: Path,
    dependencies: PreflightDependencies,
    checks: list[PreflightCheck],
    warnings: list[str],
    errors: list[str],
) -> GitState:
    try:
        git = dependencies.git_state(repository)
    except Exception as exc:
        git = GitState("unknown", False)
        _check(checks, "git", "error", {"commit": "unknown", "error": str(exc)})
        errors.append(f"git state could not be resolved: {exc}")
        return git
    detail = {"commit": git.commit, "dirty": git.dirty}
    if git.commit != "unknown" and _COMMIT.fullmatch(git.commit) is None:
        _check(checks, "git", "error", detail)
        errors.append(f"git HEAD commit is invalid: {git.commit}")
    elif git.commit == "unknown":
        _check(checks, "git", "error", detail)
        errors.append("git HEAD commit is unknown")
    elif git.dirty:
        _check(checks, "git", "warning", detail)
        warnings.append("source tree is dirty; identity seeds record dirty source state")
    else:
        _check(checks, "git", "ok", detail)
    return git


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_checkpoint(
    checkpoint: Path,
    checks: list[PreflightCheck],
    errors: list[str],
) -> str:
    path = Path(checkpoint).expanduser()
    detail: dict[str, object] = {"path": str(path.resolve(strict=False))}
    if path.is_symlink() or not path.is_file():
        _check(checks, "checkpoint", "error", detail)
        errors.append(f"checkpoint is not a regular file: {path}")
        return ""
    try:
        size = path.stat().st_size
        detail["size_bytes"] = size
        if size <= 0:
            _check(checks, "checkpoint", "error", detail)
            errors.append(f"checkpoint is empty: {path}")
            return ""
        digest = _sha256_file(path)
    except OSError as exc:
        detail["error"] = str(exc)
        _check(checks, "checkpoint", "error", detail)
        errors.append(f"checkpoint cannot be read: {path}: {exc}")
        return ""
    detail["sha256"] = digest
    _check(checks, "checkpoint", "ok", detail)
    return digest


def _validate_pointmap(path: Path, expected: tuple[int, int], selected_ids: tuple[int, ...]) -> dict[str, object]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if "point_maps" not in archive or "valid_mask" not in archive:
            raise ValueError("point-map GT archive requires point_maps and valid_mask")
        point_maps = np.asarray(archive["point_maps"])
        valid_mask = np.asarray(archive["valid_mask"])
        candidates: list[tuple[str, np.ndarray]] = []
        if "frame_ids" in archive:
            candidates.append(("frame_ids", np.asarray(archive["frame_ids"])))
        for name in ("source_frame_ids.npy", "frame_ids.npy"):
            sibling = path.parent / name
            if sibling.is_file():
                try:
                    candidates.append((name, np.asarray(np.load(sibling, allow_pickle=False))))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"point-map GT frame IDs are invalid: {sibling}") from exc
        if not candidates:
            raise ValueError("point-map GT archive requires frame IDs")
        normalized_candidates: list[tuple[str, np.ndarray]] = []
        for source_name, raw_ids in candidates:
            source_ids = np.asarray(raw_ids)
            if source_ids.ndim != 1 or source_ids.dtype.kind not in "iu":
                raise ValueError("point-map GT frame IDs must be a one-dimensional integer array")
            normalized_candidates.append(
                (source_name, source_ids.astype(np.int64, copy=False))
            )
        frame_ids = normalized_candidates[0][1]
        if any(not np.array_equal(frame_ids, other) for _, other in normalized_candidates[1:]):
            raise ValueError("point-map GT frame ID sources disagree")
        if len(set(int(value) for value in frame_ids.tolist())) != len(frame_ids):
            raise ValueError("point-map GT frame IDs must be unique")
        if point_maps.ndim != 4 or point_maps.shape[-1] != 3:
            raise ValueError("point-map GT must have shape (N,H,W,3) for all frames")
        if tuple(point_maps.shape[1:3]) != expected:
            raise ValueError(
                f"point-map GT spatial shape {point_maps.shape[1:3]} does not match {expected}"
            )
        if valid_mask.ndim != 3 or valid_mask.shape != point_maps.shape[:3]:
            raise ValueError("point-map GT valid_mask shape does not match point-map GT frames")
        if valid_mask.dtype.kind != "b":
            raise ValueError("point-map GT valid_mask must be boolean")
        if point_maps.shape[0] != len(frame_ids):
            raise ValueError("point-map GT frame IDs do not match point-map GT frames")
        if np.any(valid_mask) and not np.isfinite(point_maps[valid_mask]).all():
            raise ValueError("point-map GT contains non-finite selected values")
        present = {int(value) for value in frame_ids.tolist()}
        missing = [value for value in selected_ids if value not in present]
        if missing:
            raise ValueError(f"point-map GT is missing selected frame IDs: {missing}")
        return {
            "path": str(path),
            "frame_count": int(point_maps.shape[0]),
            "frame_id_count": int(len(frame_ids)),
            "shape": [int(value) for value in point_maps.shape],
            "spatial_shape": [int(value) for value in point_maps.shape[1:3]],
        }


def _seed_digest(seed: object) -> str:
    payload = seed.to_payload()  # type: ignore[attr-defined]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _check_scenes_and_identities(
    loaded: Any,
    plan: Any,
    git: GitState,
    checkpoint_sha: str,
    checks: list[PreflightCheck],
    errors: list[str],
) -> tuple[str, ...]:
    try:
        from .matrix import build_identity_seed
        from .scenes import resolve_scene
        from .staging import preview_staging_manifest
    except Exception as exc:
        for selected in tuple(loaded.config.selected_scenes):
            scene_name = f"scene:{selected.scene_id}"
            _check(
                checks,
                scene_name,
                "blocked",
                {"blocked_by": "campaign scene/staging imports", "error": str(exc)},
            )
        errors.append(f"scene and staging checks are blocked by imports: {exc}")
        return ()

    config = loaded.config
    selected_scenes = tuple(config.selected_scenes)
    planned_by_scene: dict[str, list[Any]] = {}
    for planned in plan.runs:
        planned_by_scene.setdefault(planned.scene_id, []).append(planned)
    relative_dirs = [run.relative_run_dir for run in plan.runs]
    unsafe_dirs = [
        path for path in relative_dirs if path.is_absolute() or ".." in path.parts
    ]
    if len(relative_dirs) != len(set(relative_dirs)) or unsafe_dirs:
        errors.append("plan contains duplicate or unsafe relative run directories")
        _check(
            checks,
            "plan:run-directories",
            "error",
            {"unique": len(relative_dirs) == len(set(relative_dirs)), "safe": not unsafe_dirs},
        )
    else:
        _check(checks, "plan:run-directories", "ok", {"unique": True, "count": len(relative_dirs)})

    seed_hashes: list[str] = []
    seen_seed_hashes: set[str] = set()
    for selected in selected_scenes:
        scene_id = selected.scene_id
        scene_name = f"scene:{scene_id}"
        scene_config = config.scenes.get(scene_id)
        try:
            resolved = resolve_scene(config, selected)
            if len(resolved.source_images) != len(resolved.selection.source_frame_ids):
                raise ValueError("selected image and frame ID counts differ")
            if any(not path.is_file() for path in resolved.source_images):
                raise FileNotFoundError("one or more selected source images are missing")
            if resolved.evaluation_kind.value == "pointcloud":
                if resolved.prepared_gt_path is None:
                    raise ValueError(
                        "point-map GT is not prepared; raw dataset preparation is blocked in preflight"
                    )
                pointmap_detail = _validate_pointmap(
                    resolved.prepared_gt_path.resolve(strict=True),
                    resolved.expected_gt_shape or (392, 518),
                    resolved.selection.source_frame_ids,
                )
            else:
                pointmap_detail = None
            _manifest_payload, manifest_sha = preview_staging_manifest(resolved)
            detail: dict[str, object] = {
                "dataset": resolved.dataset.value,
                "scene": resolved.scene,
                "source_image_count": len(resolved.source_images),
                "selected_frame_ids": list(resolved.selection.source_frame_ids),
                "selection": {
                    "start": resolved.selection.start,
                    "stop": resolved.selection.stop,
                    "stride": resolved.selection.stride,
                },
                "staging_manifest_sha256": manifest_sha,
                "staging_preview_only": True,
            }
            if resolved.dataset.value == "kitti":
                image_text = str(resolved.source_images[0])
                detail["layout"] = "official" if "/dataset/sequences/" in image_text else "normalized"
                detail["poses_path"] = str(resolved.poses_path)
            if pointmap_detail is not None:
                detail["pointmap"] = pointmap_detail
            _check(checks, scene_name, "ok", detail)
        except Exception as exc:
            dataset = getattr(scene_config, "dataset", None)
            if getattr(dataset, "value", dataset) == "kitti":
                message = (
                    f"KITTI scene {scene_id} is unavailable or invalid: {exc}; "
                    "official KITTI registration is required for the declared research purpose; "
                    "preflight never downloads datasets"
                )
            else:
                message = f"scene {scene_id} data is unavailable or invalid: {exc}"
            _check(checks, scene_name, "error", {"scene_id": scene_id, "error": str(exc)})
            errors.append(message)
            _check(
                checks,
                f"staging-preview:{scene_id}",
                "blocked",
                {"blocked_by": scene_name, "staging_preview_only": True},
            )
            continue

        if not checkpoint_sha:
            _check(
                checks,
                f"identity:{scene_id}",
                "blocked",
                {"blocked_by": "checkpoint", "seed_count": 0},
            )
            continue
        identity_error = not planned_by_scene.get(scene_id)
        if identity_error:
            errors.append(f"scene {scene_id} has no planned identity seeds")
        for planned in planned_by_scene.get(scene_id, []):
            try:
                seed = build_identity_seed(
                    loaded=loaded,
                    planned=planned,
                    frame_start=resolved.selection.start,
                    frame_stop=resolved.selection.stop,
                    frame_stride=resolved.selection.stride,
                    staged_manifest_sha256=manifest_sha,
                    source_commit=git.commit,
                    source_dirty=git.dirty,
                    checkpoint_sha256=checkpoint_sha,
                )
                digest = _seed_digest(seed)
            except Exception as exc:
                errors.append(f"identity seed for {scene_id}/{planned.run_id} is invalid: {exc}")
                identity_error = True
                continue
            if digest in seen_seed_hashes:
                errors.append(f"identity seed hash is duplicated: {digest}")
                identity_error = True
            seen_seed_hashes.add(digest)
            seed_hashes.append(digest)
        _check(
            checks,
            f"identity:{scene_id}",
            "error" if identity_error else "ok",
            {
                "seed_count": len(planned_by_scene.get(scene_id, [])),
                "unique": not identity_error,
            },
        )
    return tuple(seed_hashes)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"no existing parent for campaign root: {path}")
    return candidate


def _check_storage(
    loaded: Any,
    dependencies: PreflightDependencies,
    checks: list[PreflightCheck],
    warnings: list[str],
    errors: list[str],
) -> int:
    root = Path(loaded.config.campaign_root).expanduser()
    detail: dict[str, object] = {"campaign_root": str(root.resolve(strict=False))}
    if root.is_symlink():
        _check(checks, "storage", "error", detail)
        errors.append(f"campaign root is a symlink and is rejected: {root}")
        return 0
    if root.exists() and not root.is_dir():
        _check(checks, "storage", "error", detail)
        errors.append(f"campaign root is not a directory: {root}")
        return 0
    try:
        existing = root if root.exists() else _nearest_existing_parent(root.parent)
        usage = dependencies.disk_usage(existing)
        detail.update(
            {
                "nearest_existing_parent": str(existing),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )
    except Exception as exc:
        _check(checks, "storage", "error", {**detail, "error": str(exc)})
        errors.append(f"campaign storage could not be inspected: {exc}")
        return 0
    mode = existing.stat().st_mode
    writable = bool(mode & stat.S_IWUSR or mode & stat.S_IWGRP or mode & stat.S_IWOTH)
    writable = writable and os.access(existing, os.W_OK)
    if not writable:
        errors.append(f"campaign root parent is not writable: {existing}")
    low_space = usage.free < _MIN_FREE_BYTES
    if low_space:
        warnings.append(
            f"free disk space {usage.free / 1024**3:.1f} GiB is below required 20.0 GiB"
        )
    if not writable:
        _check(checks, "storage", "error", detail)
    elif low_space:
        _check(checks, "storage", "warning", detail)
    else:
        _check(checks, "storage", "ok", detail)
    resolved_root = root.resolve(strict=False)
    if _inside(Path("/root/autodl-fs"), resolved_root):
        warnings.append(
            f"campaign root is under /root/autodl-fs; prefer /root/autodl-tmp: {resolved_root}"
        )
    return usage.free


def _check_cuda(
    loaded: Any,
    dependencies: PreflightDependencies,
    allow_no_gpu: bool,
    checks: list[PreflightCheck],
    warnings: list[str],
    errors: list[str],
) -> None:
    selected = int(loaded.config.runtime.gpu)
    try:
        state = dependencies.cuda_state(selected)
    except Exception as exc:
        state = CudaState(False, 0, None, None, None)
        detail: dict[str, object] = {"selected_device": selected, "error": str(exc), "gpu_ready": False}
        if allow_no_gpu:
            _check(checks, "cuda", "warning", detail)
            warnings.append(f"CUDA is unavailable; no-GPU preflight allowed ({exc})")
        else:
            _check(checks, "cuda", "error", detail)
            errors.append(f"CUDA is unavailable: {exc}")
        return
    in_range = state.available and state.device_count > selected >= 0 and state.selected_device == selected
    detail = {
        "available": state.available,
        "device_count": state.device_count,
        "selected_device": selected,
        "device_name": state.device_name,
        "bfloat16_supported": state.bfloat16_supported,
        "gpu_ready": bool(in_range),
    }
    if not in_range:
        message = f"CUDA selected device {selected} is unavailable or out of range"
        if allow_no_gpu:
            _check(checks, "cuda", "warning", detail)
            warnings.append(f"CUDA unavailable/out of range; no-GPU preflight allowed ({message})")
        else:
            _check(checks, "cuda", "error", detail)
            errors.append(message)
        return
    if state.bfloat16_supported is not True:
        _check(checks, "cuda", "error", detail)
        errors.append("CUDA selected device does not support bfloat16")
        return
    _check(checks, "cuda", "ok", detail)


def preflight_campaign(
    loaded: Any,
    plan: Any,
    *,
    allow_no_gpu: bool,
    dependencies: PreflightDependencies | None = None,
) -> PreflightReport:
    """Collect independent environment, input, and identity checks."""

    if type(allow_no_gpu) is not bool:
        raise ValueError("allow_no_gpu must be a boolean")
    dependencies = dependencies or default_preflight_dependencies()
    checks: list[PreflightCheck] = []
    warnings: list[str] = []
    errors: list[str] = []
    synthetic_only = bool(getattr(plan, "runs", ())) and all(
        getattr(item.dataset, "value", item.dataset) == "synthetic"
        for item in plan.runs
    )
    if synthetic_only:
        # Synthetic checks intentionally avoid model imports, checkpoint/data
        # existence, and CUDA.  Keep the storage and source provenance checks
        # so the same campaign-owned output boundary is exercised.
        version = dependencies.python_version
        python_detail = {
            "version": ".".join(str(item) for item in version),
            "required": "3.11 for real campaigns",
            "synthetic": True,
        }
        if version[:2] != (3, 11):
            _check(checks, "python", "warning", python_detail)
            warnings.append("synthetic campaign does not import the PI3 runtime")
        else:
            _check(checks, "python", "ok", python_detail)
        repository = Path(loaded.config.repository_root)
        git = _check_git(repository, dependencies, checks, warnings, errors)
        free = _check_storage(loaded, dependencies, checks, warnings, errors)
        from .matrix import build_identity_seed
        from .scenes import resolve_scene

        planned_by_scene: dict[str, list[Any]] = {}
        for planned in plan.runs:
            planned_by_scene.setdefault(planned.scene_id, []).append(planned)
        selected = {item.scene_id: item for item in loaded.config.selected_scenes}
        sentinel = hashlib.sha256(
            b"LASER-window-reference-synthetic-checkpoint-v1"
        ).hexdigest()
        seed_hashes: list[str] = []
        for scene_id, planned_items in planned_by_scene.items():
            resolved = resolve_scene(loaded.config, selected[scene_id])
            payload = {
                "schema_version": 2,
                "dataset": "synthetic",
                "scene_id": resolved.scene_id,
                "scene": resolved.scene,
                "slice_id": resolved.slice_id,
                "source_frame_ids": list(resolved.selection.source_frame_ids),
                "selection": {
                    "start": resolved.selection.start,
                    "stop": resolved.selection.stop,
                    "stride": resolved.selection.stride,
                },
                "synthetic_fixture": "window-reference-v1",
            }
            manifest_digest = hashlib.sha256(
                (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            ).hexdigest()
            _check(
                checks,
                f"scene:{scene_id}",
                "ok",
                {
                    "dataset": "synthetic",
                    "scene": resolved.scene,
                    "source_image_count": len(resolved.source_images),
                    "staging_preview_only": True,
                    "synthetic": True,
                },
            )
            for planned in planned_items:
                seed = build_identity_seed(
                    loaded=loaded,
                    planned=planned,
                    frame_start=resolved.selection.start,
                    frame_stop=resolved.selection.stop,
                    frame_stride=resolved.selection.stride,
                    staged_manifest_sha256=manifest_digest,
                    source_commit=git.commit,
                    source_dirty=git.dirty,
                    checkpoint_sha256=sentinel,
                )
                seed_payload = json.dumps(
                    seed.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                seed_hashes.append(hashlib.sha256(seed_payload.encode()).hexdigest())
            _check(
                checks,
                f"identity:{scene_id}",
                "ok",
                {"seed_count": len(planned_items), "unique": True, "synthetic": True},
            )
        _check(
            checks,
            "cuda",
            "ok",
            {"synthetic": True, "gpu_ready": False, "not_applicable": True},
        )
        status = "error" if errors else ("ok_with_warnings" if warnings else "ok")
        return PreflightReport(
            1,
            status,
            allow_no_gpu,
            tuple(checks),
            tuple(warnings),
            tuple(errors),
            git,
            sentinel,
            free,
            tuple(seed_hashes),
        )
    _check_python(dependencies, checks, errors)
    _check_imports(dependencies, checks, errors)
    repository = Path(loaded.config.repository_root)
    git = _check_git(repository, dependencies, checks, warnings, errors)
    checkpoint_sha = _check_checkpoint(loaded.config.storage.checkpoint, checks, errors)
    seed_hashes = _check_scenes_and_identities(
        loaded, plan, git, checkpoint_sha, checks, errors
    )
    free = _check_storage(loaded, dependencies, checks, warnings, errors)
    _check_cuda(loaded, dependencies, allow_no_gpu, checks, warnings, errors)
    status = "error" if errors else ("ok_with_warnings" if warnings else "ok")
    return PreflightReport(
        1,
        status,
        allow_no_gpu,
        tuple(checks),
        tuple(warnings),
        tuple(errors),
        git,
        checkpoint_sha,
        free,
        tuple(seed_hashes),
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    serialized = json.dumps(_json_value(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_preflight_report(report: PreflightReport, campaign_root: str | Path) -> Path:
    """Atomically publish exactly one preflight report under campaign root."""

    if not isinstance(report, PreflightReport):
        raise ValueError("preflight report is invalid")
    root = Path(campaign_root)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"campaign root is not a writable directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    target = root / "preflight.json"
    return _atomic_json(target, report.to_payload())


__all__ = [
    "BootstrapAction",
    "CudaState",
    "DiskUsage",
    "GitState",
    "PreflightCheck",
    "PreflightDependencies",
    "PreflightReport",
    "build_bootstrap_actions",
    "default_preflight_dependencies",
    "preflight_campaign",
    "run_bootstrap",
    "write_preflight_report",
]
