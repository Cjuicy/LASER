from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch

from pipeline.config import PredictionCacheMode

from .fingerprint import PredictionFingerprint
from .types import (
    PREDICTION_CACHE_SCHEMA_VERSION,
    OrdinaryWindowArtifact,
    SequenceArtifact,
    WindowSpec,
)


class PredictionCacheMissError(RuntimeError):
    pass


class PredictionCacheCorruptError(RuntimeError):
    pass


@dataclass
class PredictionStoreStats:
    ordinary_hits: int = 0
    ordinary_misses: int = 0
    corrupt_count: int = 0
    read_ms: float = 0.0
    write_ms: float = 0.0
    saved_window_count: int = 0
    stored_bytes: int = 0
    events: list[dict[str, object]] = field(default_factory=list)


def _update_digest(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _update_tensor_digest(
    digest,
    tensor: torch.Tensor,
) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    _update_digest(digest, str(contiguous.dtype).encode("ascii"))
    _update_digest(
        digest,
        json.dumps(list(contiguous.shape)).encode("ascii"),
    )
    _update_digest(
        digest,
        contiguous.view(torch.uint8).numpy().tobytes(),
    )


def _sequence_artifact_digest(
    key: str,
    artifact: SequenceArtifact,
) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, b"laser-sequence-artifact-v1")
    _update_digest(digest, key.encode("ascii"))
    _update_tensor_digest(digest, artifact.reference_intrinsic)
    return digest.hexdigest()


def _window_artifact_digest(
    key: str,
    artifact: OrdinaryWindowArtifact,
) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, b"laser-window-artifact-v1")
    _update_digest(digest, key.encode("ascii"))
    _update_digest(
        digest,
        json.dumps(
            artifact.spec.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii"),
    )
    for tensor in (
        artifact.depth,
        artifact.confidence,
        artifact.camera_poses,
    ):
        _update_tensor_digest(digest, tensor)
    return digest.hexdigest()


class OrdinaryPredictionStore:
    def __init__(
        self,
        *,
        root: str | Path,
        fingerprint: PredictionFingerprint,
        mode: PredictionCacheMode,
        expected_specs: Sequence[WindowSpec],
    ) -> None:
        if not isinstance(fingerprint, PredictionFingerprint):
            raise ValueError(
                "prediction store fingerprint must be a "
                "PredictionFingerprint"
            )
        if not isinstance(mode, PredictionCacheMode):
            raise ValueError(
                "prediction store mode must be a PredictionCacheMode"
            )
        specs = tuple(expected_specs)
        if (
            not specs
            or any(not isinstance(spec, WindowSpec) for spec in specs)
            or tuple(spec.index for spec in specs)
            != tuple(range(len(specs)))
        ):
            raise ValueError(
                "prediction store requires ordered canonical WindowSpecs"
            )

        self.root = Path(root)
        self.fingerprint = fingerprint
        self.mode = mode
        self.expected_specs = specs
        image_shape = fingerprint.canonical_payload.get("image_shape")
        if (
            not isinstance(image_shape, (tuple, list))
            or len(image_shape) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in image_shape
            )
            or image_shape[0] != specs[-1].frame_end
            or image_shape[1] != 3
        ):
            raise ValueError(
                "prediction fingerprint image_shape is invalid"
            )
        fingerprint_specs = fingerprint.canonical_payload.get(
            "window_specs"
        )
        expected_specs_payload = [
            spec.to_payload() for spec in specs
        ]
        if fingerprint_specs != expected_specs_payload:
            raise ValueError(
                "prediction store WindowSpecs do not match fingerprint"
            )
        self.expected_spatial_shape = (
            image_shape[2],
            image_shape[3],
        )
        self.entry_path = (
            self.root
            / f"v{PREDICTION_CACHE_SCHEMA_VERSION}"
            / fingerprint.key
        )
        self.stats = PredictionStoreStats()
        self._refresh_prepared = False
        self._read_validation_cached = False
        self._lock_depth = 0
        self._lock_stream = None

    @property
    def manifest_path(self) -> Path:
        return self.entry_path / "manifest.json"

    @property
    def sequence_path(self) -> Path:
        return self.entry_path / "sequence.json"

    @property
    def complete_path(self) -> Path:
        return self.entry_path / "complete.json"

    @property
    def windows_path(self) -> Path:
        return self.entry_path / "windows"

    @property
    def invalid_path(self) -> Path:
        return self.entry_path / "invalid"

    @property
    def lock_path(self) -> Path:
        return self.entry_path / "locks" / "entry.lock"

    def _window_path(self, spec: WindowSpec) -> Path:
        return self.windows_path / f"{spec.index:06d}.pt"

    def _ensure_layout(self) -> None:
        self.windows_path.mkdir(parents=True, exist_ok=True)
        self.invalid_path.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)

    def _invalidate_read_validation(self) -> None:
        self._read_validation_cached = False

    @staticmethod
    def _window_description(spec: WindowSpec) -> str:
        return (
            f"window {spec.index:06d} "
            f"[{spec.frame_start},{spec.frame_end})"
        )

    def contextualize_read_error(
        self,
        spec: WindowSpec,
        error: PredictionCacheMissError | PredictionCacheCorruptError,
    ) -> PredictionCacheMissError | PredictionCacheCorruptError:
        description = self._window_description(spec)
        self.stats.events.append(
            {
                "event": "readonly_error",
                "window_index": spec.index,
                "frame_start": spec.frame_start,
                "frame_end": spec.frame_end,
                "reason": str(error),
            }
        )
        return type(error)(f"{description}: {error}")

    @contextmanager
    def entry_lock(
        self,
        *,
        blocking: bool = True,
    ) -> Iterator[None]:
        if self.mode is PredictionCacheMode.OFF:
            yield
            return
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        if self.mode is PredictionCacheMode.READONLY:
            if not self.lock_path.is_file():
                raise PredictionCacheMissError(
                    "readonly prediction cache entry is missing"
                )
        else:
            self._ensure_layout()

        stream = self.lock_path.open("a+b")
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(stream.fileno(), flags)
            self._lock_stream = stream
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
                self._lock_stream = None
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def _manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
            "key": self.fingerprint.key,
            "checkpoint_sha256": self.fingerprint.checkpoint_sha256,
            "image_manifest_sha256": (
                self.fingerprint.image_manifest_sha256
            ),
            "runtime_source_sha256": (
                self.fingerprint.runtime_source_sha256
            ),
            "fingerprint": dict(self.fingerprint.canonical_payload),
            "window_specs": [
                spec.to_payload() for spec in self.expected_specs
            ],
        }

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(text)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def _atomic_write_json(
        cls,
        path: Path,
        payload,
        *,
        validate_payload: Callable[[object], object] | None = None,
    ) -> None:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        if validate_payload is not None:
            validate_payload(json.loads(text))
        cls._atomic_write_text(path, text + "\n")

    @staticmethod
    def _atomic_torch_save(
        path: Path,
        payload,
        *,
        validate_payload: Callable[[object], object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            torch.save(payload, temporary_path)
            if validate_payload is not None:
                validate_payload(
                    torch.load(
                        temporary_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                )
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _read_json(path: Path):
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _snapshot_name(self, reason: str) -> Path:
        safe_reason = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "-"
            for character in reason
        ).strip("-")
        return self.invalid_path / (
            f"{time.time_ns()}-{safe_reason or 'invalid'}"
        )

    def _quarantine_paths(
        self,
        paths: Sequence[Path],
        *,
        reason: str,
    ) -> tuple[Path, ...]:
        self._ensure_layout()
        snapshot = self._snapshot_name(reason)
        snapshot.mkdir(parents=True, exist_ok=False)
        destinations = []
        for path in paths:
            if not path.exists():
                continue
            destination = snapshot / path.name
            original_path = str(path.resolve())
            path.replace(destination)
            destinations.append(destination)
            self.stats.events.append(
                {
                    "event": "quarantine",
                    "reason": reason,
                    "original_path": original_path,
                    "quarantine_path": str(destination.resolve()),
                }
            )
        self.stats.corrupt_count += 1
        self._invalidate_read_validation()
        return tuple(destinations)

    def _snapshot_entry(self, *, reason: str) -> None:
        self._ensure_layout()
        snapshot = self._snapshot_name(reason)
        snapshot.mkdir(parents=True, exist_ok=False)
        for path in (
            self.manifest_path,
            self.sequence_path,
            self.complete_path,
        ):
            if path.exists():
                destination = snapshot / path.name
                original_path = str(path.resolve())
                path.replace(destination)
                self.stats.events.append(
                    {
                        "event": "quarantine",
                        "reason": reason,
                        "original_path": original_path,
                        "quarantine_path": str(destination.resolve()),
                    }
                )
        window_files = tuple(self.windows_path.glob("*.pt"))
        if window_files:
            destination = snapshot / "windows"
            destination.mkdir()
            for path in window_files:
                target = destination / path.name
                original_path = str(path.resolve())
                path.replace(target)
                self.stats.events.append(
                    {
                        "event": "quarantine",
                        "reason": reason,
                        "original_path": original_path,
                        "quarantine_path": str(target.resolve()),
                    }
                )
        self._invalidate_read_validation()

    def _write_manifest(self) -> None:
        self._ensure_layout()
        self._invalidate_read_validation()
        self._atomic_write_json(
            self.manifest_path,
            self._manifest_payload(),
        )

    def _manifest_is_valid(self) -> bool:
        if not self.manifest_path.is_file():
            return False
        payload = self._read_json(self.manifest_path)
        return payload == self._manifest_payload()

    def _prepare_refresh(self) -> None:
        if (
            self.mode is not PredictionCacheMode.REFRESH
            or self._refresh_prepared
        ):
            return
        self._ensure_layout()
        if any(
            (
                self.manifest_path.exists(),
                self.sequence_path.exists(),
                self.complete_path.exists(),
                any(self.windows_path.glob("*.pt")),
            )
        ):
            self._snapshot_entry(reason="refresh")
        self._write_manifest()
        self._refresh_prepared = True

    def _ensure_manifest_for_write(self) -> None:
        self._prepare_refresh()
        if self.manifest_path.is_file():
            try:
                valid = self._manifest_is_valid()
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                valid = False
            if valid:
                return
            self._snapshot_entry(reason="manifest-corrupt")
            self.stats.corrupt_count += 1
        self._write_manifest()

    def _validate_manifest_for_read(self) -> bool:
        self._prepare_refresh()
        if self._read_validation_cached:
            return True
        if not self.manifest_path.is_file():
            if self.mode is PredictionCacheMode.READONLY:
                raise PredictionCacheMissError(
                    "readonly prediction cache manifest is missing"
                )
            return False
        try:
            valid = self._manifest_is_valid()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if self.mode is PredictionCacheMode.READONLY:
                raise PredictionCacheCorruptError(
                    f"prediction cache manifest is corrupt: {exc}"
                ) from exc
            self._snapshot_entry(reason="manifest-corrupt")
            self.stats.corrupt_count += 1
            self._write_manifest()
            return False
        if not valid:
            if self.mode is PredictionCacheMode.READONLY:
                raise PredictionCacheCorruptError(
                    "prediction cache manifest does not match its key"
                )
            self._snapshot_entry(reason="manifest-mismatch")
            self.stats.corrupt_count += 1
            self._write_manifest()
            return False
        self._validate_completion_for_read()
        self._update_stored_bytes()
        self._read_validation_cached = True
        return True

    def _validate_completion_for_read(self) -> None:
        if not self.complete_path.is_file():
            return
        try:
            payload = self._read_json(self.complete_path)
            if payload != {
                "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
                "key": self.fingerprint.key,
                "window_count": len(self.expected_specs),
            }:
                raise ValueError("completion payload is invalid")
            if not self.sequence_path.is_file() or any(
                not self._window_path(spec).is_file()
                for spec in self.expected_specs
            ):
                raise ValueError(
                    "completion marker declares missing artifacts"
                )
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            if self.mode is PredictionCacheMode.READONLY:
                raise PredictionCacheCorruptError(
                    "prediction cache completion marker is corrupt"
                ) from exc
            self._quarantine_paths(
                (self.complete_path,),
                reason="completion-corrupt",
            )

    def _validate_window_geometry(
        self,
        artifact: OrdinaryWindowArtifact,
    ) -> None:
        if tuple(artifact.depth.shape[1:]) != (
            self.expected_spatial_shape
        ):
            raise ValueError(
                "ordinary window artifact spatial geometry mismatch"
            )

    def _sequence_from_payload(self, payload) -> SequenceArtifact:
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != PREDICTION_CACHE_SCHEMA_VERSION
        ):
            raise ValueError("sequence schema_version is invalid")
        if payload.get("key") != self.fingerprint.key:
            raise ValueError("sequence prediction key mismatch")
        intrinsic = torch.tensor(
            payload["reference_intrinsic"],
            dtype=torch.float32,
        )
        artifact = SequenceArtifact(intrinsic)
        expected_digest = _sequence_artifact_digest(
            self.fingerprint.key,
            artifact,
        )
        if payload.get("artifact_sha256") != expected_digest:
            raise ValueError("sequence artifact digest mismatch")
        return artifact

    def _window_from_payload(
        self,
        payload,
        spec: WindowSpec,
    ) -> OrdinaryWindowArtifact:
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != PREDICTION_CACHE_SCHEMA_VERSION
        ):
            raise ValueError("window schema_version is invalid")
        if payload.get("key") != self.fingerprint.key:
            raise ValueError("window prediction key mismatch")
        artifact = OrdinaryWindowArtifact.from_payload(
            payload["artifact"]
        )
        if artifact.spec != spec:
            raise ValueError(
                "stored WindowSpec does not match request"
            )
        self._validate_window_geometry(artifact)
        expected_digest = _window_artifact_digest(
            self.fingerprint.key,
            artifact,
        )
        if payload.get("artifact_sha256") != expected_digest:
            raise ValueError("window artifact digest mismatch")
        return artifact

    def read_sequence(self) -> SequenceArtifact | None:
        if self.mode is PredictionCacheMode.OFF:
            return None
        started = time.perf_counter()
        try:
            with self.entry_lock():
                if not self._validate_manifest_for_read():
                    return None
                if not self.sequence_path.is_file():
                    if self.mode is PredictionCacheMode.READONLY:
                        raise PredictionCacheMissError(
                            "readonly prediction cache sequence is missing"
                        )
                    return None
                try:
                    payload = self._read_json(self.sequence_path)
                    return self._sequence_from_payload(payload)
                except (
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    if self.mode is PredictionCacheMode.READONLY:
                        raise PredictionCacheCorruptError(
                            "prediction cache sequence is corrupt"
                        ) from exc
                    self._quarantine_paths(
                        (self.sequence_path,),
                        reason="sequence-corrupt",
                    )
                    return None
        finally:
            self.stats.read_ms += (
                time.perf_counter() - started
            ) * 1000

    def write_sequence(self, artifact: SequenceArtifact) -> None:
        if self.mode is PredictionCacheMode.OFF:
            return
        if self.mode is PredictionCacheMode.READONLY:
            raise PredictionCacheMissError(
                "readonly prediction cache cannot write sequence"
            )
        if not isinstance(artifact, SequenceArtifact):
            raise ValueError("sequence artifact is invalid")
        stored = SequenceArtifact(
            artifact.reference_intrinsic.detach()
            .cpu()
            .to(torch.float32)
            .clone()
        )
        payload = {
            "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
            "key": self.fingerprint.key,
            "artifact_sha256": _sequence_artifact_digest(
                self.fingerprint.key,
                stored,
            ),
            "reference_intrinsic": stored.reference_intrinsic.tolist(),
        }
        started = time.perf_counter()
        try:
            with self.entry_lock():
                self._ensure_manifest_for_write()
                self._atomic_write_json(
                    self.sequence_path,
                    payload,
                    validate_payload=self._sequence_from_payload,
                )
                self._invalidate_read_validation()
        finally:
            self.stats.write_ms += (
                time.perf_counter() - started
            ) * 1000

    def read_window(
        self,
        spec: WindowSpec,
    ) -> OrdinaryWindowArtifact | None:
        if spec not in self.expected_specs:
            raise ValueError("requested WindowSpec is not in this store")
        if self.mode is PredictionCacheMode.OFF:
            self.stats.ordinary_misses += 1
            return None
        started = time.perf_counter()
        try:
            try:
                with self.entry_lock():
                    if not self._validate_manifest_for_read():
                        self.stats.ordinary_misses += 1
                        return None
                    path = self._window_path(spec)
                    if not path.is_file():
                        if self.mode is PredictionCacheMode.READONLY:
                            raise PredictionCacheMissError(
                                "readonly prediction cache window is missing"
                            )
                        self.stats.ordinary_misses += 1
                        return None
                    try:
                        payload = torch.load(
                            path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        artifact = self._window_from_payload(
                            payload,
                            spec,
                        )
                    except (
                        EOFError,
                        KeyError,
                        OSError,
                        pickle.UnpicklingError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        if self.mode is PredictionCacheMode.READONLY:
                            raise PredictionCacheCorruptError(
                                "readonly prediction cache window is corrupt"
                            ) from exc
                        self._quarantine_paths(
                            (path,),
                            reason=f"window-{spec.index:06d}-corrupt",
                        )
                        self.stats.ordinary_misses += 1
                        return None
                    self.stats.ordinary_hits += 1
                    return artifact
            except (
                PredictionCacheMissError,
                PredictionCacheCorruptError,
            ) as exc:
                if self.mode is PredictionCacheMode.READONLY:
                    raise self.contextualize_read_error(spec, exc) from exc
                raise
        finally:
            self.stats.read_ms += (
                time.perf_counter() - started
            ) * 1000

    def write_window(self, artifact: OrdinaryWindowArtifact) -> None:
        if self.mode is PredictionCacheMode.OFF:
            return
        if self.mode is PredictionCacheMode.READONLY:
            raise PredictionCacheMissError(
                "readonly prediction cache cannot write a window"
            )
        if (
            not isinstance(artifact, OrdinaryWindowArtifact)
            or artifact.spec not in self.expected_specs
        ):
            raise ValueError(
                "ordinary window artifact does not belong to this store"
            )
        stored = OrdinaryWindowArtifact(
            spec=artifact.spec,
            depth=artifact.depth.detach().cpu().clone(),
            confidence=artifact.confidence.detach().cpu().clone(),
            camera_poses=artifact.camera_poses.detach().cpu().clone(),
        )
        self._validate_window_geometry(stored)
        payload = {
            "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
            "key": self.fingerprint.key,
            "artifact_sha256": _window_artifact_digest(
                self.fingerprint.key,
                stored,
            ),
            "artifact": stored.to_payload(),
        }
        started = time.perf_counter()
        try:
            with self.entry_lock():
                self._ensure_manifest_for_write()
                self._atomic_torch_save(
                    self._window_path(stored.spec),
                    payload,
                    validate_payload=lambda candidate: (
                        self._window_from_payload(
                            candidate,
                            stored.spec,
                        )
                    ),
                )
                self.stats.saved_window_count += 1
                self._invalidate_read_validation()
        finally:
            self.stats.write_ms += (
                time.perf_counter() - started
            ) * 1000

    def finalize(self) -> None:
        if self.mode is PredictionCacheMode.OFF:
            raise PredictionCacheMissError(
                "off prediction cache cannot finalize"
            )
        if self.mode is PredictionCacheMode.READONLY:
            if not self.complete_path.is_file():
                raise PredictionCacheMissError(
                    "readonly prediction cache is incomplete"
                )
            return
        with self.entry_lock():
            self._ensure_manifest_for_write()
            missing = []
            if not self.sequence_path.is_file():
                missing.append("sequence")
            missing.extend(
                f"window {spec.index}"
                for spec in self.expected_specs
                if not self._window_path(spec).is_file()
            )
            if missing:
                raise PredictionCacheMissError(
                    "cannot finalize incomplete prediction cache: "
                    + ", ".join(missing)
                )
            self._atomic_write_json(
                self.complete_path,
                {
                    "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
                    "key": self.fingerprint.key,
                    "window_count": len(self.expected_specs),
                },
            )
            self._invalidate_read_validation()
            self._update_stored_bytes()

    def _update_stored_bytes(self) -> None:
        if not self.entry_path.exists():
            self.stats.stored_bytes = 0
            return
        self.stats.stored_bytes = sum(
            path.stat().st_size
            for path in self.entry_path.rglob("*")
            if path.is_file() and "invalid" not in path.parts
        )
