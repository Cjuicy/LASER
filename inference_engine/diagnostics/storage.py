"""Bounded, resumable, run-owned diagnostic storage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import fcntl
from pathlib import Path
import shutil
import socket
import tempfile
import threading
from typing import Any, Iterator

import numpy as np


GIB = 1024 ** 3
OWNER_FILE = ".laser-diagnostic-owner.json"
_JSONL_LOCK = threading.Lock()


class StorageLimitExceeded(RuntimeError):
    pass


class FreeSpaceReserveExceeded(RuntimeError):
    pass


class RunLockError(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetState:
    used_bytes: int
    projected_bytes: int
    free_bytes: int
    max_bytes: int
    warn_bytes: int
    min_free_bytes: int
    level: str


def directory_size(path: str | os.PathLike[str]) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for current, _, files in os.walk(root):
        for filename in files:
            try:
                total += (Path(current) / filename).stat().st_size
            except FileNotFoundError:
                continue
    return total


def atomic_write_json(path: str | os.PathLike[str], data: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def append_jsonl(path: str | os.PathLike[str], record: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with _JSONL_LOCK:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return destination


def atomic_write_npz(path: str | os.PathLike[str], **arrays: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


class RunLock:
    def __init__(self, path: str | os.PathLike[str], run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }
        # The persistent guard serializes stale inspection and replacement.
        # It is intentionally never unlinked: unlinking a flock inode would
        # reintroduce a race between old and new guard-file descriptors.
        guard_path = self.path.with_name(self.path.name + ".guard")
        guard_descriptor = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(guard_descriptor, fcntl.LOCK_EX)
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError as exc:
                # A killed cloud job cannot execute ``__exit__``. Recover only
                # this exact run on this host whose PID is provably dead.
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(existing["pid"])
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    existing = {}
                    pid = -1
                recoverable = (
                    existing.get("run_id") == self.run_id
                    and existing.get("hostname") == socket.gethostname()
                    and not _pid_is_alive(pid)
                )
                if not recoverable:
                    raise RunLockError(f"Diagnostic run is already locked: {self.path}") from exc
                self.path.unlink(missing_ok=True)
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(guard_descriptor, fcntl.LOCK_UN)
            os.close(guard_descriptor)
        self._owned = True

    def release(self) -> None:
        if not self._owned:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        if payload.get("run_id") == self.run_id and payload.get("pid") == os.getpid():
            self.path.unlink(missing_ok=True)
        self._owned = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_owned_directory(path: str | os.PathLike[str], run_id: str) -> None:
    directory = Path(path)
    marker = directory / OWNER_FILE
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PermissionError(f"Cannot verify ownership for {directory}") from exc
    if payload.get("run_id") != run_id:
        raise PermissionError(f"Cannot verify ownership for {directory}")
    shutil.rmtree(directory)


@contextmanager
def owned_temp_directory(
    root: str | os.PathLike[str],
    run_id: str,
    *,
    cleanup: bool = True,
) -> Iterator[Path]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"laser-diag-{run_id}-", dir=root_path))
    atomic_write_json(directory / OWNER_FILE, {"run_id": run_id})
    try:
        yield directory
    finally:
        if cleanup and directory.exists():
            cleanup_owned_directory(directory, run_id)


class StorageBudget:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_gib: float = 50,
        warn_gib: float = 40,
        min_free_gib: float = 10,
        max_bytes: int | None = None,
        warn_bytes: int | None = None,
        min_free_bytes: int | None = None,
    ):
        self.root = Path(root)
        self.max_bytes = int(max_gib * GIB) if max_bytes is None else int(max_bytes)
        self.warn_bytes = int(warn_gib * GIB) if warn_bytes is None else int(warn_bytes)
        self.min_free_bytes = (
            int(min_free_gib * GIB) if min_free_bytes is None else int(min_free_bytes)
        )
        if self.max_bytes <= 0:
            raise ValueError("max threshold must be positive")
        if not 0 <= self.warn_bytes <= self.max_bytes:
            raise ValueError("warn threshold must be between zero and max")
        if self.min_free_bytes < 0:
            raise ValueError("min_free threshold must be non-negative")

    def state(
        self,
        *,
        used_bytes: int | None = None,
        free_bytes: int | None = None,
        estimated_bytes: int = 0,
    ) -> BudgetState:
        used = directory_size(self.root) if used_bytes is None else int(used_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free if free_bytes is None else int(free_bytes)
        projected = used + int(estimated_bytes)
        level = "warning" if projected >= self.warn_bytes else "ok"
        return BudgetState(
            used_bytes=used,
            projected_bytes=projected,
            free_bytes=free,
            max_bytes=self.max_bytes,
            warn_bytes=self.warn_bytes,
            min_free_bytes=self.min_free_bytes,
            level=level,
        )

    def enforce(
        self,
        *,
        used_bytes: int | None = None,
        free_bytes: int | None = None,
        estimated_bytes: int = 0,
    ) -> BudgetState:
        state = self.state(
            used_bytes=used_bytes,
            free_bytes=free_bytes,
            estimated_bytes=estimated_bytes,
        )
        if state.projected_bytes > state.max_bytes:
            raise StorageLimitExceeded(
                f"Projected temporary usage {state.projected_bytes} exceeds "
                f"hard limit {state.max_bytes}."
            )
        if state.free_bytes - estimated_bytes < state.min_free_bytes:
            raise FreeSpaceReserveExceeded(
                f"Write would violate free-space reserve {state.min_free_bytes}."
            )
        return state
