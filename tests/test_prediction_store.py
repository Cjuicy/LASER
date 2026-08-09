from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from inference_engine.prediction_cache.fingerprint import (
    PredictionFingerprint,
)
from inference_engine.prediction_cache.store import (
    OrdinaryPredictionStore,
    PredictionCacheCorruptError,
    PredictionCacheMissError,
)
from inference_engine.prediction_cache.types import (
    OrdinaryWindowArtifact,
    SequenceArtifact,
    WindowSpec,
)
from pipeline.config import PredictionCacheMode


def _fingerprint(key: str = "a" * 64) -> PredictionFingerprint:
    return PredictionFingerprint(
        key=key,
        checkpoint_sha256="b" * 64,
        image_manifest_sha256="c" * 64,
        runtime_source_sha256="d" * 64,
        canonical_payload={
            "model_name": "pi3",
            "dtype": "float32",
            "image_shape": [3, 3, 2, 3],
            "window_specs": [spec.to_payload() for spec in SPECS],
        },
    )


SPECS = (WindowSpec(0, 0, 2), WindowSpec(1, 1, 3))


def _artifact(spec: WindowSpec, value: float = 1.0):
    return OrdinaryWindowArtifact(
        spec=spec,
        depth=torch.full((spec.frame_count, 2, 3), value),
        confidence=torch.full((spec.frame_count, 2, 3), value + 1),
        camera_poses=torch.eye(4).repeat(spec.frame_count, 1, 1),
    )


def _store(
    tmp_path,
    mode=PredictionCacheMode.AUTO,
    *,
    key: str = "a" * 64,
):
    return OrdinaryPredictionStore(
        root=tmp_path / "predictions",
        fingerprint=_fingerprint(key),
        mode=mode,
        expected_specs=SPECS,
    )


def test_store_rejects_specs_that_do_not_match_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="do not match fingerprint"):
        OrdinaryPredictionStore(
            root=tmp_path / "predictions",
            fingerprint=_fingerprint(),
            mode=PredictionCacheMode.AUTO,
            expected_specs=(WindowSpec(0, 0, 3),),
        )


def test_auto_store_round_trips_exact_layout(tmp_path):
    store = _store(tmp_path)
    assert store.read_sequence() is None
    assert store.read_window(SPECS[0]) is None
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0], 2.0))
    store.write_window(_artifact(SPECS[1], 3.0))
    store.finalize()

    assert (store.entry_path / "manifest.json").is_file()
    assert (store.entry_path / "sequence.json").is_file()
    assert (store.entry_path / "windows/000000.pt").is_file()
    assert (store.entry_path / "windows/000001.pt").is_file()
    assert (store.entry_path / "locks/entry.lock").is_file()
    assert (store.entry_path / "invalid").is_dir()
    assert (store.entry_path / "complete.json").is_file()

    warm = _store(tmp_path)
    assert torch.equal(
        warm.read_sequence().reference_intrinsic,
        torch.eye(3),
    )
    assert warm.read_window(SPECS[0]).depth[0, 0, 0].item() == 2.0
    assert warm.read_window(SPECS[1]).depth[0, 0, 0].item() == 3.0
    assert warm.stats.ordinary_hits == 2
    assert warm.stats.stored_bytes > 0
    assert store.stats.saved_window_count == 2
    assert store.stats.stored_bytes > 0


def test_valid_partial_auto_entry_resumes_only_missing_windows(tmp_path):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))

    resumed = _store(tmp_path)
    assert resumed.read_window(SPECS[0]) is not None
    assert resumed.read_window(SPECS[1]) is None
    assert resumed.stats.ordinary_hits == 1
    assert resumed.stats.ordinary_misses == 1
    with pytest.raises(PredictionCacheMissError, match="finalize"):
        resumed.finalize()


def test_readonly_missing_entry_never_creates_directories(tmp_path):
    store = _store(tmp_path, PredictionCacheMode.READONLY)

    with pytest.raises(
        PredictionCacheMissError,
        match=r"window 000000 \[0,2\).*readonly",
    ):
        store.read_window(SPECS[0])

    assert not (tmp_path / "predictions").exists()


def test_off_mode_never_reads_or_writes_disk(tmp_path):
    store = _store(tmp_path, PredictionCacheMode.OFF)

    assert store.read_sequence() is None
    assert store.read_window(SPECS[0]) is None
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))
    with pytest.raises(PredictionCacheMissError):
        store.finalize()

    assert store.stats.ordinary_misses == 1
    assert not (tmp_path / "predictions").exists()


def test_auto_quarantines_truncated_window_and_returns_miss(tmp_path):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))
    (store.entry_path / "windows/000000.pt").write_bytes(b"truncated")

    assert store.read_window(SPECS[0]) is None
    assert store.stats.corrupt_count == 1
    assert store.stats.ordinary_misses == 1
    assert list((store.entry_path / "invalid").rglob("000000.pt"))
    event = store.stats.events[-1]
    assert event["event"] == "quarantine"
    assert event["reason"] == "window-000000-corrupt"
    assert event["original_path"] == str(
        (store.entry_path / "windows/000000.pt").resolve()
    )
    assert Path(event["quarantine_path"]).is_file()


def test_auto_rejects_window_copied_from_another_prediction_key(tmp_path):
    source = _store(tmp_path, key="a" * 64)
    source.write_sequence(SequenceArtifact(torch.eye(3)))
    source.write_window(_artifact(SPECS[0]))

    target = _store(tmp_path, key="e" * 64)
    target.write_sequence(SequenceArtifact(torch.eye(3)))
    target_window = target.entry_path / "windows/000000.pt"
    target_window.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source.entry_path / "windows/000000.pt",
        target_window,
    )

    assert target.read_window(SPECS[0]) is None
    assert target.stats.corrupt_count == 1
    assert list(target.invalid_path.rglob("000000.pt"))


def test_auto_rejects_same_shape_value_corruption(tmp_path):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))
    window_path = store.entry_path / "windows/000000.pt"
    payload = torch.load(window_path, weights_only=False)
    payload["artifact"]["depth"][0, 0, 0] += 7.0
    torch.save(payload, window_path)

    assert store.read_window(SPECS[0]) is None
    assert store.stats.corrupt_count == 1


def test_store_rejects_window_with_wrong_spatial_geometry(tmp_path):
    store = _store(tmp_path)
    wrong_geometry = OrdinaryWindowArtifact(
        spec=SPECS[0],
        depth=torch.ones((2, 4, 4)),
        confidence=torch.ones((2, 4, 4)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
    )

    with pytest.raises(ValueError, match="spatial geometry"):
        store.write_window(wrong_geometry)


def test_auto_rejects_sequence_value_corruption(tmp_path):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    payload = store._read_json(store.sequence_path)
    payload["reference_intrinsic"][0][0] += 3.0
    store._atomic_write_json(store.sequence_path, payload)

    assert store.read_sequence() is None
    assert store.stats.corrupt_count == 1
    assert list(store.invalid_path.rglob("sequence.json"))


def test_readonly_corruption_fails_without_quarantine(tmp_path):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))
    corrupt_path = store.entry_path / "windows/000000.pt"
    corrupt_path.write_bytes(b"truncated")

    readonly = _store(tmp_path, PredictionCacheMode.READONLY)
    with pytest.raises(PredictionCacheCorruptError, match="000000"):
        readonly.read_window(SPECS[0])

    assert corrupt_path.exists()
    assert list((store.entry_path / "invalid").iterdir()) == []


def test_wrong_window_spec_is_corrupt(tmp_path):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))
    window_path = store.entry_path / "windows/000000.pt"
    payload = torch.load(window_path, weights_only=False)
    payload["artifact"]["spec"]["frame_end"] = 1
    torch.save(payload, window_path)

    assert store.read_window(SPECS[0]) is None
    assert store.stats.corrupt_count == 1


def test_completion_marker_with_missing_window_is_corrupt_but_resumable(
    tmp_path,
):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))
    store.write_window(_artifact(SPECS[1]))
    store.finalize()
    (store.entry_path / "windows/000001.pt").unlink()

    resumed = _store(tmp_path)
    assert resumed.read_window(SPECS[1]) is None
    assert resumed.stats.corrupt_count == 1
    assert resumed.stats.ordinary_misses == 1
    assert not resumed.complete_path.exists()
    assert list(resumed.invalid_path.rglob("complete.json"))


def test_refresh_snapshots_old_artifacts_and_rebuilds(tmp_path):
    old = _store(tmp_path)
    old.write_sequence(SequenceArtifact(torch.eye(3)))
    old.write_window(_artifact(SPECS[0], 7.0))

    refreshed = _store(tmp_path, PredictionCacheMode.REFRESH)
    assert refreshed.read_window(SPECS[0]) is None
    assert list((refreshed.entry_path / "invalid").rglob("000000.pt"))
    refreshed.write_sequence(SequenceArtifact(torch.eye(3)))
    refreshed.write_window(_artifact(SPECS[0], 8.0))
    assert (
        refreshed.read_window(SPECS[0]).depth[0, 0, 0].item()
        == 8.0
    )


def test_nonblocking_entry_lock_rejects_second_owner(tmp_path):
    first = _store(tmp_path)
    second = _store(tmp_path)

    with first.entry_lock():
        with pytest.raises(BlockingIOError):
            with second.entry_lock(blocking=False):
                raise AssertionError("second lock unexpectedly acquired")


def test_failed_atomic_window_write_leaves_no_final_file(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    store.write_sequence(SequenceArtifact(torch.eye(3)))

    def fail_save(payload, path):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(
        "inference_engine.prediction_cache.store.torch.save",
        fail_save,
    )
    with pytest.raises(RuntimeError, match="serialization failed"):
        store.write_window(_artifact(SPECS[0]))

    assert not (store.entry_path / "windows/000000.pt").exists()
    assert not list((store.entry_path / "windows").glob("*.tmp"))


def test_warm_validation_and_size_scan_run_once_per_store(
    tmp_path,
    monkeypatch,
):
    cold = _store(tmp_path)
    cold.write_sequence(SequenceArtifact(torch.eye(3)))
    cold.write_window(_artifact(SPECS[0]))
    cold.write_window(_artifact(SPECS[1]))
    cold.finalize()

    warm = _store(tmp_path)
    manifest_calls = 0
    size_calls = 0
    original_manifest = warm._manifest_is_valid
    original_size = warm._update_stored_bytes

    def count_manifest():
        nonlocal manifest_calls
        manifest_calls += 1
        return original_manifest()

    def count_size():
        nonlocal size_calls
        size_calls += 1
        return original_size()

    monkeypatch.setattr(warm, "_manifest_is_valid", count_manifest)
    monkeypatch.setattr(warm, "_update_stored_bytes", count_size)

    assert warm.read_sequence() is not None
    assert warm.read_window(SPECS[0]) is not None
    assert warm.read_window(SPECS[1]) is not None
    assert manifest_calls == 1
    assert size_calls == 1


def test_cold_size_scan_runs_only_at_finalize(tmp_path, monkeypatch):
    store = _store(tmp_path)
    size_calls = 0
    original_size = store._update_stored_bytes

    def count_size():
        nonlocal size_calls
        size_calls += 1
        return original_size()

    monkeypatch.setattr(store, "_update_stored_bytes", count_size)
    store.write_sequence(SequenceArtifact(torch.eye(3)))
    store.write_window(_artifact(SPECS[0]))
    store.write_window(_artifact(SPECS[1]))
    store.finalize()

    assert size_calls == 1
