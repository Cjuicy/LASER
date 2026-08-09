from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from inference_engine.prediction_cache.types import (
    OrdinaryWindowArtifact,
    SequenceArtifact,
    WindowSpec,
    build_window_specs,
    validate_window_specs,
)


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    (
        (
            15,
            (
                WindowSpec(0, 0, 10),
                WindowSpec(1, 5, 15),
            ),
        ),
        (
            12,
            (
                WindowSpec(0, 0, 10),
                WindowSpec(1, 5, 12),
            ),
        ),
        (6, (WindowSpec(0, 0, 6),)),
        (10, (WindowSpec(0, 0, 10),)),
    ),
)
def test_window_schedule_matches_existing_sliding_semantics(
    frame_count,
    expected,
):
    assert build_window_specs(frame_count, 10, 5) == expected


def test_trailing_overlap_only_window_is_omitted():
    assert build_window_specs(20, 10, 5) == (
        WindowSpec(0, 0, 10),
        WindowSpec(1, 5, 15),
        WindowSpec(2, 10, 20),
    )


@pytest.mark.parametrize(
    "arguments",
    (
        (5, 10, 5),
        (10, 5, 5),
        (10, 5, 0),
        (0, 10, 5),
        (True, 10, 5),
    ),
)
def test_invalid_window_schedule_is_rejected(arguments):
    with pytest.raises(ValueError):
        build_window_specs(*arguments)


def test_window_specs_are_immutable_and_report_frame_count():
    spec = WindowSpec(1, 5, 12)
    assert spec.frame_count == 7
    with pytest.raises(FrozenInstanceError):
        spec.frame_end = 13


def test_validate_window_specs_rejects_noncanonical_ranges():
    with pytest.raises(ValueError, match="canonical"):
        validate_window_specs(
            (
                WindowSpec(0, 0, 10),
                WindowSpec(2, 6, 15),
            ),
            frame_count=15,
            window_size=10,
            overlap=5,
        )


def _artifact(**changes) -> OrdinaryWindowArtifact:
    values = {
        "spec": WindowSpec(0, 0, 2),
        "depth": torch.ones((2, 3, 4)),
        "confidence": torch.ones((2, 3, 4)),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
    }
    values.update(changes)
    return OrdinaryWindowArtifact(**values)


def test_compact_window_artifact_accepts_exact_cpu_shapes():
    artifact = _artifact()
    assert artifact.depth.shape == (2, 3, 4)
    assert artifact.confidence.shape == (2, 3, 4)
    assert artifact.camera_poses.shape == (2, 4, 4)
    assert artifact.depth.device.type == "cpu"


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"depth": torch.ones((1, 3, 4))}, "depth"),
        ({"confidence": torch.ones((2, 3, 3))}, "confidence"),
        ({"camera_poses": torch.eye(4)}, "camera_poses"),
        ({"depth": torch.ones((2, 3, 4), dtype=torch.int64)}, "floating"),
        (
            {"confidence": torch.full((2, 3, 4), float("nan"))},
            "finite",
        ),
        (
            {"camera_poses": torch.full((2, 4, 4), float("inf"))},
            "finite",
        ),
    ),
)
def test_compact_window_artifact_rejects_invalid_tensors(changes, message):
    with pytest.raises(ValueError, match=message):
        _artifact(**changes)


def test_sequence_artifact_requires_finite_three_by_three_intrinsic():
    intrinsic = torch.tensor(
        [
            [500.0, 0.0, 259.0],
            [0.0, 500.0, 77.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert SequenceArtifact(intrinsic).reference_intrinsic is intrinsic

    with pytest.raises(ValueError, match="shape"):
        SequenceArtifact(torch.eye(4))
    with pytest.raises(ValueError, match="finite"):
        SequenceArtifact(torch.full((3, 3), float("nan")))
