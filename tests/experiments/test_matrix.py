from __future__ import annotations

import pytest

from experiments.config import EvaluationKind
from experiments.matrix import MatrixEntry, build_matrix, validate_matrix_entry
from pipeline.config import ReconstructionMode, SegmentationMethod


def test_ate_matrix_is_exact_three_by_three_product():
    entries = build_matrix(EvaluationKind.ATE)

    assert {
        (entry.segmentation_method.value, entry.reconstruction_mode.value)
        for entry in entries
    } == {
        (segmentation, mode)
        for segmentation in ("depth", "geometry", "atomic")
        for mode in ("no_loop", "traditional", "corrected")
    }
    assert len(entries) == 9


def test_pointcloud_matrix_is_exact_three_no_loop_entries():
    entries = build_matrix(EvaluationKind.POINTCLOUD)

    assert [
        (entry.segmentation_method.value, entry.reconstruction_mode.value)
        for entry in entries
    ] == [
        ("depth", "no_loop"),
        ("geometry", "no_loop"),
        ("atomic", "no_loop"),
    ]


def test_pointcloud_matrix_rejects_loop_mode_before_runner_call():
    entry = MatrixEntry(
        SegmentationMethod.DEPTH,
        ReconstructionMode.TRADITIONAL,
    )

    with pytest.raises(ValueError, match="pointcloud matrix only supports no_loop"):
        validate_matrix_entry(EvaluationKind.POINTCLOUD, entry)


def test_matrix_entries_have_stable_unique_names():
    entries = (*build_matrix(EvaluationKind.ATE), *build_matrix(EvaluationKind.POINTCLOUD))
    assert all(entry.name == f"{entry.segmentation_method.value}-{entry.reconstruction_mode.value}" for entry in entries)
    assert len({entry.name for entry in build_matrix(EvaluationKind.ATE)}) == 9
