from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine.utils import registration_confidence
from inference_engine.utils.registration_confidence import (
    select_top_confidence_mask,
    validate_confidence_keep_ratio,
)


def test_top_thirty_percent_uses_point_seven_quantile():
    confidence = torch.arange(10, dtype=torch.float32)

    mask = select_top_confidence_mask(confidence, keep_ratio=0.3)

    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_top_confidence_mask_supports_mixed_precision_cpu_tensors(dtype):
    confidence = torch.arange(10, dtype=dtype)

    mask = select_top_confidence_mask(confidence, keep_ratio=0.3)

    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_non_finite_confidence_is_never_selected():
    confidence = torch.tensor([1.0, float("nan"), float("inf"), 4.0])

    mask = select_top_confidence_mask(confidence, keep_ratio=0.5)

    assert mask.tolist() == [False, False, False, True]


def test_confidence_selection_rejects_no_finite_values():
    confidence = torch.tensor([float("nan"), float("inf")])

    with pytest.raises(ValueError, match="no finite values"):
        select_top_confidence_mask(confidence, keep_ratio=0.3)


def test_mutual_confidence_mask_rejects_disjoint_selections():
    source = torch.tensor([True, False])
    target = torch.tensor([False, True])

    with pytest.raises(
        ValueError,
        match="loop candidate has no shared high-confidence pixels",
    ):
        registration_confidence.intersect_confidence_masks(
            source,
            target,
            context="loop candidate",
        )


def test_mutual_confidence_mask_returns_shared_selection():
    source = torch.tensor([True, True, False])
    target = torch.tensor([False, True, True])

    mutual = registration_confidence.intersect_confidence_masks(
        source,
        target,
    )

    assert mutual.tolist() == [False, True, False]


@pytest.mark.parametrize("keep_ratio", [0.0, -0.1, 1.1, float("nan")])
def test_confidence_keep_ratio_must_be_in_unit_interval(keep_ratio):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        validate_confidence_keep_ratio(keep_ratio)
