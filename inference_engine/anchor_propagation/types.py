"""Shared immutable records for HART anchor propagation."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


ANCHOR_PROPAGATION_MODES = ("none", "legacy_iou", "hart")


def resolve_anchor_propagation(depth_refine, explicit=None):
    """Resolve the new propagation selector without changing legacy callers."""
    if explicit is None:
        return "legacy_iou" if depth_refine else "none"
    if explicit not in ANCHOR_PROPAGATION_MODES:
        raise ValueError(
            f"Unknown anchor_propagation: {explicit!r}; expected one of "
            f"{ANCHOR_PROPAGATION_MODES}."
        )
    return explicit


@dataclass(frozen=True)
class SegmentationFrame:
    leaf_labels: np.ndarray
    parent_labels: np.ndarray
    anchor_labels: np.ndarray
    leaf_to_parent: np.ndarray
    anchor_to_leaf: np.ndarray
    split_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class SegmentationWindow:
    frames: tuple[SegmentationFrame, ...]
    segment_mode: str

    @property
    def shape(self):
        if not self.frames:
            return (0, 0, 0)
        return (len(self.frames), *self.frames[0].leaf_labels.shape)


@dataclass(frozen=True)
class PairRelation:
    src_id: int
    tgt_id: int
    intersection: int
    iou: float
    src_coverage: float
    tgt_coverage: float

    @property
    def correspondence_score(self):
        return max(self.iou, self.src_coverage, self.tgt_coverage)


@dataclass(frozen=True)
class TemporalEdge:
    src_id: int
    tgt_id: int
    relation: PairRelation
    is_primary: bool


@dataclass(frozen=True)
class TrackWindow:
    segment_ids: tuple[np.ndarray, ...]
    primary_edges: tuple[tuple[TemporalEdge, ...], ...]
    secondary_edges: tuple[tuple[TemporalEdge, ...], ...]
    lineage_parents: dict[int, int | None] = field(default_factory=dict)

    @property
    def segment_count(self):
        if not self.segment_ids:
            return 0
        return len(
            {
                int(segment_id)
                for frame_ids in self.segment_ids
                for segment_id in frame_ids.tolist()
            }
        )


@dataclass(frozen=True)
class RegistrationState:
    final_base_points_tail: torch.Tensor
    final_base_poses_tail: torch.Tensor
    pose_support_mask_tail: np.ndarray
    cumulative_sim3: tuple[Any, Any, Any] | None = None


@dataclass(frozen=True)
class AnchorPropagationState:
    local_residual_tail: np.ndarray
    confidence_tail: np.ndarray
    segments_tail: tuple[SegmentationFrame, ...]


@dataclass(frozen=True)
class PropagationResult:
    window_scale: float
    local_residual_mask: np.ndarray
    pose_support_mask: np.ndarray
    next_state: AnchorPropagationState
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PoseConsensus:
    window_scale: float
    segment_ids: frozenset[int]
    group_count: int
    support_pixels: int
    valid_pixels: int
    support_ratio: float
    accepted: bool


@dataclass(frozen=True)
class DirectAnchor:
    current_frame: int
    current_anchor_id: int
    current_segment_id: int
    previous_segment_id: int
    scale: float
    pixel_count: int
