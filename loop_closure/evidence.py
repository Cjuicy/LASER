from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from loop_closure.types import LoopCandidate, Sim3


@runtime_checkable
class LoopWindow(Protocol):
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor


class LoopEvidenceProvider(Protocol):
    def estimate(
        self,
        window_a: LoopWindow,
        window_b: LoopWindow,
        candidate: LoopCandidate,
    ) -> tuple[Sim3, Sim3]:
        raise NotImplementedError
