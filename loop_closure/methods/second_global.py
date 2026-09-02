from __future__ import annotations

from loop_closure.methods.corrected import CorrectedLoopProcessor
from pipeline.config import ReconstructionMode


class SecondGlobalLoopProcessor(CorrectedLoopProcessor):
    """Residual graph optimization for materialized Traditional geometry."""

    name = ReconstructionMode.TRADITIONAL_SECOND_GLOBAL


__all__ = ["SecondGlobalLoopProcessor"]
