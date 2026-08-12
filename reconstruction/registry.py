from __future__ import annotations

from dataclasses import dataclass

from loop_closure.detection import LoopDetector
from loop_closure.evidence import LoopEvidenceProvider
from pipeline.config import OptimizerConfig, ReconstructionMode
from reconstruction.modes.base import ReconstructionModeRunner
from reconstruction.modes.corrected import CorrectedReconstructionMode
from reconstruction.modes.no_loop import NoLoopReconstructionMode
from reconstruction.modes.traditional import TraditionalReconstructionMode


@dataclass(frozen=True)
class ReconstructionServices:
    detector: LoopDetector | None = None
    evidence: LoopEvidenceProvider | None = None
    optimizer_config: OptimizerConfig | None = None


def build_reconstruction_mode(
    mode: ReconstructionMode,
    services: ReconstructionServices,
) -> ReconstructionModeRunner:
    if not isinstance(mode, ReconstructionMode):
        raise ValueError("reconstruction mode is invalid")
    if not isinstance(services, ReconstructionServices):
        raise ValueError("reconstruction services are invalid")
    loop_services = (
        services.detector,
        services.evidence,
        services.optimizer_config,
    )
    if mode is ReconstructionMode.NO_LOOP:
        if any(item is not None for item in loop_services):
            raise ValueError("no_loop must not receive loop services")
        return NoLoopReconstructionMode()
    if any(item is None for item in loop_services):
        raise ValueError(f"{mode.value} requires complete loop services")
    arguments = {
        "detector": services.detector,
        "evidence": services.evidence,
        "optimizer_config": services.optimizer_config,
    }
    if mode is ReconstructionMode.TRADITIONAL:
        return TraditionalReconstructionMode(**arguments)
    if mode is ReconstructionMode.CORRECTED:
        return CorrectedReconstructionMode(**arguments)
    raise ValueError(f"unsupported reconstruction mode: {mode!r}")


__all__ = ["ReconstructionServices", "build_reconstruction_mode"]
