from .base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopCandidate,
    LoopClosureStrategy,
    LoopConstraint,
    LoopSolution,
    ReconstructionResult,
    WindowCache,
    validate_sim3,
)
from .shared import detect_loop_candidates
from .corrected import (
    CorrectedLoopProcessor,
    build_local_loop_constraint,
)
from .traditional import (
    TraditionalLoopProcessor,
    compute_sim3_ab,
)
from .registry import LOOP_PROCESSORS, build_loop_processor


__all__ = [
    "WINDOW_CACHE_SCHEMA_VERSION",
    "LoopCandidate",
    "LoopClosureStrategy",
    "LoopConstraint",
    "LoopSolution",
    "LOOP_PROCESSORS",
    "ReconstructionResult",
    "CorrectedLoopProcessor",
    "TraditionalLoopProcessor",
    "WindowCache",
    "build_local_loop_constraint",
    "build_loop_processor",
    "compute_sim3_ab",
    "detect_loop_candidates",
    "validate_sim3",
]
