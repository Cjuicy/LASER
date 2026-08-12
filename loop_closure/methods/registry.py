from __future__ import annotations

from pipeline.config import ReconstructionMode

from .corrected import CorrectedLoopProcessor
from .traditional import TraditionalLoopProcessor


LOOP_PROCESSORS = {
    ReconstructionMode.TRADITIONAL: TraditionalLoopProcessor,
    ReconstructionMode.CORRECTED: CorrectedLoopProcessor,
}


def build_loop_processor(mode: ReconstructionMode, **dependencies):
    try:
        processor_type = LOOP_PROCESSORS[mode]
    except KeyError:
        raise ValueError(f"unsupported loop reconstruction mode: {mode!r}") from None
    return processor_type(**dependencies)
