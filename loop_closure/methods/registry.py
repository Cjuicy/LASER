from __future__ import annotations

from pipeline.config import LoopMethod

from .corrected import CorrectedLoopClosureStrategy
from .traditional import TraditionalLoopClosureStrategy


LOOP_STRATEGIES = {
    LoopMethod.TRADITIONAL: TraditionalLoopClosureStrategy,
    LoopMethod.CORRECTED: CorrectedLoopClosureStrategy,
}


def build_loop_strategy(method: LoopMethod, **dependencies):
    try:
        strategy_type = LOOP_STRATEGIES[method]
    except KeyError:
        raise ValueError(f"unsupported loop method: {method!r}") from None
    return strategy_type(**dependencies)

