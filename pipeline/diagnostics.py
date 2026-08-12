from __future__ import annotations

from collections.abc import Mapping, Sequence

from pipeline.artifacts import ReconstructionDiagnostics


def summarize_split_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    integer_totals = (
        "split_parent_count",
        "split_proposed_count",
        "split_accepted_count",
        "split_added_regions",
        "split_reject_no_markers",
        "split_reject_small_child",
        "split_reject_low_score",
    )
    totals: dict[str, int | float] = {
        key: sum(int(item.get(key, 0)) for item in diagnostics)
        for key in integer_totals
    }
    totals["split_runtime_ms"] = sum(
        float(item.get("split_runtime_ms", 0.0)) for item in diagnostics
    )
    return totals


def diagnostics_summary(
    diagnostics: ReconstructionDiagnostics,
) -> dict[str, object]:
    if not isinstance(diagnostics, ReconstructionDiagnostics):
        raise ValueError("diagnostics must be ReconstructionDiagnostics")
    segmentation = tuple(diagnostics.segmentation_summaries)
    return {
        "candidate_count": diagnostics.candidate_count,
        "constraint_count": diagnostics.constraint_count,
        "stage_timings_ms": dict(diagnostics.stage_timings_ms),
        "mode_scalars": dict(diagnostics.mode_scalars),
        "split_totals": summarize_split_diagnostics(segmentation),
    }


__all__ = ["diagnostics_summary", "summarize_split_diagnostics"]
