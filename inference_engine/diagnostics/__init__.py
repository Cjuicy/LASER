"""Opt-in diagnostics for LASER segmentation experiments."""

from .schema import (
    SCHEMA_VERSION,
    DiagnosticContext,
    RunManifest,
    SelectedInterval,
)
from .segmentation import compare_labelings, summarize_labels, trace_segmentation_frame
from .merge import DiagnosticParityError, LayerAtomicMergeTrace, analyze_layer_atomic_merge

__all__ = [
    "SCHEMA_VERSION",
    "DiagnosticContext",
    "RunManifest",
    "SelectedInterval",
    "compare_labelings",
    "summarize_labels",
    "trace_segmentation_frame",
    "DiagnosticParityError",
    "LayerAtomicMergeTrace",
    "analyze_layer_atomic_merge",
]
