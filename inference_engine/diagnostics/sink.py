"""No-op-by-default diagnostic event sinks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np

from .schema import DiagnosticContext, SelectedInterval
from .storage import StorageBudget, append_jsonl


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class NullDiagnosticSink:
    def emit_segmentation(self, context, local_index, metrics, arrays=None):
        return None
    def snapshot_direct_anchors(self, context, graphs):
        return None
    def emit_scale(self, context, metrics, arrays=None):
        return None
    def emit_temporal(self, context, metrics, arrays=None):
        return None
    def emit_inputs(self, context, images, point_maps, confidence):
        return None
    def close(self):
        return None


class FileDiagnosticSink:
    """Writes scalar events for both passes and dense arrays only for selected frames."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        selected_intervals: list[SelectedInterval] | tuple[SelectedInterval, ...] = (),
        budget: StorageBudget | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.selected_intervals = tuple(selected_intervals)
        self.budget = budget or StorageBudget(self.root)
        self._lock = threading.Lock()
        self._closed = False

    def _base(self, context: DiagnosticContext) -> Path:
        return self.root / context.config_id / context.sequence_id / f"pass{context.pass_id}"

    def _selected(self, context: DiagnosticContext, frame_id: int) -> bool:
        return context.pass_id == 2 and any(
            interval.sequence_id == context.sequence_id and interval.contains(frame_id)
            for interval in self.selected_intervals
        )

    def _record(self, context: DiagnosticContext, family: str, metrics: dict[str, Any], **extra: Any) -> None:
        if self._closed:
            raise RuntimeError("diagnostic sink is closed")
        record = {"context": context.to_dict(), "metrics": _json_safe(metrics), **_json_safe(extra)}
        encoded_size = len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 1
        self.budget.enforce(estimated_bytes=encoded_size)
        append_jsonl(self._base(context) / f"{family}.jsonl", record)

    @staticmethod
    def _compact_arrays(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
        compact: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype == bool:
                compact[f"{name}__packed"] = np.packbits(array.reshape(-1))
                compact[f"{name}__shape"] = np.asarray(array.shape, dtype=np.int32)
            elif "label" in name and np.issubdtype(array.dtype, np.integer):
                largest = int(array.max()) if array.size else 0
                compact[name] = array.astype(np.uint16 if largest <= np.iinfo(np.uint16).max else np.uint32)
            elif np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 4:
                compact[name] = array.astype(np.float32)
            else:
                compact[name] = array
        return compact

    def _write_npz(self, path: Path, arrays: dict[str, Any]) -> Path:
        compact = self._compact_arrays(arrays)
        estimated = int(sum(value.nbytes for value in compact.values()))
        self.budget.enforce(estimated_bytes=estimated)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        try:
            with self._lock, partial.open("wb") as handle:
                np.savez_compressed(handle, **compact)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
        self.budget.enforce()
        return path

    def emit_segmentation(self, context, local_index, metrics, arrays=None):
        frame_id = context.frame_id(local_index)
        self._record(context, "segmentation", metrics, frame_id=frame_id, local_index=local_index)
        if arrays and self._selected(context, frame_id):
            self._write_npz(self._base(context) / "traces" / f"segmentation-frame-{frame_id:06d}.npz", arrays)

    def snapshot_direct_anchors(self, context, graphs):
        return [[{
            "scale": [float(value) for value in vertex.cache.get("scale", ())],
            "iou": [float(value) for value in vertex.cache.get("iou", ())],
        } for vertex in layer] for layer in graphs]

    def emit_scale(self, context, metrics, arrays=None):
        self._record(context, "scale", metrics)
        if arrays and any(self._selected(context, context.frame_id(index)) for index in range(len(next(iter(arrays.values()))))):
            self._write_npz(self._base(context) / "traces" / f"scale-window-{context.window_id:06d}.npz", arrays)

    def emit_temporal(self, context, metrics, arrays=None):
        self._record(context, "temporal", metrics)
        if arrays and context.pass_id == 2:
            self._write_npz(self._base(context) / "traces" / f"temporal-window-{context.window_id:06d}.npz", arrays)

    def emit_inputs(self, context, images, point_maps, confidence):
        arrays = {"rgb": images, "point_map": point_maps, "confidence": confidence}
        count = int(np.asarray(point_maps).shape[0])
        for local_index in range(count):
            frame_id = context.frame_id(local_index)
            if self._selected(context, frame_id):
                self._write_npz(
                    self._base(context) / "traces" / f"inputs-frame-{frame_id:06d}.npz",
                    {name: np.asarray(value)[local_index] for name, value in arrays.items()},
                )

    def close(self):
        self._closed = True
