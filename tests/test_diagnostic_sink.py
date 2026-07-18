import json

import numpy as np

from inference_engine.diagnostics.schema import SCHEMA_VERSION, DiagnosticContext, SelectedInterval
from inference_engine.diagnostics.sink import FileDiagnosticSink, NullDiagnosticSink
from inference_engine.diagnostics.storage import StorageBudget


def _context(pass_id):
    return DiagnosticContext("run", "layer_atomic_split", "02", pass_id, 3, 12)


def test_null_sink_is_safe_for_every_event():
    sink = NullDiagnosticSink()
    assert sink.emit_segmentation(_context(1), 0, {"x": 1}, {"labels": np.zeros((2, 2))}) is None
    assert sink.emit_scale(_context(1), {"x": 1}) is None
    assert sink.close() is None


def test_pass1_writes_scalar_jsonl_but_no_dense_arrays(tmp_path):
    budget = StorageBudget(tmp_path, max_bytes=100_000, warn_bytes=90_000, min_free_bytes=0)
    sink = FileDiagnosticSink(tmp_path, budget=budget)
    sink.emit_segmentation(_context(1), 0, {"segments": np.int64(3)}, {"labels": np.arange(4).reshape(2, 2)})

    records = list(tmp_path.rglob("segmentation.jsonl"))
    assert len(records) == 1
    assert json.loads(records[0].read_text().splitlines()[0])["metrics"]["segments"] == 3
    stored = json.loads(records[0].read_text().splitlines()[0])
    assert stored["schema_version"] == SCHEMA_VERSION
    assert stored["run_id"] == "run"
    assert not list(tmp_path.rglob("*.npz"))


def test_pass2_only_saves_selected_dense_arrays_and_compacts_types(tmp_path):
    selected = [SelectedInterval("02", 13, 13, ("trajectory",), 2.0)]
    sink = FileDiagnosticSink(
        tmp_path,
        selected_intervals=selected,
        budget=StorageBudget(tmp_path, max_bytes=1_000_000, warn_bytes=900_000, min_free_bytes=0),
    )
    context = _context(2)
    sink.emit_segmentation(context, 0, {"segments": 2}, {"labels": np.arange(4).reshape(2, 2), "mask": np.array([[1, 0], [0, 1]], bool)})
    sink.emit_segmentation(context, 1, {"segments": 2}, {"labels": np.arange(4).reshape(2, 2), "mask": np.array([[1, 0], [0, 1]], bool)})

    traces = list(tmp_path.rglob("*.npz"))
    assert len(traces) == 1
    with np.load(traces[0]) as data:
        assert data["labels"].dtype == np.uint16
        assert data["mask__packed"].dtype == np.uint8
        assert data["mask__shape"].tolist() == [2, 2]


def test_input_capture_uses_selected_frames(tmp_path):
    sink = FileDiagnosticSink(tmp_path, selected_intervals=[SelectedInterval("02", 12, 12, ("control",), 1.0)])
    context = _context(2)
    sink.emit_inputs(context, np.ones((2, 2, 2, 3)), np.ones((2, 2, 2, 3)), np.ones((2, 2, 2)))
    assert len(list(tmp_path.rglob("inputs-frame-000012.npz"))) == 1
    assert not list(tmp_path.rglob("inputs-frame-000013.npz"))
