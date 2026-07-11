# Local Pi3 Weights and Image Sorting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load Pi3 from `weights/model.safetensors` by default without network fallback and feed naturally sorted image frames into sliding-window inference.

**Architecture:** Keep command-line parsing and inference orchestration in `demo.py`. Add pure helpers for natural filename keys and image discovery so ordering and sampling can be tested without executing the model; validate the local checkpoint before constructing `Pi3`.

**Tech Stack:** Python, argparse, pathlib/os, re, PyTorch, safetensors, pytest

## Global Constraints

- Preserve the user's existing explanatory comments and unrelated local modifications.
- Keep `--model_ckpt` available as an override.
- Never fall back to `Pi3.from_pretrained`.
- Sort supported images naturally and case-insensitively before applying `sample_interval`.
- Avoid loading the real checkpoint in tests.

---

### Task 1: Local checkpoint selection and validation

**Files:**
- Modify: `demo.py`
- Create: `tests/test_demo.py`

**Interfaces:**
- Consumes: `get_args_parser() -> argparse.ArgumentParser`, `load_model(args)`
- Produces: default `args.model_ckpt == "weights/model.safetensors"`; `load_model` raises `FileNotFoundError` for a missing path before constructing `Pi3`

- [ ] **Step 1: Write failing parser and missing-file tests**

```python
def test_model_checkpoint_defaults_to_local_safetensors():
    args = demo.get_args_parser().parse_args([])
    assert args.model_ckpt == "weights/model.safetensors"


def test_model_checkpoint_can_be_overridden():
    args = demo.get_args_parser().parse_args(["--model_ckpt", "custom/model.pt"])
    assert args.model_ckpt == "custom/model.pt"


def test_load_model_rejects_missing_checkpoint_before_model_construction(tmp_path, monkeypatch):
    args = demo.get_args_parser().parse_args(["--model_ckpt", str(tmp_path / "missing.safetensors")])
    monkeypatch.setattr(demo, "Pi3", lambda: pytest.fail("Pi3 must not be constructed"))
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        demo.load_model(args)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_demo.py -k 'model_checkpoint or missing_checkpoint'`

Expected: default-path assertion and missing-checkpoint assertion fail because current code defaults to `None` and constructs `Pi3` first.

- [ ] **Step 3: Implement local-only checkpoint behavior**

Set the parser default to `weights/model.safetensors`. At the beginning of `load_model`, resolve `args.model_ckpt`, raise `FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")` if it is not a file, construct `Pi3`, and load the local checkpoint based on its suffix. Remove the `from_pretrained` branch.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_demo.py -k 'model_checkpoint or missing_checkpoint'`

Expected: `3 passed`.

### Task 2: Natural image ordering before sampling

**Files:**
- Modify: `demo.py`
- Modify: `tests/test_demo.py`

**Interfaces:**
- Produces: `natural_sort_key(path: str) -> list[tuple[int, object]]`; `discover_images(data_path: str, sample_interval: int) -> list[str]`
- Consumed by: `run_dynamic_scene(args)` calls `discover_images(args.data_path, args.sample_interval)`

- [ ] **Step 1: Write failing natural-order and discovery tests**

```python
def test_natural_sort_key_orders_embedded_numbers_numerically():
    names = ["frame10.jpg", "Frame2.jpg", "frame1.jpg"]
    assert sorted(names, key=demo.natural_sort_key) == ["frame1.jpg", "Frame2.jpg", "frame10.jpg"]


def test_discover_images_filters_sorts_then_samples(tmp_path):
    for name in ["frame10.jpg", "frame2.PNG", "frame1.jpeg", "notes.txt", "frame3.jpg"]:
        (tmp_path / name).touch()
    assert [Path(path).name for path in demo.discover_images(str(tmp_path), 2)] == ["frame1.jpeg", "frame3.jpg"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_demo.py -k 'natural_sort or discover_images'`

Expected: tests fail because both helpers are undefined.

- [ ] **Step 3: Implement natural sorting and discovery**

Add `import re`. Implement a natural key by splitting `os.path.basename(path).lower()` on `(\d+)`, representing numeric and text pieces with tagged tuples so all keys remain comparable. Implement `discover_images` by filtering `.png`, `.jpg`, and `.jpeg` case-insensitively, joining paths, applying `sorted(..., key=natural_sort_key)`, then slicing with `sample_interval`. Replace the existing `os.listdir` list comprehension in `run_dynamic_scene` with this helper.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_demo.py -k 'natural_sort or discover_images'`

Expected: `2 passed`.

### Task 3: Regression verification

**Files:**
- Verify: `demo.py`
- Verify: `tests/test_demo.py`

**Interfaces:**
- Consumes all behavior implemented in Tasks 1 and 2.
- Produces a syntactically valid demo and a passing focused test suite.

- [ ] **Step 1: Run all focused tests**

Run: `pytest -q tests/test_demo.py`

Expected: all tests pass.

- [ ] **Step 2: Compile the modified script**

Run: `python -m py_compile demo.py`

Expected: exit code 0 with no output.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff --check && git diff -- demo.py tests/test_demo.py`

Expected: no whitespace errors; diff contains only local checkpoint loading, natural image discovery, and their tests while retaining existing comments.
