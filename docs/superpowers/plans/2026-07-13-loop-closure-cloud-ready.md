# Loop-Closure Cloud-Ready Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 AutoDL 上的 `demo_lc.py` 对 `data/00/image_2` 使用唯一、自然排序的帧清单执行回环检测，并默认启用现有 layer-atomic 新分割。

**Architecture:** 把图片发现与自然排序提取到无模型依赖的公共模块，入口只生成一次帧清单，再显式传给流式引擎、`LoopClosureEngine` 和 SALAD `LoopDetector`。在回环入口增加默认模式、启动前校验、窗口缓存完整性校验和简洁诊断日志，但不改变分割、SALAD、Sim(3) 或缓存数据格式。

**Tech Stack:** Python 3.11、PyTorch、NumPy、SALAD/FAISS、PyPose、pytest、现有 Cython 分割扩展。

## Global Constraints

- 实现分支为 `codex/loop-closure-cloud-ready`，基线提交为 `7dca6248b58ebbeacdda9cdf108aab439c527479`。
- Pi3、SALAD 和 DINO 权重已在 `weights/`；不得下载、复制或重新打包权重。
- 输入数据为 AutoDL 上的 `data/00/image_2` 平铺图片目录。
- 不修改 layer-atomic 分割算法、参数或阈值。
- 不修改 SALAD 相似度阈值、NMS、Sim(3) 估计或优化算法。
- 不增加 Docker、Conda、数据集下载或权重下载功能。
- 不合并 `demo.py` 与 `demo_lc.py`，不修改缓存张量和可视化输出格式。
- 图片扩展名必须不区分大小写地支持 `.png`、`.jpg` 和 `.jpeg`。
- `sample_interval` 必须在自然排序后且仅应用一次。
- 帧索引 `i` 在推理、SALAD 和回环约束阶段必须指向同一图片路径。
- 回环入口默认启用 layer-atomic 深度细化，并提供 `--no-depth-refine` 关闭选项。
- 本地不要求完整模型推理；必须重新编译 Cython 扩展并通过完整 pytest 测试集。

---

## File Responsibility Map

- `utils/image_paths.py`：唯一的图片过滤、自然排序和采样实现，无模型依赖。
- `demo.py`：复用公共图片工具，保持普通推理行为不变。
- `demo_lc.py`：回环 CLI、输入校验、唯一帧清单、缓存检查和日志。
- `loop_closure/loop_model.py`：SALAD 接收显式帧清单；仅在未提供时扫描。
- `loop_closure/loop_closure.py`：回环引擎复用同一清单；保留目录回退。
- `tests/test_image_paths.py`：公共图片工具行为测试。
- `tests/test_demo.py`、`tests/test_demo_lc.py`：入口接线、CLI 和保护检查测试。
- `tests/test_loop_image_manifest.py`：帧清单贯穿 SALAD 与回环引擎的测试。
- `docs/autodl-loop-closure.md`：中文拉取、验证、运行和排错手册。
- `README.md`：链接 AutoDL 手册并更新默认深度细化说明。

---

### Task 1: 提取唯一的图片发现与自然排序模块

**Files:**
- Create: `utils/image_paths.py`
- Create: `tests/test_image_paths.py`
- Modify: `demo.py:19-67`
- Modify: `demo_lc.py:19-66`
- Modify: `tests/test_demo.py`
- Modify: `tests/test_demo_lc.py`

**Interfaces:**
- Produces: `natural_sort_key(path: str | os.PathLike[str]) -> list[tuple[int, str | int]]`
- Produces: `discover_images(data_path: str | os.PathLike[str], sample_interval: int = 1) -> list[str]`
- Preserves: the two helper names as imports on both demo modules.

- [ ] **Step 1: 写失败测试**

Create `tests/test_image_paths.py`:

```python
from pathlib import Path

import pytest

from utils.image_paths import discover_images, natural_sort_key


def test_natural_sort_key_orders_embedded_numbers_numerically():
    names = ["frame10.jpg", "Frame2.jpg", "frame1.jpg"]
    assert sorted(names, key=natural_sort_key) == [
        "frame1.jpg", "Frame2.jpg", "frame10.jpg",
    ]


def test_discover_images_filters_sorts_then_samples(tmp_path):
    for name in [
        "frame10.jpg", "frame2.PNG", "frame1.jpeg",
        "notes.txt", "frame3.JPG",
    ]:
        (tmp_path / name).touch()
    image_names = discover_images(tmp_path, sample_interval=2)
    assert [Path(path).name for path in image_names] == [
        "frame1.jpeg", "frame3.JPG",
    ]


def test_discover_images_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        discover_images(tmp_path / "missing")


@pytest.mark.parametrize("sample_interval", [0, -1])
def test_discover_images_rejects_non_positive_interval(tmp_path, sample_interval):
    with pytest.raises(ValueError, match="sample_interval must be at least 1"):
        discover_images(tmp_path, sample_interval=sample_interval)
```

- [ ] **Step 2: 确认 RED**

Run: `pytest -q tests/test_image_paths.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'utils.image_paths'`.

- [ ] **Step 3: 实现公共工具**

Create `utils/image_paths.py`:

```python
import os
from pathlib import Path
import re


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def natural_sort_key(path: str | os.PathLike[str]) -> list[tuple[int, str | int]]:
    parts = re.split(r"(\d+)", Path(path).name.lower())
    return [(1, int(part)) if part.isdigit() else (0, part) for part in parts]


def discover_images(
    data_path: str | os.PathLike[str],
    sample_interval: int = 1,
) -> list[str]:
    if sample_interval < 1:
        raise ValueError("sample_interval must be at least 1")
    image_dir = Path(data_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_paths = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_paths.sort(key=natural_sort_key)
    return [str(path) for path in image_paths[::sample_interval]]
```

- [ ] **Step 4: 接入两个入口**

Add to both `demo.py` and `demo_lc.py`:

```python
from utils.image_paths import discover_images, natural_sort_key
```

Remove their local helper definitions and unused `re` imports. Append to the corresponding demo tests:

```python
from utils.image_paths import (
    discover_images as shared_discover_images,
    natural_sort_key as shared_natural_sort_key,
)


def test_demo_reexports_shared_image_helpers():
    assert demo.discover_images is shared_discover_images
    assert demo.natural_sort_key is shared_natural_sort_key
```

Append this complete counterpart to `tests/test_demo_lc.py`:

```python
from utils.image_paths import (
    discover_images as shared_discover_images,
    natural_sort_key as shared_natural_sort_key,
)


def test_demo_lc_reexports_shared_image_helpers():
    assert demo_lc.discover_images is shared_discover_images
    assert demo_lc.natural_sort_key is shared_natural_sort_key
```

- [ ] **Step 5: 验证 GREEN 与回归**

Run: `pytest -q tests/test_image_paths.py tests/test_demo.py tests/test_demo_lc.py`

Expected: all focused tests pass.

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 6: 提交**

```bash
git add utils/image_paths.py demo.py demo_lc.py tests/test_image_paths.py tests/test_demo.py tests/test_demo_lc.py
git commit -m "refactor: unify image sequence discovery"
```

---

### Task 2: 把唯一帧清单贯穿 SALAD 与回环引擎

**Files:**
- Modify: `loop_closure/loop_model.py:14-100`
- Modify: `loop_closure/loop_closure.py:48-92,202-216`
- Modify: `demo_lc.py:69-215`
- Create: `tests/test_loop_image_manifest.py`
- Modify: `tests/test_demo_lc.py`

**Interfaces:**
- Consumes: Task 1 `discover_images`.
- Changes: `LoopDetector.__init__(image_dir, sample_interval=1, output="loop_closures.txt", config=None, image_paths=None)`.
- Changes: `LoopClosureEngine.__init__(config, image_dir, output_dir, pi3_model, window_size, overlap, sample_interval=1, top_conf_percentile=0.5, image_paths=None)`.
- Produces: `build_loop_closure_engine(config, args, pi3_model, image_paths, cache_path_lc)`.
- Changes: `run_dynamic_scene(args) -> tuple[str, list[str]]`.

- [ ] **Step 1: 写 LoopDetector 失败测试**

Create `tests/test_loop_image_manifest.py`:

```python
from pathlib import Path

from loop_closure import loop_model


def loop_config():
    return {
        "Weights": {"SALAD": "weights/dino_salad.ckpt"},
        "Loop": {"SALAD": {
            "image_size": [336, 336], "batch_size": 32,
            "similarity_threshold": 0.7, "top_k": 5,
            "use_nms": True, "nms_threshold": 25,
        }},
    }


def test_loop_detector_uses_explicit_manifest_without_rescanning(monkeypatch, tmp_path):
    image_paths = [str(tmp_path / "frame2.png"), str(tmp_path / "frame10.png")]
    monkeypatch.setattr(
        loop_model,
        "discover_images",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("explicit manifest must not rescan")
        ),
    )
    detector = loop_model.LoopDetector(
        image_dir=tmp_path, sample_interval=3,
        config=loop_config(), image_paths=image_paths,
    )
    assert detector.get_image_paths() is image_paths


def test_loop_detector_fallback_uses_shared_discovery(monkeypatch, tmp_path):
    expected = [str(tmp_path / "frame1.png")]
    calls = []
    def fake_discover(data_path, sample_interval):
        calls.append((Path(data_path), sample_interval))
        return expected
    monkeypatch.setattr(loop_model, "discover_images", fake_discover)
    detector = loop_model.LoopDetector(
        image_dir=tmp_path, sample_interval=4, config=loop_config(),
    )
    assert detector.get_image_paths() is expected
    assert calls == [(tmp_path, 4)]
```

- [ ] **Step 2: 确认 RED**

Run: `pytest -q tests/test_loop_image_manifest.py`

Expected: collection or construction fails because the new import and keyword do not exist.

- [ ] **Step 3: 实现 LoopDetector 显式清单**

Import `discover_images` from `utils.image_paths`. Add `image_paths=None` to the constructor, set `self.image_paths = image_paths`, and replace `get_image_paths` with:

```python
def get_image_paths(self):
    if self.image_paths is None:
        self.image_paths = discover_images(
            self.image_dir,
            sample_interval=self.sample_interval,
        )
    return self.image_paths
```

- [ ] **Step 4: 写回环引擎与入口失败测试**

Append to `tests/test_loop_image_manifest.py` a module-loader test that stubs unavailable heavy dependencies and records detector construction:

```python
import importlib.util
import sys
import types
import torch


def load_engine_module(monkeypatch, detector_calls):
    fake_loop_model = types.ModuleType("loop_closure.loop_model")
    class RecordingDetector:
        def __init__(self, **kwargs):
            detector_calls.append(kwargs)
    fake_loop_model.LoopDetector = RecordingDetector
    monkeypatch.setitem(sys.modules, "loop_closure.loop_model", fake_loop_model)

    fake_optimizer_module = types.ModuleType("loop_closure.utils.sim3loop")
    class IdentityOptimizer:
        def __init__(self, config):
            self.calls = []
        def optimize(self, transforms, constraints):
            self.calls.append((transforms, constraints))
            return transforms
    fake_optimizer_module.Sim3LoopOptimizer = IdentityOptimizer
    monkeypatch.setitem(sys.modules, "loop_closure.utils.sim3loop", fake_optimizer_module)

    fake_utils = types.ModuleType("loop_closure.utils.sim3utils")
    fake_utils.process_loop_list = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "loop_closure.utils.sim3utils", fake_utils)
    fake_load = types.ModuleType("utils.load_fn")
    fake_load.load_and_preprocess_images = lambda paths: paths
    monkeypatch.setitem(sys.modules, "utils.load_fn", fake_load)
    fake_pi3 = types.ModuleType("pi3.models.pi3")
    fake_pi3.Pi3 = object
    monkeypatch.setitem(sys.modules, "pi3.models.pi3", fake_pi3)
    fake_engine = types.ModuleType("inference_engine")
    fake_engine.StreamingWindowEngine = object
    monkeypatch.setitem(sys.modules, "inference_engine", fake_engine)
    fake_inference = types.ModuleType("inference_engine.inference_utils")
    fake_inference.register_adjacent_windows = lambda *args: None
    monkeypatch.setitem(sys.modules, "inference_engine.inference_utils", fake_inference)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))

    name = "loop_closure._manifest_test_module"
    path = Path(__file__).resolve().parents[1] / "loop_closure" / "loop_closure.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "loop_closure"
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def engine_config():
    config = loop_config()
    config["Model"] = {"loop_chunk_size": 20}
    config["Loop"]["SIM3_Optimizer"] = {
        "lang_version": "python", "max_iterations": 30,
        "lambda_init": "1e-6",
    }
    return config


def test_loop_engine_passes_explicit_manifest_to_detector(monkeypatch, tmp_path):
    calls = []
    module = load_engine_module(monkeypatch, calls)
    image_paths = [str(tmp_path / "frame2.png"), str(tmp_path / "frame10.png")]
    engine = module.LoopClosureEngine(
        engine_config(), tmp_path, tmp_path / "output", object(), 10, 5,
        sample_interval=3, image_paths=image_paths,
    )
    assert engine.img_list is image_paths
    assert calls[0]["image_paths"] is image_paths


def test_zero_loop_constraints_preserve_existing_transforms(monkeypatch, tmp_path):
    calls = []
    module = load_engine_module(monkeypatch, calls)
    images = [str(tmp_path / f"frame{i}.png") for i in range(16)]
    engine = module.LoopClosureEngine(
        engine_config(), tmp_path, tmp_path / "output", object(), 10, 5,
        image_paths=images,
    )
    original = [(1.0, torch.eye(3), torch.zeros(3))]
    engine.sim3_list = original
    engine.process_loops([])
    assert engine.sim3_list is original
    assert engine.loop_optimizer.calls == [(original, [])]
```

Append to `tests/test_demo_lc.py`:

```python
def test_build_loop_closure_engine_passes_canonical_manifest(monkeypatch, tmp_path):
    calls = []
    def recording_engine(*args, **kwargs):
        calls.append((args, kwargs))
        return object()
    monkeypatch.setattr(demo_lc, "LoopClosureEngine", recording_engine)
    args = demo_lc.get_args_parser().parse_args(["--data_path", str(tmp_path)])
    images = [str(tmp_path / "frame1.png")]
    demo_lc.build_loop_closure_engine({}, args, object(), images, tmp_path / "lc")
    assert calls[0][1]["image_paths"] is images
```

- [ ] **Step 5: 确认第二轮 RED**

Run: `pytest -q tests/test_loop_image_manifest.py tests/test_demo_lc.py -k "manifest or canonical"`

Expected: the engine rejects `image_paths`, and `build_loop_closure_engine` is missing.

- [ ] **Step 6: 实现 LoopClosureEngine 与入口接线**

In `loop_closure/loop_closure.py`, import `discover_images`, add `image_paths=None` to the constructor, set `self.img_list = image_paths`, and pass `image_paths=image_paths` into `LoopDetector`. Replace the scan at the start of `run` with:

```python
if self.img_list is None:
    self.img_list = discover_images(
        self.img_dir,
        sample_interval=self.sample_interval,
    )
if not self.img_list:
    raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
```

After `get_loop_list()` print `Detected loop pairs: {len(self.loop_list)}`.

In `demo_lc.py`, add:

```python
def build_loop_closure_engine(config, args, pi3_model, image_paths, cache_path_lc):
    return LoopClosureEngine(
        config, args.data_path, cache_path_lc, pi3_model,
        args.window_size, args.overlap, args.sample_interval,
        image_paths=image_paths,
    )
```

Change `run_dynamic_scene` to return `(scene_name, image_names)`, using `Path(data_path).name` for the default scene. At the main call site, unpack both values and construct the loop engine through `build_loop_closure_engine`.

- [ ] **Step 7: 验证 GREEN 与回归**

Run: `pytest -q tests/test_loop_image_manifest.py tests/test_demo_lc.py`

Expected: all focused tests pass.

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 8: 提交**

```bash
git add loop_closure/loop_model.py loop_closure/loop_closure.py demo_lc.py tests/test_loop_image_manifest.py tests/test_demo_lc.py
git commit -m "fix: preserve frame order through loop closure"
```

---

### Task 3: 默认启用新分割并增加 AutoDL 保护检查

**Files:**
- Modify: `demo_lc.py`
- Modify: `loop_closure/loop_closure.py:14-15`
- Modify: `tests/test_demo_lc.py`

**Interfaces:**
- Produces: `select_runtime_dtype(runtime_device: str) -> torch.dtype`.
- Produces: `load_and_validate_inputs(args) -> tuple[dict, list[str]]`.
- Produces: `require_complete_cache_files(cache_dir, expected_count: int) -> list[Path]`.
- Changes: `run_model(image_name_windows) -> None`.
- Changes: `run_dynamic_scene(args, image_names) -> tuple[str, int]`.
- CLI defaults: `config_path == "configs/loop_config.yaml"` and `depth_refine is True`.

- [ ] **Step 1: 写 CLI 默认值的失败测试**

Append to `tests/test_demo_lc.py`:

```python
def test_loop_defaults_enable_layer_atomic_refinement_and_standard_config():
    args = demo_lc.get_args_parser().parse_args([])
    assert args.config_path == "configs/loop_config.yaml"
    assert args.depth_refine is True


def test_no_depth_refine_disables_layer_atomic_refinement():
    args = demo_lc.get_args_parser().parse_args(["--no-depth-refine"])
    assert args.depth_refine is False
```

- [ ] **Step 2: 确认 RED**

Run: `pytest -q tests/test_demo_lc.py -k "loop_defaults or no_depth_refine"`

Expected: both tests fail because current defaults are `None` and `False`.

- [ ] **Step 3: 实现回环默认模式**

Replace the related parser arguments in `demo_lc.py` with:

```python
parser.add_argument(
    '--config_path', default='configs/loop_config.yaml', type=str,
    help='loop closure config',
)
depth_refine_group = parser.add_mutually_exclusive_group()
depth_refine_group.add_argument(
    '--depth-refine', dest='depth_refine', action='store_true',
    help='enable layer-atomic depth refinement',
)
depth_refine_group.add_argument(
    '--no-depth-refine', dest='depth_refine', action='store_false',
    help='disable layer-atomic depth refinement',
)
parser.set_defaults(depth_refine=True)
```

- [ ] **Step 4: 写启动检查的失败测试**

Append to `tests/test_demo_lc.py`:

```python
def valid_cloud_args_and_config(tmp_path):
    data_path = tmp_path / "image_2"
    data_path.mkdir()
    for name in ["000000.png", "000001.png"]:
        (data_path / name).touch()
    model_ckpt = tmp_path / "model.safetensors"
    config_path = tmp_path / "loop_config.yaml"
    salad = tmp_path / "dino_salad.ckpt"
    dino = tmp_path / "dinov2.pth"
    for path in [model_ckpt, config_path, salad, dino]:
        path.touch()
    args = demo_lc.get_args_parser().parse_args([
        "--data_path", str(data_path), "--model_ckpt", str(model_ckpt),
        "--config_path", str(config_path), "--window_size", "2",
        "--overlap", "1",
    ])
    config = {
        "Weights": {"SALAD": str(salad), "DINO": str(dino)},
        "Model": {"loop_enable": True},
    }
    return args, config


def test_load_and_validate_inputs_returns_canonical_images(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)
    loaded, images = demo_lc.load_and_validate_inputs(args)
    assert loaded is config
    assert [Path(path).name for path in images] == ["000000.png", "000001.png"]


def test_load_and_validate_inputs_requires_cuda(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        demo_lc.load_and_validate_inputs(args)


@pytest.mark.parametrize(("window_size", "overlap"), [(5, 5), (4, 5), (5, 0)])
def test_load_and_validate_inputs_rejects_invalid_windowing(
    tmp_path, monkeypatch, window_size, overlap,
):
    args, config = valid_cloud_args_and_config(tmp_path)
    args.window_size = window_size
    args.overlap = overlap
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)
    with pytest.raises(ValueError, match="window_size must be greater than overlap"):
        demo_lc.load_and_validate_inputs(args)


def test_load_and_validate_inputs_requires_enabled_loop_mode(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    config["Model"]["loop_enable"] = False
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)
    with pytest.raises(ValueError, match="Model.loop_enable must be true"):
        demo_lc.load_and_validate_inputs(args)


def test_load_and_validate_inputs_reports_missing_loop_weight(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    Path(config["Weights"]["SALAD"]).unlink()
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)
    with pytest.raises(FileNotFoundError, match="SALAD checkpoint not found"):
        demo_lc.load_and_validate_inputs(args)
```

- [ ] **Step 5: 确认启动检查 RED**

Run: `pytest -q tests/test_demo_lc.py -k "validate_inputs"`

Expected: tests fail because `load_and_validate_inputs` is missing.

- [ ] **Step 6: 实现 CPU 安全导入和输入检查**

Replace module-level dtype selection in `demo_lc.py` with:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"


def select_runtime_dtype(runtime_device):
    if runtime_device != "cuda":
        return torch.float32
    return (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )


dtype = select_runtime_dtype(device)
```

Use this CPU-safe form in `loop_closure/loop_closure.py`:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16 if device == "cuda" else torch.float32
)
```

Add to `demo_lc.py`:

```python
def require_file(path_value, label):
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_and_validate_inputs(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for demo_lc.py")
    if args.overlap < 1 or args.window_size <= args.overlap:
        raise ValueError(
            "window_size must be greater than overlap, and overlap must be at least 1"
        )
    require_file(args.model_ckpt, "Pi3 checkpoint")
    config_path = require_file(args.config_path, "Loop config")
    config = load_config(str(config_path))
    if config.get("Model", {}).get("loop_enable") is not True:
        raise ValueError("Model.loop_enable must be true for demo_lc.py")
    weights = config.get("Weights", {})
    require_file(weights.get("SALAD", ""), "SALAD checkpoint")
    require_file(weights.get("DINO", ""), "DINO checkpoint")
    image_paths = discover_images(args.data_path, args.sample_interval)
    if len(image_paths) <= args.overlap:
        raise ValueError(
            f"Image count must be greater than overlap: "
            f"images={len(image_paths)}, overlap={args.overlap}"
        )
    return config, image_paths
```

- [ ] **Step 7: 写缓存完整性失败测试**

Append to `tests/test_demo_lc.py`:

```python
def test_require_complete_cache_files_sorts_numeric_ids(tmp_path):
    for name in ["window_cache_10.pt", "window_cache_2.pt", "window_cache_1.pt"]:
        (tmp_path / name).touch()
    files = demo_lc.require_complete_cache_files(tmp_path, expected_count=3)
    assert [path.name for path in files] == [
        "window_cache_1.pt", "window_cache_2.pt", "window_cache_10.pt",
    ]


def test_require_complete_cache_files_rejects_incomplete_output(tmp_path):
    (tmp_path / "window_cache_0.pt").touch()
    with pytest.raises(RuntimeError, match="Window cache count mismatch"):
        demo_lc.require_complete_cache_files(tmp_path, expected_count=2)
```

- [ ] **Step 8: 确认缓存检查 RED**

Run: `pytest -q tests/test_demo_lc.py -k "complete_cache"`

Expected: tests fail because `require_complete_cache_files` is missing.

- [ ] **Step 9: 实现缓存检查、窗口计数与日志**

Add to `demo_lc.py`:

```python
def require_complete_cache_files(cache_dir, expected_count):
    cache_files = sorted(
        Path(cache_dir).glob("window_cache_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if len(cache_files) != expected_count:
        raise RuntimeError(
            "Window cache count mismatch: "
            f"expected={expected_count}, actual={len(cache_files)}, "
            f"cache_dir={cache_dir}"
        )
    print(f"Completed cache windows: {len(cache_files)}/{expected_count}")
    return cache_files


def print_run_configuration(args, image_paths, expected_windows):
    print("Loop closure: enabled")
    state = "enabled" if args.depth_refine else "disabled"
    print(f"Layer-atomic depth refinement: {state}")
    print(f"Input directory: {args.data_path}")
    print(
        f"Frames: {len(image_paths)} "
        f"(first={Path(image_paths[0]).name}, last={Path(image_paths[-1]).name})"
    )
    print(
        f"Windows: {expected_windows} "
        f"(window_size={args.window_size}, overlap={args.overlap})"
    )
```

Change `run_model` to consume precomputed windows: rename its parameter to `image_name_windows` and remove its internal `model.img_sliding_window(image_names)` call. It continues to run inference and print timing, and does not return a value.

Replace `run_dynamic_scene` with the single window-construction point so diagnostics are printed before inference:

```python
def run_dynamic_scene(args, image_names):
    scene_name = (
        Path(args.data_path).name
        if args.scene_name is None
        else args.scene_name
    )
    image_name_windows = model.img_sliding_window(image_names)
    expected_windows = len(image_name_windows)
    print_run_configuration(args, image_names, expected_windows)
    run_model(image_name_windows)
    return scene_name, expected_windows
```

Reorder the main block to:

```python
args = get_args_parser().parse_args()
config, image_names = load_and_validate_inputs(args)
pi3_model, model = load_model(args)
model.eval()
scene_name, expected_windows = run_dynamic_scene(args, image_names)
cache_path = Path(args.cache_path)
cache_path_lc = cache_path.parent / f'{cache_path.name}_lc'
lc_engine = build_loop_closure_engine(
    config, args, pi3_model, image_names, cache_path_lc,
)
cache_files = require_complete_cache_files(
    model.temp_cache_dir, expected_windows,
)
raw_predictions = [
    StreamingWindowEngineLC.parse_cache_file(path) for path in cache_files
]
```

Keep the existing Sim(3) correction and aggregation below this block. After `save_for_viser`, add:

```python
print(f"Saved loop-closure result: {Path(args.output_path) / scene_name}")
```

- [ ] **Step 10: 验证 GREEN 与回归**

Run: `pytest -q tests/test_demo_lc.py tests/test_loop_image_manifest.py`

Expected: all focused tests pass.

Run: `pytest -q`

Expected: all tests pass without new warnings.

- [ ] **Step 11: 提交**

```bash
git add demo_lc.py loop_closure/loop_closure.py tests/test_demo_lc.py
git commit -m "feat: harden loop closure cloud runner"
```

---

### Task 4: 编写 AutoDL 中文手册并完成交付验证

**Files:**
- Create: `docs/autodl-loop-closure.md`
- Modify: `README.md:89-109`

**Interfaces:**
- Documents: branch checkout, Cython build, pytest, KITTI 00 smoke/full runs, opt-out, output, diagnostics.
- Preserves: existing installation and non-loop inference documentation.

- [ ] **Step 1: 创建中文运行手册**

Create `docs/autodl-loop-closure.md` with:

````markdown
# AutoDL 回环检测运行手册

## 拉取、编译和测试

```bash
cd ~/autodl-tmp/LASER
git fetch origin codex/loop-closure-cloud-ready
git switch codex/loop-closure-cloud-ready
python setup.py build_ext --inplace
pytest -q
```

如果环境尚未安装回环依赖：

```bash
pip install -r requirements.txt
pip install faiss-gpu-cu12 numpy==1.26.4
```

## 检查数据与权重

```bash
ls data/00/image_2 | head
ls weights/model.safetensors \
   weights/dino_salad.ckpt \
   weights/dinov2_vitb14_pretrain.pth
```

## 冒烟运行

```bash
python demo_lc.py \
  --data_path data/00/image_2 \
  --scene_name kitti_00_smoke \
  --cache_path inference_cache/kitti_00_smoke \
  --output_path viser_results \
  --sample_interval 10 \
  --window_size 10 \
  --overlap 5
```

## 完整运行

```bash
python demo_lc.py \
  --data_path data/00/image_2 \
  --scene_name kitti_00 \
  --cache_path inference_cache/kitti_00 \
  --output_path viser_results \
  --window_size 10 \
  --overlap 5
```

回环入口默认开启 layer-atomic 深度细化。只有基线对比时才添加
`--no-depth-refine`。结果位于 `viser_results/kitti_00`，可视化命令：

```bash
python viser/visualizer_monst3r.py --data viser_results/kitti_00
```
````

- [ ] **Step 2: 更新 README**

Under `### Loop-closure inference`, link `docs/autodl-loop-closure.md`, state that layer-atomic refinement is on by default, document `--no-depth-refine`, and use:

```bash
python demo_lc.py \
    --data_path "data/00/image_2" \
    --scene_name "kitti_00" \
    --output_path "./viser_results" \
    --cache_path "./inference_cache/kitti_00" \
    --sample_interval 1 \
    --window_size 10 \
    --overlap 5
```

- [ ] **Step 3: 验证文档、语法和接口名称**

Run:

```bash
rg -n "data/00/image_2|no-depth-refine|codex/loop-closure-cloud-ready|kitti_00" README.md docs/autodl-loop-closure.md
python -m py_compile demo.py demo_lc.py loop_closure/loop_model.py loop_closure/loop_closure.py utils/image_paths.py
```

Expected: `rg` finds every handoff term and `py_compile` exits 0.

- [ ] **Step 4: 重新编译并运行完整测试**

Run: `python setup.py build_ext --inplace`

Expected: both Cython segmentation extensions build successfully.

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 5: 检查范围和工作区**

Run:

```bash
git diff 7dca6248b58ebbeacdda9cdf108aab439c527479 --stat
git diff --check 7dca6248b58ebbeacdda9cdf108aab439c527479
git status --short
```

Expected: only approved docs, shared discovery, loop wiring, tests and AutoDL guide changed; `git diff --check` exits 0. Ignore the pre-existing untracked `.DS_Store` files and never stage them.

- [ ] **Step 6: 提交文档**

```bash
git add README.md docs/autodl-loop-closure.md
git commit -m "docs: add AutoDL loop closure guide"
```

- [ ] **Step 7: 推送分支**

Run: `git push -u origin codex/loop-closure-cloud-ready`

Expected: `origin/codex/loop-closure-cloud-ready` is available for AutoDL checkout.
