# LASER 回环矫正流水线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让回环模式先完整执行非回环式前向矫正，再基于统一图像清单和最高置信度 30% 的对应点构建回环约束，最后只应用优化增量。

**Architecture:** `demo_lc.py` 生成唯一图像清单并向下传递；流式 LC 引擎立即应用窗口绝对 Sim(3) 和现有分割 mask，同时缓存绝对变换与相对顺序边；回环引擎使用矫正缓存构建 `C_AB`，优化后重新累积绝对变换；聚合阶段计算 `D_i = G_hat_i compose inverse(G_i)`，不重复应用原始尺度或 mask。

**Tech Stack:** Python 3.11+、PyTorch、NumPy、Pi3、SALAD、现有 Sim(3) 优化器、pytest、Cython。

## Global Constraints

- 实现分支必须是 `codex/loop-closure-corrected-pipeline`。
- 基线必须包含 `98cce5f9f470599aca0cf5a6614f39409d929d58`。
- 回环链路默认 `registration_top_confidence_ratio=0.3`，内部使用 `0.7` 分位点。
- 非回环入口未显式传入新参数时，默认行为必须保持不变。
- 三种分割方法、分割参数、区域匹配、锚点传播和 `refine_depth_segments()` 不得修改。
- SALAD 描述子提取和 `loop_closure/utils/sim3loop.py` 的优化器数学实现不得修改。
- 显式图像清单已经排序和采样，下游不得重新扫描、排序或采样。
- 分割 mask 必须在第一阶段立即应用，并且最终聚合不得重复应用。
- 所有生产代码必须遵循 RED → GREEN → REFACTOR。

---

### Task 1: 统一图像清单

**Files:**
- Create: `utils/image_paths.py`
- Create: `tests/test_loop_image_manifest.py`
- Modify: `demo_lc.py:19-74`
- Modify: `loop_closure/loop_model.py:11-100`
- Modify: `loop_closure/loop_closure.py:48-90,202-216`
- Test: `tests/test_demo_lc.py`

**Interfaces:**
- Produces: `natural_sort_key(path: str | os.PathLike[str]) -> list[tuple[int, str | int]]`
- Produces: `discover_images(data_path: str | os.PathLike[str], sample_interval: int = 1) -> list[str]`
- Produces: `LoopDetector(..., image_paths: list[str] | None = None)`
- Produces: `LoopClosureEngine(..., image_paths: list[str] | None = None)`

- [ ] **Step 1: 写图像发现和显式清单失败测试**

```python
def test_discover_images_naturally_sorts_before_sampling(tmp_path):
    for name in ("frame10.jpg", "frame2.PNG", "frame1.jpeg", "note.txt"):
        (tmp_path / name).touch()

    paths = discover_images(tmp_path, sample_interval=2)

    assert [Path(path).name for path in paths] == [
        "frame1.jpeg",
        "frame10.jpg",
    ]


def test_loop_detector_preserves_explicit_manifest(monkeypatch, tmp_path):
    manifest = [
        str(tmp_path / "frame2.png"),
        str(tmp_path / "frame10.png"),
    ]
    detector = LoopDetector(
        image_dir=tmp_path,
        sample_interval=4,
        config=loop_config(),
        image_paths=manifest,
    )

    assert detector.get_image_paths() is manifest
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
pytest -q tests/test_loop_image_manifest.py tests/test_demo_lc.py -k "manifest or discover_images or natural_sort"
```

Expected: 因 `utils.image_paths`、`image_paths` 构造参数或显式清单转发尚不存在而失败。

- [ ] **Step 3: 实现公共图像发现**

```python
import os
from pathlib import Path
import re


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def natural_sort_key(
    path: str | os.PathLike[str],
) -> list[tuple[int, str | int]]:
    parts = re.split(r"(\d+)", Path(path).name.lower())
    return [
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
    ]


def discover_images(
    data_path: str | os.PathLike[str],
    sample_interval: int = 1,
) -> list[str]:
    if sample_interval < 1:
        raise ValueError("sample_interval must be at least 1")
    image_dir = Path(data_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    paths.sort(key=natural_sort_key)
    return [str(path) for path in paths[::sample_interval]]
```

`demo_lc.py` 从 `utils.image_paths` 导入两个函数。`LoopDetector` 和
`LoopClosureEngine` 保存显式 `image_paths`；只有其值为 `None` 时才调用
`discover_images()`。

- [ ] **Step 4: 运行图像清单测试并确认 GREEN**

Run:

```bash
pytest -q tests/test_loop_image_manifest.py tests/test_demo_lc.py
```

Expected: PASS。

- [ ] **Step 5: 提交 Task 1**

```bash
git add utils/image_paths.py tests/test_loop_image_manifest.py tests/test_demo_lc.py demo_lc.py loop_closure/loop_model.py loop_closure/loop_closure.py
git commit -m "fix: preserve image order through loop closure"
```

### Task 2: 独立且统一的 0.3 配准置信度

**Files:**
- Create: `inference_engine/utils/registration_confidence.py`
- Create: `tests/test_registration_confidence.py`
- Modify: `inference_engine/streaming_window_engine.py:69-126,265-297`
- Modify: `inference_engine/streaming_window_engine_lc.py:22-94`
- Modify: `loop_closure/loop_closure.py:48-68,145-193`
- Modify: `demo_lc.py:34-104,181-193`
- Test: `tests/test_segmentation_engine_modes.py`

**Interfaces:**
- Produces: `validate_confidence_keep_ratio(keep_ratio: float) -> float`
- Produces: `select_top_confidence_mask(confidence: torch.Tensor, keep_ratio: float = 0.3) -> torch.Tensor`
- Produces: `StreamingWindowEngine(..., registration_top_confidence_ratio: float | None = None)`
- Produces: `StreamingWindowEngineLC(..., registration_top_confidence_ratio: float = 0.3)`
- Produces: `LoopClosureEngine(..., registration_top_confidence_ratio: float = 0.3)`

- [ ] **Step 1: 写比例语义、非有限值和兼容性失败测试**

```python
def test_top_thirty_percent_uses_point_seven_quantile():
    conf = torch.arange(10, dtype=torch.float32)

    mask = select_top_confidence_mask(conf, keep_ratio=0.3)

    assert mask.tolist() == [
        False, False, False, False, False,
        False, True, True, True, True,
    ]


def test_non_finite_confidence_is_never_selected():
    conf = torch.tensor([1.0, float("nan"), float("inf"), 4.0])

    mask = select_top_confidence_mask(conf, keep_ratio=0.5)

    assert mask.tolist() == [True, False, False, True]


def test_base_engine_keeps_legacy_registration_ratio_by_default(tmp_path):
    engine = StreamingWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        benchmark_latency=False,
        top_conf_percentile=0.5,
    )

    assert engine.registration_top_confidence_ratio == 0.5
    assert engine.top_conf_percentile == 0.5
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
pytest -q tests/test_registration_confidence.py tests/test_segmentation_engine_modes.py
```

Expected: 因公共选择函数和新构造参数不存在而失败。

- [ ] **Step 3: 实现配准专用 mask**

```python
import math

import torch


def validate_confidence_keep_ratio(keep_ratio: float) -> float:
    ratio = float(keep_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("confidence keep ratio must be in (0, 1]")
    return ratio


def select_top_confidence_mask(
    confidence: torch.Tensor,
    keep_ratio: float = 0.3,
) -> torch.Tensor:
    ratio = validate_confidence_keep_ratio(keep_ratio)
    finite = torch.isfinite(confidence)
    finite_values = confidence[finite]
    if finite_values.numel() == 0:
        raise ValueError("confidence contains no finite values")
    threshold = torch.quantile(
        finite_values,
        1.0 - ratio,
        interpolation="nearest",
    )
    return finite & (confidence >= threshold)
```

基础引擎保存：

```python
legacy_ratio = 1.0 if top_conf_percentile is None else top_conf_percentile
self.registration_top_confidence_ratio = validate_confidence_keep_ratio(
    legacy_ratio
    if registration_top_confidence_ratio is None
    else registration_top_confidence_ratio
)
self.top_conf_percentile = (
    1 - top_conf_percentile
    if top_conf_percentile is not None
    else 0.0
)
```

父引擎、LC 引擎和回环 A/B 配准都通过
`select_top_confidence_mask()` 分别筛选两侧，再取 `&`。

- [ ] **Step 4: 运行置信度测试并确认 GREEN**

Run:

```bash
pytest -q tests/test_registration_confidence.py tests/test_segmentation_engine_modes.py
```

Expected: PASS，并验证分割字段仍保持旧语义。

- [ ] **Step 5: 提交 Task 2**

```bash
git add inference_engine/utils/registration_confidence.py tests/test_registration_confidence.py tests/test_segmentation_engine_modes.py inference_engine/streaming_window_engine.py inference_engine/streaming_window_engine_lc.py loop_closure/loop_closure.py demo_lc.py
git commit -m "fix: unify loop registration confidence ratio"
```

### Task 3: LC 第一阶段立即矫正并缓存明确变换

**Files:**
- Create: `tests/test_streaming_window_engine_lc_pipeline.py`
- Modify: `inference_engine/streaming_window_engine_lc.py:52-173`

**Interfaces:**
- Produces in every cache: `sim3_abs: tuple[float | torch.Tensor, torch.Tensor, torch.Tensor]`
- Produces in non-first caches: `sim3_edge: tuple[float | torch.Tensor, torch.Tensor, torch.Tensor]`
- Preserves: `scale_mask` is diagnostic and already applied

- [ ] **Step 1: 写前向传播和边组合失败测试**

使用伪造的反投影、配准和分割细化函数，直接向注册队列送入三个小窗口：

```python
def test_lc_worker_uses_corrected_previous_window_for_next_registration(
    monkeypatch,
    tmp_path,
):
    registered_sources = []
    masks = iter((2.0, 3.0))

    def fake_register(source_points, target_points, *args):
        registered_sources.append(source_points.clone())
        return 2.0, torch.eye(3), torch.zeros(3)

    def fake_refine(*args):
        value = next(masks)
        return torch.full((2, 1, 1, 1), value)

    engine = make_lc_engine(tmp_path, depth_refine=True)
    monkeypatch.setattr(lc_module, "register_adjacent_windows", fake_register)
    monkeypatch.setattr(lc_module, "refine_depth_segments", fake_refine)

    run_registration_worker(engine, [window(1.0), window(1.0), window(1.0)])

    assert torch.all(registered_sources[1] == 4.0)


def test_lc_cache_edge_recomposes_current_absolute_transform(tmp_path):
    caches = run_two_window_fixture(tmp_path)
    previous_abs = caches[0]["sim3_abs"]
    current_abs = caches[1]["sim3_abs"]
    edge = caches[1]["sim3_edge"]

    recomposed = accumulate_sim3(previous_abs, edge)

    assert_sim3_close(recomposed, current_abs)
```

- [ ] **Step 2: 运行 LC 流水线测试并确认 RED**

Run:

```bash
pytest -q tests/test_streaming_window_engine_lc_pipeline.py
```

Expected: 当前缓存没有 `sim3_abs/sim3_edge`，并且第三窗口配准源仍未矫正。

- [ ] **Step 3: 按非回环顺序实现 LC 注册 worker**

非首窗口的核心顺序为：

```python
current_abs = register_adjacent_windows(
    self.prev_window_cache["local_points"][-self.overlap:],
    working_window["local_points"][:self.overlap],
    self.prev_window_cache["camera_poses"][-self.overlap:],
    working_window["camera_poses"][:self.overlap],
    conf_mask,
)
previous_abs = self.prev_window_cache["sim3_abs"]
previous_inverse = closed_form_inverse_sim3(*previous_abs)
working_window["sim3_abs"] = current_abs
working_window["sim3_edge"] = accumulate_sim3(
    previous_inverse,
    current_abs,
)
s_d, rotation, translation = current_abs
working_window["local_points"] = s_d * working_window["local_points"]
working_window["camera_poses"] = apply_sim3_to_pose(
    working_window["camera_poses"],
    s_d,
    rotation,
    translation,
)
```

启用细化时：

```python
scale_mask = refine_depth_segments(
    self.prev_window_cache["local_points"].cpu().numpy(),
    working_window["local_points"].cpu().numpy(),
    self.anchor_sp_graph,
    target_graph,
    self.overlap,
)
working_window["scale_mask"] = scale_mask
working_window["local_points"] = (
    working_window["local_points"] * scale_mask
)
```

首窗口缓存单位 `sim3_abs`，不包含 `sim3_edge`。

- [ ] **Step 4: 运行 LC 流水线和现有分割模式测试**

Run:

```bash
pytest -q tests/test_streaming_window_engine_lc_pipeline.py tests/test_segmentation_engine_modes.py
```

Expected: PASS。

- [ ] **Step 5: 提交 Task 3**

```bash
git add tests/test_streaming_window_engine_lc_pipeline.py inference_engine/streaming_window_engine_lc.py
git commit -m "fix: propagate corrected windows before loop detection"
```

### Task 4: 使用矫正缓存构建回环约束并返回优化绝对变换

**Files:**
- Create: `tests/test_loop_closure_pipeline.py`
- Modify: `loop_closure/loop_closure.py:88-216`

**Interfaces:**
- Consumes: `raw_predictions[i]["sim3_abs"]`
- Consumes: `raw_predictions[i]["sim3_edge"]` for `i > 0`
- Produces: `LoopClosureEngine.run(raw_predictions) -> list[Sim3]`，长度等于窗口数
- Produces: `C_AB = L_B compose inverse(L_A)`

- [ ] **Step 1: 写无回环、矫正缓存和优化边失败测试**

```python
def test_no_loop_returns_original_absolute_transforms(engine, caches):
    engine.loop_detector.run = lambda: None
    engine.loop_detector.get_loop_list = lambda: []

    optimized_abs = engine.run(caches)

    assert_sim3_lists_close(
        optimized_abs,
        [cache["sim3_abs"] for cache in caches],
    )
    assert engine.loop_optimizer.calls == []


def test_loop_registration_receives_corrected_cache_points(
    engine,
    corrected_caches,
    monkeypatch,
):
    sources = []

    def fake_register(source_points, *args):
        sources.append(source_points.clone())
        return 1.0, torch.eye(3), torch.zeros(3)

    monkeypatch.setattr(loop_module, "register_adjacent_windows", fake_register)
    engine.process_loops(corrected_caches)

    assert torch.equal(
        sources[0],
        corrected_caches[engine.loop_results[0][0]]["local_points"],
    )


def test_optimizer_edges_are_reaccumulated_from_identity(engine, caches):
    engine.loop_constraints = [identity_loop_constraint()]
    engine.loop_optimizer.optimize = lambda edges, constraints: edges

    optimized_abs = engine.optimize_trajectory(caches)

    assert_sim3_lists_close(
        optimized_abs,
        [cache["sim3_abs"] for cache in caches],
    )
```

- [ ] **Step 2: 运行回环引擎测试并确认 RED**

Run:

```bash
pytest -q tests/test_loop_closure_pipeline.py
```

Expected: 当前引擎仍消费模糊 `sim3` 字段、无回环仍进入优化流程，返回长度为
`N - 1` 的边而不是长度为 `N` 的绝对变换。

- [ ] **Step 3: 重构回环处理和优化输出**

`run()` 的顺序固定为：

```python
def run(self, raw_predictions):
    self._ensure_image_manifest()
    self._validate_cache_count(raw_predictions)
    self.get_loop_pairs()
    initial_absolute = [
        prediction["sim3_abs"]
        for prediction in raw_predictions
    ]
    if not self.loop_list:
        return initial_absolute
    loop_constraints = self.process_loops(raw_predictions)
    if not loop_constraints:
        return initial_absolute
    sequential_edges = [
        prediction["sim3_edge"]
        for prediction in raw_predictions[1:]
    ]
    optimized_edges = self.loop_optimizer.optimize(
        sequential_edges,
        loop_constraints,
    )
    if len(optimized_edges) != len(sequential_edges):
        raise ValueError(
            "optimized edge count does not match sequential edge count"
        )
    return accumulate_edges_from_identity(optimized_edges)
```

回环两侧 mask 都调用 Task 2 的公共函数。回环测量继续调用现有
`compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))`，并断言结果方向为
`L_B compose inverse(L_A)`。

- [ ] **Step 4: 运行回环流水线测试并确认 GREEN**

Run:

```bash
pytest -q tests/test_loop_closure_pipeline.py
```

Expected: PASS。

- [ ] **Step 5: 提交 Task 4**

```bash
git add tests/test_loop_closure_pipeline.py loop_closure/loop_closure.py
git commit -m "fix: optimize corrected loop closure constraints"
```

### Task 5: 增量聚合与入口接线

**Files:**
- Modify: `inference_engine/streaming_window_engine_lc.py:139-173`
- Modify: `demo_lc.py:160-226`
- Modify: `tests/test_streaming_window_engine_lc_pipeline.py`
- Modify: `tests/test_demo_lc.py`

**Interfaces:**
- Consumes: `StreamingWindowEngineLC.aggregate_caches(parsed_caches, optimized_abs)`
- Requires: `len(parsed_caches) == len(optimized_abs)`
- Applies: `D_i = optimized_abs[i] compose inverse(cache["sim3_abs"])`

- [ ] **Step 1: 写增量只应用一次和入口转发失败测试**

```python
def test_aggregate_applies_optimization_delta_without_reapplying_mask():
    cache = corrected_cache(
        local_depth=6.0,
        absolute_scale=2.0,
        stored_mask=3.0,
    )
    optimized_abs = [
        (4.0, torch.eye(3), torch.zeros(3)),
    ]

    result = StreamingWindowEngineLC.aggregate_caches(
        [cache],
        optimized_abs,
    )

    assert torch.all(result["local_points"] == 12.0)
    assert torch.all(cache["local_points"] == 6.0)


def test_demo_passes_one_manifest_and_ratio_to_both_engines(
    monkeypatch,
    tmp_path,
):
    captured = {}
    args = parsed_args(
        tmp_path,
        registration_top_confidence_ratio=0.3,
    )

    build_loop_closure_engine(
        config={},
        args=args,
        pi3_model=object(),
        image_paths=["frame1.png"],
        cache_path_lc=tmp_path / "cache_lc",
        capture=captured,
    )

    assert captured["image_paths"] == ["frame1.png"]
    assert captured["registration_top_confidence_ratio"] == 0.3
```

- [ ] **Step 2: 运行聚合与入口测试并确认 RED**

Run:

```bash
pytest -q tests/test_streaming_window_engine_lc_pipeline.py tests/test_demo_lc.py
```

Expected: `aggregate_caches` 不接受优化绝对变换，且入口仍重写二次缓存。

- [ ] **Step 3: 实现增量聚合**

```python
@staticmethod
def aggregate_caches(parsed_caches, optimized_abs=None):
    if optimized_abs is None:
        optimized_abs = [
            cache["sim3_abs"]
            for cache in parsed_caches
        ]
    if len(parsed_caches) != len(optimized_abs):
        raise ValueError(
            "optimized transform count does not match cache count"
        )

    adjusted_caches = []
    for cache, optimized in zip(parsed_caches, optimized_abs):
        adjusted = dict(cache)
        original_inverse = closed_form_inverse_sim3(
            *cache["sim3_abs"]
        )
        delta = accumulate_sim3(optimized, original_inverse)
        delta_scale, delta_rotation, delta_translation = delta
        adjusted["local_points"] = (
            delta_scale * cache["local_points"]
        )
        adjusted["camera_poses"] = apply_sim3_to_pose(
            cache["camera_poses"],
            delta_scale,
            delta_rotation,
            delta_translation,
        )
        adjusted_caches.append(adjusted)

    return StreamingWindowEngine.aggregate_caches(adjusted_caches)
```

`demo_lc.py` 直接读取原始窗口缓存，调用 `LoopClosureEngine.run()` 得到长度为
窗口数的 `optimized_abs`，删除创建、保存、重读 `_lc` 二次缓存的逻辑，最后调用：

```python
ret_dict = StreamingWindowEngineLC._post_process_pred(
    StreamingWindowEngineLC.aggregate_caches(
        parsed_caches,
        optimized_abs,
    )
)
```

- [ ] **Step 4: 运行入口、聚合和回环测试并确认 GREEN**

Run:

```bash
pytest -q tests/test_streaming_window_engine_lc_pipeline.py tests/test_loop_closure_pipeline.py tests/test_demo_lc.py
```

Expected: PASS。

- [ ] **Step 5: 提交 Task 5**

```bash
git add inference_engine/streaming_window_engine_lc.py demo_lc.py tests/test_streaming_window_engine_lc_pipeline.py tests/test_demo_lc.py
git commit -m "fix: apply loop optimization delta once"
```

### Task 6: 全量验证、说明和云端分支

**Files:**
- Create: `docs/loop-closure-cloud-validation.md`

**Interfaces:**
- Documents: 云端克隆、依赖安装、Cython 编译、无回环对照、有效回环运行命令
- Verifies: 工作树只包含预期代码、测试和文档改动

- [ ] **Step 1: 写云端验证文档**

文档必须包含以下实际命令：

```bash
git clone --branch codex/loop-closure-corrected-pipeline --single-branch https://github.com/Cjuicy/LASER.git
cd LASER
pip install -r requirements.txt
python setup.py build_ext --inplace
pytest -q
python demo_lc.py \
  --config_path configs/loop_config.yaml \
  --model_ckpt weights/model.safetensors \
  --data_path /path/to/sequence \
  --cache_path ./inference_cache \
  --output_path ./viser_results \
  --window_size 10 \
  --overlap 5 \
  --depth_refine \
  --registration_top_confidence_ratio 0.3
```

同时记录：无回环对照必须保持图像、采样、窗口、overlap、分割方法、模型权重和
随机环境一致。

- [ ] **Step 2: 运行 Cython 构建和完整测试**

Run:

```bash
python setup.py build_ext --inplace
pytest -q
```

Expected: 两个扩展构建成功，完整测试全部通过。

- [ ] **Step 3: 检查差异范围**

Run:

```bash
git diff --check
git status --short
```

Expected: 没有空白错误；只有计划内代码、测试和文档改动。本地生成的
`inference_engine/utils/_segmentation_cy.cpp` 和
`inference_engine/utils/fast_seg.cpp` 不得暂存。

- [ ] **Step 4: 提交验证文档**

```bash
git add docs/loop-closure-cloud-validation.md
git commit -m "docs: add loop closure cloud validation"
```

- [ ] **Step 5: 推送实现分支**

```bash
git push -u origin codex/loop-closure-corrected-pipeline
```

Expected: 远端分支创建成功，可通过文档中的单分支克隆命令获取。
