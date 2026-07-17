# Post-Merge Normal Watershed Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `layer_atomic` 合并之后，以法向为唯一候选边界、RGB 或归一化三维间隙为确认信号，一次性恢复 2–4 个明显独立物体区域，同时保持旧模式结果和整体效率。

**Architecture:** 新模块 `post_merge_split.py` 只处理单帧 Auto 最终标签，并返回细分标签和轻量诊断。`layer_atomic_geometry.py` 复用 Auto merge 已计算的初始原子尺度，再调用该模块；`lsa.py`、streaming engine 和 CLI 只负责把当前帧 RGB 与唯一公开阈值接入。现有 `layer_atomic`、LSA 匹配、尺度估计和传播不改。

**Tech Stack:** Python 3.11、NumPy、SciPy `ndimage`、scikit-image `watershed`、PyTorch streaming engine、pytest、Pillow。

## Global Constraints

- 基线为 `codex/unified-segmentation-methods@98cce5f9f470599aca0cf5a6614f39409d929d58`，实现分支为 `codex/auto-post-merge-split`。
- 最终新增模式名固定为 `layer_atomic_split`；原 `layer_atomic` 公共签名和逐像素输出必须保持不变。
- 新增 split 不读取任何质量权重；有效法向一律等权。
- split 固定发生在 Auto merge 之后，只处理面积足以容纳至少两个合法子区的 Auto 最终区域。
- 法向屏障固定为 `30°`；最小子区面积固定为 `max(seg_min_size, ceil(0.02 * parent_area))`。
- 每个父区域只执行一次 marker-controlled watershed，不递归，markers 和最终叶子最多 4 个。
- RGB 不能单独产生候选边界；辅助确认开启时使用 `S = G_normal * max(C_rgb, C_gap)`，关闭时使用 `S = G_normal`。
- 唯一新增公开数值参数是 `split_score_thresh`，初始默认值为 `0.10`。
- 新增布尔消融开关 `split_aux_confirmation=True`；两种状态共用同一个阈值，不新增模式或独立阈值。
- 不新增第三方依赖，不加入平面拟合、曲率、SLIC、时序投票或数据集特定规则。
- 分割阶段 CPU 时间相对 `layer_atomic` 增幅不超过 20%；端到端单窗口延迟目标不超过 5%，硬上限 10%。
- KITTI/TUM 区域数中位增长不超过 30%，单条轨迹不超过 50%；ATE 中位数恶化不超过 2%，单序列不超过 5%。

## File Map

- Create `inference_engine/utils/post_merge_split.py`: 法向边缘、marker、单次 watershed、统一评分、标签替换和诊断。
- Create `tests/test_post_merge_split.py`: 新模块的合成几何单元测试。
- Modify `inference_engine/utils/layer_atomic_geometry.py`: 提取可复用 atom metadata，并新增 `segment_point_map_layer_atomic_split`。
- Create `tests/test_layer_atomic_split_integration.py`: Auto merge 后调用顺序、RGB 布局和旧模式不变性。
- Modify `inference_engine/utils/lsa.py`: 注册新模式并向它传入 RGB、法向方法和唯一阈值。
- Modify `inference_engine/streaming_window_engine.py`: 从现有 `working_window['images']` 传入 RGB。
- Modify `demo.py`, `demo_lc.py`, `eval_launch.py`: 暴露新模式、`--split_score_thresh` 和辅助确认正反开关。
- Modify `tests/test_segmentation_modes.py`, `tests/test_segmentation_engine_modes.py`, `tests/test_demo.py`, `tests/test_demo_lc.py`, `tests/test_segmentation_smoke_script.py`: 路由和 CLI 回归。
- Modify `scripts/verify_segmentation_modes.py`: 第四种模式 CPU smoke test。
- Create `scripts/evaluate_post_merge_split.py`: 29 条诊断 trace 的阈值、碎片、耗时和 PNG 对比评估。
- Create `tests/test_post_merge_split_evaluation.py`: trace 读取和汇总测试。
- Modify `docs/unified-segmentation-cloud-validation.md`: 新模式的本地与云端验证命令。

---

### Task 1: Implement the single-frame post-merge splitter

**Files:**
- Create: `inference_engine/utils/post_merge_split.py`
- Create: `tests/test_post_merge_split.py`

**Interfaces:**
- Consumes: `point_map: np.ndarray[H,W,3]`, optional `rgb_image: np.ndarray[H,W,3]`, `auto_labels: np.ndarray[H,W]`, compact `atom_labels: np.ndarray[H,W]`, `atom_scales: np.ndarray[num_atoms]`, `seg_min_size: int`, `normal_method: str`, `split_score_thresh: float`, `split_aux_confirmation: bool`.
- Produces: `refine_auto_regions(point_map, rgb_image, auto_labels, atom_labels, atom_scales, seg_min_size, normal_method, split_score_thresh, split_aux_confirmation=True) -> tuple[np.ndarray, SplitDiagnostics]`.
- Produces: immutable `SplitDiagnostics` with `as_dict() -> dict[str, int | float]` for Task 4.

- [ ] **Step 1: Write failing tests for sign-invariant normal edges and equal-weight dispersion**

Create the test file with these first tests:

```python
import numpy as np

from inference_engine.utils import post_merge_split as pms


def test_normal_edges_ignore_normal_sign_flips():
    normals = np.zeros((6, 8, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    normals[:, 4:, 2] = -1.0
    valid = np.ones((6, 8), dtype=bool)

    edge = pms._normal_edge_map(normals, valid)

    assert np.max(edge) == 0.0


def test_normal_dispersion_is_equal_weight_and_sign_invariant():
    normals = np.asarray(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]],
        dtype=np.float64,
    )
    mask = np.ones((1, 2), dtype=bool)

    assert pms._normal_dispersion(normals, mask) == 0.0
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
python -m pytest -q tests/test_post_merge_split.py
```

Expected: collection fails with `ImportError` for `post_merge_split`.

- [ ] **Step 3: Add constants, diagnostics, normal estimation and edge helpers**

Create `post_merge_split.py` with the fixed algorithm constants and these interfaces:

```python
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .geometry import compute_normals_cross_np, compute_normals_sobel_np


NORMAL_BARRIER_RAD = np.deg2rad(30.0)
MAX_LEAVES = 4
MIN_CHILD_FRACTION = 0.02
EPS = 1e-8


@dataclass(frozen=True)
class SplitDiagnostics:
    split_parent_count: int = 0
    split_proposed_count: int = 0
    split_accepted_count: int = 0
    split_added_regions: int = 0
    split_score_mean: float = 0.0
    split_score_max: float = 0.0
    split_reject_no_markers: int = 0
    split_reject_small_child: int = 0
    split_reject_low_score: int = 0
    split_runtime_ms: float = 0.0
    split_aux_confirmation: bool = True

    def as_dict(self):
        return asdict(self)


def _normal_map(point_map, normal_method):
    if normal_method == "cross":
        normals = compute_normals_cross_np(point_map)
    elif normal_method == "sobel":
        normals = compute_normals_sobel_np(point_map)
    else:
        raise ValueError(f"Unknown normal_method: {normal_method}")
    valid = np.isfinite(point_map).all(axis=-1)
    valid &= np.isfinite(normals).all(axis=-1)
    valid &= np.linalg.norm(normals, axis=-1) > EPS
    filtered = np.stack(
        [ndimage.median_filter(normals[..., c], size=3, mode="nearest") for c in range(3)],
        axis=-1,
    )
    norm = np.linalg.norm(filtered, axis=-1, keepdims=True)
    filtered = np.divide(filtered, norm, out=np.zeros_like(filtered), where=norm > EPS)
    filtered[~valid] = 0.0
    return filtered.astype(np.float32, copy=False), valid


def _normal_edge_map(normals, valid):
    edge = np.zeros(normals.shape[:2], dtype=np.float32)
    for left, right, valid_left, valid_right, left_slice, right_slice in (
        (normals[:, :-1], normals[:, 1:], valid[:, :-1], valid[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        (normals[:-1], normals[1:], valid[:-1], valid[1:], (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pair_valid = valid_left & valid_right
        dot = np.sum(left * right, axis=-1)
        angle = np.zeros(dot.shape, dtype=np.float32)
        angle[pair_valid] = np.arccos(np.clip(np.abs(dot[pair_valid]), 0.0, 1.0))
        edge[left_slice] = np.maximum(edge[left_slice], angle)
        edge[right_slice] = np.maximum(edge[right_slice], angle)
    return edge


def _normal_dispersion(normals, mask):
    selected = normals[mask]
    if selected.size == 0:
        return np.nan
    moment = np.einsum("ni,nj->ij", selected, selected) / selected.shape[0]
    return float(1.0 - np.linalg.eigvalsh(moment)[-1])
```

- [ ] **Step 4: Run the two normal tests**

Run:

```bash
python -m pytest -q tests/test_post_merge_split.py
```

Expected: `2 passed`.

- [ ] **Step 5: Add failing tests for conservative markers and variable 2–4 leaf output**

Append tests that monkeypatch `_normal_map` so marker behavior is isolated from finite-difference estimation:

```python
def _fixture(height=24, width=32):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    labels = np.zeros((height, width), dtype=np.intp)
    atoms = np.zeros_like(labels)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    return points, labels, atoms, rgb


def test_texture_only_plane_is_not_split(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    rgb[:, 16:] = 1.0
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, np.ones(labels.shape, bool)))

    refined, stats = pms.refine_auto_regions(
        points, rgb, labels, atoms, np.asarray([1.0]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.10,
    )

    assert np.unique(refined).size == 1
    assert stats.split_accepted_count == 0


def test_gradual_twenty_degree_turn_is_not_split(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    angles = np.deg2rad(np.linspace(0.0, 20.0, labels.shape[1]))
    normals = np.zeros_like(points)
    normals[..., 0] = np.sin(angles)[None, :]
    normals[..., 2] = np.cos(angles)[None, :]
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, np.ones(labels.shape, bool)))

    refined, _ = pms.refine_auto_regions(
        points, rgb, labels, atoms, np.asarray([1.0]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.05,
    )

    assert np.unique(refined).size == 1


def test_one_pass_can_create_two_children(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    rgb[:, 16:] = 1.0
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, np.ones(labels.shape, bool)))

    refined, stats = pms.refine_auto_regions(
        points, rgb, labels, atoms, np.asarray([1.0]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.10,
    )

    assert np.unique(refined).size == 2
    assert stats.split_accepted_count == 1


def test_one_pass_never_exceeds_four_children(monkeypatch):
    points, labels, atoms, rgb = _fixture(height=30, width=40)
    normals = np.zeros_like(points)
    stripe_normals = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1]]
    for stripe, normal in enumerate(stripe_normals):
        start = stripe * 8
        normals[:, start:start + 8] = normal
        rgb[:, start:start + 8] = stripe / 4.0
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, np.ones(labels.shape, bool)))

    refined, _ = pms.refine_auto_regions(
        points, rgb, labels, atoms, np.asarray([1.0]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.05,
    )

    assert np.unique(refined).size == 4
```

- [ ] **Step 6: Implement marker extraction and one cropped watershed**

Add `_minimum_child_area`, `_markers_for_region`, and `_candidate_partition`:

```python
def _minimum_child_area(parent_area, seg_min_size):
    return max(int(seg_min_size), int(np.ceil(MIN_CHILD_FRACTION * parent_area)))


def _markers_for_region(parent_mask, valid, normal_edge, min_child_area):
    core = parent_mask & valid & (normal_edge < NORMAL_BARRIER_RAD)
    components, count = ndimage.label(core, structure=ndimage.generate_binary_structure(2, 1))
    if count == 0:
        return np.zeros(parent_mask.shape, dtype=np.int32), 0
    sizes = np.bincount(components.reshape(-1), minlength=count + 1)
    eligible = np.flatnonzero(sizes >= min_child_area)
    eligible = eligible[eligible != 0]
    order = sorted(eligible.tolist(), key=lambda label: (-int(sizes[label]), int(label)))[:MAX_LEAVES]
    markers = np.zeros(parent_mask.shape, dtype=np.int32)
    for marker_id, component_id in enumerate(order, start=1):
        markers[components == component_id] = marker_id
    return markers, len(order)


def _candidate_partition(parent_mask, valid, normal_edge, markers):
    ys, xs = np.where(parent_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_mask = parent_mask[y0:y1, x0:x1]
    crop_valid = valid[y0:y1, x0:x1]
    elevation = normal_edge[y0:y1, x0:x1].copy()
    valid_values = elevation[crop_mask & crop_valid]
    invalid_level = float(valid_values.max()) if valid_values.size else float(np.pi)
    elevation[crop_mask & ~crop_valid] = invalid_level
    candidate_crop = watershed(
        elevation,
        markers[y0:y1, x0:x1],
        mask=crop_mask,
        watershed_line=False,
    )
    candidate = np.zeros(parent_mask.shape, dtype=np.int32)
    candidate[y0:y1, x0:x1] = candidate_crop
    return candidate
```

- [ ] **Step 7: Add failing tests for RGB-or-gap confirmation, small-child rejection and scale invariance**

Append:

```python
def test_gap_can_confirm_split_when_rgb_is_flat(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    points[:, 16:, 0] += 20.0
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, np.ones(labels.shape, bool)))

    first, _ = pms.refine_auto_regions(
        points, None, labels, atoms, np.asarray([1.0]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.10,
    )
    scaled, _ = pms.refine_auto_regions(
        points * 1e6, None, labels, atoms, np.asarray([1e6]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.10,
    )

    np.testing.assert_array_equal(first, scaled)
    assert np.unique(first).size == 2


def test_aux_switch_uses_normal_only_and_skips_aux_edges(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, np.ones(labels.shape, bool)))

    with_aux, _ = pms.refine_auto_regions(
        points, rgb, labels, atoms, np.asarray([1.0]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.10,
        split_aux_confirmation=True,
    )
    monkeypatch.setattr(
        pms,
        "_edge_fields",
        lambda *args: (_ for _ in ()).throw(AssertionError("aux edges must be skipped")),
    )
    normal_only, stats = pms.refine_auto_regions(
        points, rgb, labels, atoms, np.asarray([1.0]),
        seg_min_size=20, normal_method="cross", split_score_thresh=0.10,
        split_aux_confirmation=False,
    )

    assert np.unique(with_aux).size == 1
    assert np.unique(normal_only).size == 2
    assert stats.split_aux_confirmation is False


def test_small_candidate_keeps_entire_parent(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    normals[:, -2:, 2] = 0.0
    normals[:, -2:, 0] = 1.0
    rgb[:, -2:] = 1.0
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, np.ones(labels.shape, bool)))

    refined, stats = pms.refine_auto_regions(
        points, rgb, labels, atoms, np.asarray([1.0]),
        seg_min_size=80, normal_method="cross", split_score_thresh=0.05,
    )

    assert np.unique(refined).size == 1
    assert stats.split_accepted_count == 0


def test_invalid_points_keep_full_deterministic_coverage(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    points[4:8, 14:18] = np.nan
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    valid = np.isfinite(points).all(axis=-1)
    normals[~valid] = 0.0
    rgb[:, 16:] = 1.0
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, valid))

    results = [
        pms.refine_auto_regions(
            points, rgb, labels, atoms, np.asarray([1.0]),
            seg_min_size=20, normal_method="cross", split_score_thresh=0.05,
        )[0]
        for _ in range(3)
    ]

    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])
    np.testing.assert_array_equal(np.unique(results[0]), np.arange(np.unique(results[0]).size))
    assert results[0].shape == labels.shape
    assert np.all(results[0] >= 0)
```

- [ ] **Step 8: Implement reusable RGB/gap edges, one score and label replacement**

Add vectorized right/down edge construction, robust contrast, and the public entry. Keep boundary decisions inside this module:

```python
def _edge_fields(point_map, rgb_image, atom_labels, atom_scales):
    rgb = None if rgb_image is None else np.clip(np.asarray(rgb_image, dtype=np.float32), 0.0, 1.0)
    fields = {}
    for name, a_slice, b_slice in (
        ("right", (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ("down", (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pa, pb = point_map[a_slice], point_map[b_slice]
        aa, ab = atom_labels[a_slice], atom_labels[b_slice]
        denominator = np.sqrt(atom_scales[aa] * atom_scales[ab])
        distance = np.linalg.norm(pa - pb, axis=-1)
        gap = np.divide(distance, denominator, out=np.full(distance.shape, np.nan), where=np.isfinite(denominator) & (denominator > 0))
        color = None if rgb is None else np.linalg.norm(rgb[a_slice] - rgb[b_slice], axis=-1)
        fields[name] = (gap, color)
    return fields


def _contrast(boundary_values, interior_values):
    boundary_values = boundary_values[np.isfinite(boundary_values)]
    interior_values = interior_values[np.isfinite(interior_values)]
    if boundary_values.size == 0 or interior_values.size == 0:
        return 0.0
    ratio = np.median(boundary_values) / (np.percentile(interior_values, 75) + EPS)
    return float(np.clip((ratio - 1.0) / (ratio + 1.0), 0.0, 1.0))


def _partition_score(parent_mask, child_labels, normals, valid, edge_fields):
    parent_h = _normal_dispersion(normals, parent_mask & valid)
    if not np.isfinite(parent_h) or parent_h <= EPS:
        return 0.0
    child_h = 0.0
    parent_area = int(parent_mask.sum())
    for child_id in np.unique(child_labels[parent_mask]):
        child = parent_mask & (child_labels == child_id)
        dispersion = _normal_dispersion(normals, child & valid)
        if not np.isfinite(dispersion):
            return 0.0
        child_h += child.sum() / parent_area * dispersion
    normal_gain = float(np.clip((parent_h - child_h) / (parent_h + EPS), 0.0, 1.0))
    if edge_fields is None:
        return normal_gain
    confirmations = []
    for name, a_slice, b_slice in (
        ("right", (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ("down", (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        inside = parent_mask[a_slice] & parent_mask[b_slice]
        boundary = inside & (child_labels[a_slice] != child_labels[b_slice])
        interior = inside & ~boundary
        gap, color = edge_fields[name]
        confirmations.append((gap[boundary], gap[interior], None if color is None else color[boundary], None if color is None else color[interior]))
    gap_c = _contrast(np.concatenate([item[0] for item in confirmations]), np.concatenate([item[1] for item in confirmations]))
    color_c = 0.0
    if confirmations[0][2] is not None:
        color_c = _contrast(np.concatenate([item[2] for item in confirmations]), np.concatenate([item[3] for item in confirmations]))
    return normal_gain * max(gap_c, color_c)
```

Add compact relabeling and the complete deterministic orchestration:

```python
def _compact_labels(labels):
    _, inverse = np.unique(np.asarray(labels), return_inverse=True)
    return inverse.reshape(np.asarray(labels).shape).astype(np.intp, copy=False)


def refine_auto_regions(
    point_map,
    rgb_image,
    auto_labels,
    atom_labels,
    atom_scales,
    seg_min_size,
    normal_method,
    split_score_thresh,
    split_aux_confirmation=True,
):
    started = time.perf_counter()
    point_map = np.asarray(point_map)
    auto_labels = _compact_labels(auto_labels)
    atom_labels = np.asarray(atom_labels, dtype=np.intp)
    atom_scales = np.asarray(atom_scales, dtype=np.float64)
    if point_map.shape[:2] != auto_labels.shape or atom_labels.shape != auto_labels.shape:
        raise ValueError("point_map, auto_labels, and atom_labels must share H,W")
    if not 0.0 <= split_score_thresh <= 1.0:
        raise ValueError("split_score_thresh must be in [0, 1]")

    normals, valid = _normal_map(point_map, normal_method)
    normal_edge = _normal_edge_map(normals, valid)
    edge_fields = (
        _edge_fields(point_map, rgb_image, atom_labels, atom_scales)
        if split_aux_confirmation
        else None
    )
    output = auto_labels.copy()
    next_label = int(output.max()) + 1
    proposed = accepted = added = 0
    reject_no_markers = reject_small = reject_score = 0
    scores = []

    parent_ids = np.unique(auto_labels)
    for parent_id in parent_ids:
        parent_mask = auto_labels == parent_id
        parent_area = int(parent_mask.sum())
        min_child_area = _minimum_child_area(parent_area, seg_min_size)
        if parent_area < 2 * min_child_area:
            reject_small += 1
            continue
        markers, marker_count = _markers_for_region(
            parent_mask, valid, normal_edge, min_child_area
        )
        if marker_count < 2:
            reject_no_markers += 1
            continue
        candidate = _candidate_partition(parent_mask, valid, normal_edge, markers)
        child_ids, child_sizes = np.unique(
            candidate[parent_mask], return_counts=True
        )
        if (
            child_ids.size < 2
            or child_ids.size > MAX_LEAVES
            or np.any(child_ids == 0)
            or np.any(child_sizes < min_child_area)
        ):
            reject_small += 1
            continue
        proposed += 1
        score = _partition_score(
            parent_mask, candidate, normals, valid, edge_fields
        )
        scores.append(score)
        if score < split_score_thresh:
            reject_score += 1
            continue
        for offset, child_id in enumerate(child_ids):
            output[parent_mask & (candidate == child_id)] = next_label + offset
        next_label += child_ids.size
        accepted += 1
        added += int(child_ids.size - 1)

    output = _compact_labels(output)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    diagnostics = SplitDiagnostics(
        split_parent_count=int(parent_ids.size),
        split_proposed_count=proposed,
        split_accepted_count=accepted,
        split_added_regions=added,
        split_score_mean=float(np.mean(scores)) if scores else 0.0,
        split_score_max=float(np.max(scores)) if scores else 0.0,
        split_reject_no_markers=reject_no_markers,
        split_reject_small_child=reject_small,
        split_reject_low_score=reject_score,
        split_runtime_ms=runtime_ms,
        split_aux_confirmation=bool(split_aux_confirmation),
    )
    return output, diagnostics
```

- [ ] **Step 9: Run all core tests**

Run:

```bash
python -m pytest -q tests/test_post_merge_split.py
```

Expected: all tests pass; repeated calls produce identical arrays and every pixel has exactly one non-negative compact label.

- [ ] **Step 10: Commit the core splitter**

```bash
git add inference_engine/utils/post_merge_split.py tests/test_post_merge_split.py
git commit -m "feat: add conservative post-merge splitter"
```

---

### Task 2: Reuse atom scales and add the final layer-atomic entry

**Files:**
- Modify: `inference_engine/utils/layer_atomic_geometry.py`
- Create: `tests/test_layer_atomic_split_integration.py`
- Test: `tests/test_layer_atomic_geometry.py`

**Interfaces:**
- Produces private `AtomMergeResult(labels, atom_labels, atom_scales)`.
- Produces `segment_point_map_layer_atomic_split(point_map, depth_merge_thresh, rgb_images=None, normal_method="cross", split_score_thresh=0.10, split_aux_confirmation=True, conf_map=None, top_conf_percentile=None, seg_scale=300, seg_sigma=1.1, seg_min_size=500, batch_idx=None) -> np.ndarray`.
- Preserves `merge_layer_atoms(...) -> np.ndarray` and `segment_point_map_layer_atomic(...) -> np.ndarray` exactly.

- [ ] **Step 1: Write failing integration tests for order, metadata reuse and no split weighting input**

Create tests that patch the depth stages, metadata merge and post-merge refinement:

```python
import numpy as np

from inference_engine.utils import layer_atomic_geometry as lag


def test_split_entry_runs_after_merge_and_passes_only_geometry_rgb_and_scales(monkeypatch):
    points = np.zeros((8, 10, 3), dtype=np.float32)
    rgb_batch = np.zeros((2, 3, 8, 10), dtype=np.float32)
    initial = np.zeros((8, 10), dtype=np.intp)
    coarse = np.zeros_like(initial)
    merged = np.zeros_like(initial)
    atom_scales = np.asarray([0.25], dtype=np.float64)
    calls = []

    monkeypatch.setattr(lag, "segment_depth_felzenszwalb_rag_stages", lambda *args: (initial, coarse, 0.1))
    monkeypatch.setattr(
        lag,
        "_merge_layer_atoms_with_metadata",
        lambda *args: lag.AtomMergeResult(merged, initial, atom_scales),
    )

    def fake_refine(point_map, rgb_image, auto_labels, atom_labels, scales, **kwargs):
        calls.append((point_map, rgb_image, auto_labels, atom_labels, scales, kwargs))
        return auto_labels.copy(), lag.SplitDiagnostics()

    monkeypatch.setattr(lag, "refine_auto_regions", fake_refine)

    result = lag.segment_point_map_layer_atomic_split(
        points,
        depth_merge_thresh=0.1,
        rgb_images=rgb_batch,
        conf_map=np.ones((2, 8, 10), dtype=np.float32),
        top_conf_percentile=0.5,
        seg_min_size=20,
        split_aux_confirmation=False,
        batch_idx=1,
    )

    assert result.shape == merged.shape
    assert len(calls) == 1
    _, rgb, received_auto, received_atoms, received_scales, kwargs = calls[0]
    assert rgb.shape == (8, 10, 3)
    assert received_auto is merged
    assert received_atoms is initial
    assert received_scales is atom_scales
    assert set(kwargs) == {
        "seg_min_size",
        "normal_method",
        "split_score_thresh",
        "split_aux_confirmation",
    }
    assert kwargs["split_aux_confirmation"] is False


def test_old_public_merger_matches_metadata_labels():
    point_map = np.zeros((4, 6, 3), dtype=np.float64)
    point_map[..., 0] = np.arange(6)
    initial = np.tile(np.asarray([0, 0, 1, 1, 2, 2]), (4, 1))
    coarse = np.zeros_like(initial)

    public = lag.merge_layer_atoms(point_map, initial, coarse, 0.1)
    metadata = lag._merge_layer_atoms_with_metadata(point_map, initial, coarse, 0.1)

    np.testing.assert_array_equal(public, metadata.labels)
```

- [ ] **Step 2: Run the integration tests and verify missing symbols**

Run:

```bash
python -m pytest -q tests/test_layer_atomic_split_integration.py
```

Expected: failures for `AtomMergeResult`, `_merge_layer_atoms_with_metadata`, and `segment_point_map_layer_atomic_split`.

- [ ] **Step 3: Refactor the existing merge body behind metadata without changing its public wrapper**

Add:

```python
from dataclasses import dataclass

from .post_merge_split import SplitDiagnostics, refine_auto_regions


@dataclass(frozen=True)
class AtomMergeResult:
    labels: np.ndarray
    atom_labels: np.ndarray
    atom_scales: np.ndarray
```

Rename the current `merge_layer_atoms` body to `_merge_layer_atoms_with_metadata`. Keep all validation, DSU decisions and arithmetic unchanged. Replace its final return with:

```python
    roots = np.asarray([dsu.find(atom) for atom in range(n_atoms)])
    labels = _compact_labels(roots[atom_labels])
    return AtomMergeResult(labels, atom_labels, scales)
```

Restore the public wrapper:

```python
def merge_layer_atoms(point_map, initial_labels, coarse_labels, depth_merge_thresh):
    return _merge_layer_atoms_with_metadata(
        point_map,
        initial_labels,
        coarse_labels,
        depth_merge_thresh,
    ).labels
```

- [ ] **Step 4: Add strict RGB frame selection and the final segmentation entry**

Implement:

```python
def _select_rgb_frame(rgb_images, batch_idx, height, width):
    if rgb_images is None:
        return None
    rgb = np.asarray(rgb_images)
    if rgb.ndim == 4:
        if batch_idx is None:
            raise ValueError("batch_idx is required for batched rgb_images")
        rgb = rgb[batch_idx]
    if rgb.shape == (3, height, width):
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.shape != (height, width, 3):
        raise ValueError("rgb_images must contain RGB frames aligned with point_map")
    return np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)
```

Implement `segment_point_map_layer_atomic_split` by running the same existing depth stages, calling `_merge_layer_atoms_with_metadata`, selecting the RGB frame, then calling `refine_auto_regions` with both `split_score_thresh` and `split_aux_confirmation`, and returning only the refined labels. The independent evaluation script calls `refine_auto_regions` directly when it needs `SplitDiagnostics`. Do not pass `conf_map` or any derived value into `refine_auto_regions`.

- [ ] **Step 5: Run new integration tests and the complete old atom suite**

Run:

```bash
python -m pytest -q \
  tests/test_layer_atomic_split_integration.py \
  tests/test_layer_atomic_geometry.py \
  tests/test_layer_atomic_integration.py
```

Expected: all tests pass, including the old “only unions atoms” and scale-invariance assertions.

- [ ] **Step 6: Commit atom-scale reuse and the new entry**

```bash
git add inference_engine/utils/layer_atomic_geometry.py \
  tests/test_layer_atomic_split_integration.py
git commit -m "feat: refine merged layer-atomic regions"
```

---

### Task 3: Wire RGB and the final mode through LSA, streaming, CLI and evaluation

**Files:**
- Modify: `inference_engine/utils/lsa.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `demo.py`
- Modify: `demo_lc.py`
- Modify: `eval_launch.py`
- Modify: `tests/test_segmentation_modes.py`
- Modify: `tests/test_segmentation_engine_modes.py`
- Modify: `tests/test_demo.py`
- Modify: `tests/test_demo_lc.py`

**Interfaces:**
- Extends `SEGMENT_MODES` with `layer_atomic_split`.
- Extends `make_sp_graph(..., rgb_images=None, split_score_thresh=0.10, split_aux_confirmation=True)`.
- Extends `StreamingWindowEngine(..., split_score_thresh=0.10, split_aux_confirmation=True)` and `_build_segment_graph(local_points, conf, images=None)`.
- Adds CLI `--segment_mode layer_atomic_split`, `--split_score_thresh 0.10`, and `--split_aux_confirmation` / `--no-split_aux_confirmation`.

- [ ] **Step 1: Add failing routing tests**

Extend `tests/test_segmentation_modes.py`:

```python
def test_layer_atomic_split_routes_rgb_without_changing_front_filter(monkeypatch):
    calls = _capture_route(monkeypatch)
    point_maps, conf_map = _inputs()
    rgb = np.zeros((2, 3, 4, 5), dtype=np.float32)

    graph = lsa.make_sp_graph(
        point_maps,
        conf_map=conf_map,
        top_conf_percentile=0.8,
        rgb_images=rgb,
        split_score_thresh=0.15,
        split_aux_confirmation=False,
        segment_mode="layer_atomic_split",
        normal_method="sobel",
    )

    assert graph == "shared_graph"
    assert calls["op_func"] is lsa.segment_point_map_layer_atomic_split
    assert calls["kwargs"]["rgb_images"] is rgb
    assert calls["kwargs"]["split_score_thresh"] == 0.15
    assert calls["kwargs"]["split_aux_confirmation"] is False
    assert calls["kwargs"]["normal_method"] == "sobel"
    assert calls["kwargs"]["conf_map"] is conf_map
```

Extend `tests/test_segmentation_engine_modes.py` so `layer_atomic_split` is accepted only with depth refinement, validates `normal_method`, stores both split settings, and `_build_segment_graph(local_points, conf, images)` forwards the exact RGB tensor converted to NumPy plus the auxiliary boolean.

- [ ] **Step 2: Run routing tests and verify the new mode is rejected**

Run:

```bash
python -m pytest -q tests/test_segmentation_modes.py tests/test_segmentation_engine_modes.py
```

Expected: failures because `layer_atomic_split`, RGB, the threshold, and the auxiliary switch are not registered.

- [ ] **Step 3: Register the mode and route its frame inputs in `lsa.py`**

Make these exact structural changes:

```python
from .layer_atomic_geometry import (
    segment_point_map_layer_atomic,
    segment_point_map_layer_atomic_split,
)

SEGMENT_MODES = ("depth", "geometry", "layer_atomic", "layer_atomic_split")
```

Extend `make_sp_graph` with `rgb_images=None`, `split_score_thresh=0.10`, and `split_aux_confirmation=True`. Keep the three existing branches byte-for-byte equivalent. Add a distinct fourth branch:

```python
    elif segment_mode == "layer_atomic_split":
        if normal_method not in NORMAL_METHODS:
            raise ValueError(
                f"Unknown normal_method: {normal_method!r}; expected one of {NORMAL_METHODS}."
            )
        images = point_maps
        segmentation_op = segment_point_map_layer_atomic_split
        common_kwargs.update(
            rgb_images=rgb_images,
            normal_method=normal_method,
            split_score_thresh=split_score_thresh,
            split_aux_confirmation=split_aux_confirmation,
        )
```

Do not add RGB kwargs to depth, geometry, or old layer-atomic calls.

- [ ] **Step 4: Pass existing window images from the streaming engine**

Extend the constructor with `split_score_thresh: float = 0.10` and `split_aux_confirmation: bool = True`, validate `0.0 <= split_score_thresh <= 1.0`, normalize the switch with `bool(...)`, and store both. Validate `normal_method` for both `geometry` and `layer_atomic_split`.

Change the helper to:

```python
def _build_segment_graph(self, local_points, conf, images=None):
    rgb_images = None if images is None else images.cpu().numpy()
    return make_sp_graph(
        local_points.cpu().numpy(),
        conf_map=conf.cpu().numpy(),
        top_conf_percentile=self.top_conf_percentile,
        segment_mode=self.segment_mode,
        normal_method=self.normal_method,
        geometry_seg_profile=self.geometry_seg_profile,
        rgb_images=rgb_images,
        split_score_thresh=self.split_score_thresh,
        split_aux_confirmation=self.split_aux_confirmation,
    )
```

At both `_registration_worker` call sites, pass `working_window.get('images')` as the third argument. Do not reload images from disk.

- [ ] **Step 5: Add CLI and evaluation options with unchanged defaults**

In `demo.py`, `demo_lc.py`, and `eval_launch.py`, add:

```python
parser.add_argument(
    "--split_score_thresh",
    default=0.10,
    type=float,
    help="acceptance threshold for layer_atomic_split",
)
parser.add_argument(
    "--split_aux_confirmation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="enable RGB or normalized-gap confirmation for layer_atomic_split",
)
```

Add `layer_atomic_split` to existing `segment_mode` choices, and pass both split settings to engine construction. In `eval_launch.py`, also add `--segment_mode` and `--normal_method`, and include all split settings in both streaming engine partials so KITTI/TUM ATE can compare Auto, auxiliary-on, and auxiliary-off under identical settings.

- [ ] **Step 6: Extend parser tests and rerun routing tests**

Add `layer_atomic_split` to parser parametrizations and assert the threshold default/forwarding:

```python
def test_split_threshold_defaults_and_forwards(tmp_path, monkeypatch):
    args = demo.get_args_parser().parse_args([])
    assert args.split_score_thresh == 0.10
    assert args.split_aux_confirmation is True
    disabled = demo.get_args_parser().parse_args(["--no-split_aux_confirmation"])
    assert disabled.split_aux_confirmation is False
```

Run:

```bash
python -m pytest -q \
  tests/test_segmentation_modes.py \
  tests/test_segmentation_engine_modes.py \
  tests/test_demo.py \
  tests/test_demo_lc.py
```

Expected: all tests pass and old default mode remains `depth`.

- [ ] **Step 7: Commit mode and RGB plumbing**

```bash
git add inference_engine/utils/lsa.py \
  inference_engine/streaming_window_engine.py \
  demo.py demo_lc.py eval_launch.py \
  tests/test_segmentation_modes.py \
  tests/test_segmentation_engine_modes.py \
  tests/test_demo.py tests/test_demo_lc.py
git commit -m "feat: expose layer atomic split mode"
```

---

### Task 4: Add deterministic smoke, diagnostics and trace evaluation

**Files:**
- Modify: `scripts/verify_segmentation_modes.py`
- Modify: `tests/test_segmentation_smoke_script.py`
- Create: `scripts/evaluate_post_merge_split.py`
- Create: `tests/test_post_merge_split_evaluation.py`
- Modify: `docs/unified-segmentation-cloud-validation.md`

**Interfaces:**
- Produces CLI `evaluate_post_merge_split.py --trace-glob PATTERN --thresholds VALUES --aux-states on off --repeats 30 --output-dir DIRECTORY`.
- Reads the existing NPZ keys `layer_atomic__inputs__rgb`, `layer_atomic__inputs__point_map`, `layer_atomic__segmentation__final_labels`, `layer_atomic__segmentation__initial_labels`, `layer_atomic__segmentation__atom_scales`, and optional `geometry_baseline__segmentation__final_labels`.
- Writes `summary.json`, `per_trace.json`, and deterministic PNG panels for selected traces.

- [ ] **Step 1: Extend the smoke fixture and write a failing fourth-mode assertion**

Make `make_fixture` return `[N,3,H,W]` RGB and pass it as `rgb_images` to `make_sp_graph`. Update the smoke test assertion:

```python
for mode in ("depth", "geometry", "layer_atomic", "layer_atomic_split"):
    assert f"[PASS] mode={mode} frames=2" in result.stdout
```

- [ ] **Step 2: Run the smoke test and verify the fourth line is absent**

Run:

```bash
python -m pytest -q tests/test_segmentation_smoke_script.py
```

Expected: failure for missing `[PASS] mode=layer_atomic_split`.

- [ ] **Step 3: Update the smoke script and verify exact partition coverage**

Use a deterministic RGB fixture with a uniform plane so the new mode is expected to keep one stable surface. Iterate over all `SEGMENT_MODES`, pass `rgb_images=rgb`, and retain the existing `coverage == 1` assertion.

Run:

```bash
python scripts/verify_segmentation_modes.py
```

Expected: four `[PASS]` lines and no missing/overlapping pixels.

- [ ] **Step 4: Write failing tests for trace loading and acceptance summaries**

Create a temporary NPZ with the exact split input keys:

```python
import numpy as np

from scripts import evaluate_post_merge_split as evaluation


def test_load_trace_uses_only_split_inputs(tmp_path):
    path = tmp_path / "trace.npz"
    np.savez(
        path,
        layer_atomic__inputs__rgb=np.zeros((3, 8, 10), dtype=np.float32),
        layer_atomic__inputs__point_map=np.zeros((8, 10, 3), dtype=np.float32),
        layer_atomic__segmentation__final_labels=np.zeros((8, 10), dtype=np.intp),
        layer_atomic__segmentation__initial_labels=np.zeros((8, 10), dtype=np.intp),
        layer_atomic__segmentation__atom_scales=np.ones(1, dtype=np.float64),
    )

    trace = evaluation.load_trace(path)

    assert set(trace) == {"rgb", "point_map", "auto_labels", "atom_labels", "atom_scales", "geometry_labels"}
    assert trace["geometry_labels"] is None


def test_summary_enforces_region_and_runtime_budgets():
    summary = evaluation.summarize(
        [
            {"auto_regions": 10, "split_regions": 12, "auto_ms": 100.0, "split_ms": 118.0},
            {"auto_regions": 20, "split_regions": 25, "auto_ms": 200.0, "split_ms": 235.0},
        ]
    )

    assert summary["median_region_growth"] <= 1.30
    assert summary["median_runtime_overhead"] <= 0.20
```

- [ ] **Step 5: Implement the evaluation CLI**

Use `argparse`, `glob`, `json`, `time.perf_counter`, NumPy and Pillow only. For each threshold, auxiliary state, and trace:

1. Load the six keys shown in the interface.
2. Run `refine_auto_regions` once for labels/diagnostics with the selected boolean.
3. Run `segment_point_map_layer_atomic` once per trace for the shared baseline, then run `segment_point_map_layer_atomic_split` for both auxiliary states for `repeats` timed iterations after one warm-up.
4. Record Auto/split region counts, score diagnostics, median runtime and P90.
5. If Geometry labels exist, report boundary support as a diagnostic proxy only; never treat it as ground truth.
6. Save panels for these four exact traces when present:
   - `laser-case-00-003825-004079-trace.npz`
   - `laser-case-05-002160-002414-trace.npz`
   - `laser-tum-freiburg1_360-000000-000254-trace.npz`
   - `laser-tum-freiburg1_desk-000135-000389-trace.npz`
7. At every shared threshold, report paired deltas `aux_on - aux_off` for accepted splits, region growth, runtime, and Geometry-supported boundary proxy.
8. Select `0.10` using only `aux_on`; if it violates the region/runtime budgets, select the first of `0.15`, then `0.20`, that satisfies both. Never tune a separate threshold for `aux_off`; exit non-zero if the production state has no acceptable threshold.

Expose pure `load_trace(path)` and `summarize(records)` functions so the tests do not launch subprocesses.

- [ ] **Step 6: Run evaluation-script tests and smoke regression**

Run:

```bash
python -m pytest -q \
  tests/test_post_merge_split_evaluation.py \
  tests/test_segmentation_smoke_script.py
```

Expected: all tests pass.

- [ ] **Step 7: Update cloud validation documentation**

Add the fourth mode to the method list and smoke output. Add this command next to `layer_atomic`:

```bash
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./comparison_cache/layer_atomic_split" \
    --output_path "./comparison_results/layer_atomic_split" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --depth_refine \
    --segment_mode layer_atomic_split \
    --normal_method cross \
    --split_score_thresh 0.10 \
    --split_aux_confirmation
```

Add the paired normal-only ablation command with the same method and threshold:

```bash
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./comparison_cache/layer_atomic_split_no_aux" \
    --output_path "./comparison_results/layer_atomic_split_no_aux" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --depth_refine \
    --segment_mode layer_atomic_split \
    --normal_method cross \
    --split_score_thresh 0.10 \
    --no-split_aux_confirmation
```

- [ ] **Step 8: Commit evaluation and documentation**

```bash
git add scripts/verify_segmentation_modes.py \
  scripts/evaluate_post_merge_split.py \
  tests/test_segmentation_smoke_script.py \
  tests/test_post_merge_split_evaluation.py \
  docs/unified-segmentation-cloud-validation.md
git commit -m "test: evaluate post-merge split behavior"
```

---

### Task 5: Run complete regression, 29-trace calibration and target-server acceptance

**Files:**
- Verify: all files modified in Tasks 1–4
- Modify only if selected threshold changes: `inference_engine/utils/lsa.py`, `inference_engine/streaming_window_engine.py`, `demo.py`, `demo_lc.py`, `eval_launch.py`, associated tests, and both design/validation docs.

**Interfaces:**
- Consumes Task 4 `summary.json` and pose-evaluation JSON files.
- Produces one frozen default threshold, passing test evidence, trace panels, and target-server timing/ATE evidence.

- [ ] **Step 1: Run the focused segmentation regression**

Run:

```bash
python -m pytest -q \
  tests/test_depth_segmentation_stages.py \
  tests/test_geometry_segmentation.py \
  tests/test_layer_atomic_geometry.py \
  tests/test_layer_atomic_integration.py \
  tests/test_post_merge_split.py \
  tests/test_layer_atomic_split_integration.py \
  tests/test_segmentation_engine_modes.py \
  tests/test_segmentation_modes.py \
  tests/test_segmentation_smoke_script.py \
  tests/test_post_merge_split_evaluation.py \
  tests/test_demo.py \
  tests/test_demo_lc.py
```

Expected: all tests pass; the original 54 segmentation tests remain green and the new tests add coverage without replacing them.

- [ ] **Step 2: Run the full repository suite**

Run:

```bash
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 3: Evaluate the fixed 17 KITTI and 12 TUM traces**

Run:

```bash
python scripts/evaluate_post_merge_split.py \
  --trace-glob '/private/tmp/laser-case-*-trace.npz' \
  --trace-glob '/private/tmp/laser-tum-*-trace.npz' \
  --thresholds 0.05 0.075 0.10 0.15 0.20 \
  --aux-states on off \
  --repeats 30 \
  --output-dir /private/tmp/post-merge-split-eval
```

Expected: exactly 29 traces for each auxiliary state, paired on/off deltas are present, `selected_threshold` is chosen from production `aux_on` and is one of `0.10`, `0.15`, `0.20`, median region growth is at most 1.30, every trace growth is at most 1.50, and median segmentation-stage overhead is at most 0.20.

- [ ] **Step 4: Inspect the four deterministic panels**

Inspect:

```text
/private/tmp/post-merge-split-eval/panels/laser-case-00-003825-004079-trace.png
/private/tmp/post-merge-split-eval/panels/laser-case-05-002160-002414-trace.png
/private/tmp/post-merge-split-eval/panels/laser-tum-freiburg1_360-000000-000254-trace.png
/private/tmp/post-merge-split-eval/panels/laser-tum-freiburg1_desk-000135-000389-trace.png
```

Acceptance: obvious independent objects gain coherent boundaries; continuous road/wall/table support remains connected; no Geometry-like scattered islands appear.

- [ ] **Step 5: Freeze the single threshold**

Read `/private/tmp/post-merge-split-eval/summary.json`. If `selected_threshold` is `0.10`, make no code change. If it is `0.15` or `0.20`, replace every public default and assertion with that exact selected value, update the design and validation documents, rerun Steps 1–3, and commit:

```bash
git add inference_engine/utils/lsa.py \
  inference_engine/streaming_window_engine.py \
  demo.py demo_lc.py eval_launch.py \
  tests/test_segmentation_modes.py \
  tests/test_segmentation_engine_modes.py \
  tests/test_demo.py tests/test_demo_lc.py \
  docs/superpowers/specs/2026-07-17-post-merge-normal-watershed-split-design.md \
  docs/unified-segmentation-cloud-validation.md
git commit -m "tune: freeze post-merge split threshold"
```

Do not retain per-dataset defaults or alternative implementations.

- [ ] **Step 6: Run matched KITTI pose evaluation on the target server**

Run baseline and new mode with identical seed/model/window settings:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3 \
  --eval_dataset=kitti \
  --output_dir=outputs/cam_pose/kitti_layer_atomic \
  --segment_mode layer_atomic \
  --normal_method cross

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12346 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3 \
  --eval_dataset=kitti \
  --output_dir=outputs/cam_pose/kitti_layer_atomic_split \
  --segment_mode layer_atomic_split \
  --normal_method cross \
  --split_aux_confirmation

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12349 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3 \
  --eval_dataset=kitti \
  --output_dir=outputs/cam_pose/kitti_layer_atomic_split_no_aux \
  --segment_mode layer_atomic_split \
  --normal_method cross \
  --no-split_aux_confirmation
```

Expected: report both `aux_on - layer_atomic` and `aux_off - layer_atomic`; production `aux_on` ATE median degradation is at most 2%, no sequence degrades more than 5%, and end-to-end median window latency overhead is at most 10%. The off run uses the same frozen threshold and is diagnostic only.

- [ ] **Step 7: Run matched TUM pose evaluation including both indoor scenes**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12347 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3 \
  --eval_dataset=tum \
  --output_dir=outputs/cam_pose/tum_layer_atomic \
  --segment_mode layer_atomic \
  --normal_method cross

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12348 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3 \
  --eval_dataset=tum \
  --output_dir=outputs/cam_pose/tum_layer_atomic_split \
  --segment_mode layer_atomic_split \
  --normal_method cross \
  --split_aux_confirmation

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12350 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3 \
  --eval_dataset=tum \
  --output_dir=outputs/cam_pose/tum_layer_atomic_split_no_aux \
  --segment_mode layer_atomic_split \
  --normal_method cross \
  --no-split_aux_confirmation
```

Confirm all three runs include `freiburg1_360` and `freiburg1_desk`. Apply the same production guards to `aux_on`, and report the same-threshold `aux_off` deltas without tuning it separately.

- [ ] **Step 8: Perform final repository hygiene and diff review**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -6
```

Expected: no whitespace errors; only the previously generated `_segmentation_cy.cpp` and `fast_seg.cpp` may remain untracked; no plane, curvature, recursive split, SLIC, temporal split, quality-weight, or alternate-method code is present.

- [ ] **Step 9: Commit final verification documentation if evidence files were added**

Only repository documentation belongs in Git; keep large NPZ, cache, PNG and trajectory outputs outside the repository.

```bash
git add docs/unified-segmentation-cloud-validation.md
git commit -m "docs: record post-merge split validation"
```

Expected: skip this commit when the documentation has no evidence update.
