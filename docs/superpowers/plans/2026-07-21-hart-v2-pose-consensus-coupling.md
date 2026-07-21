# HART v2 Pose Consensus Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 HART 区域尺度分解为当前窗口公共尺度和局部残差，使公共尺度立即进入普通与 LC 引擎的当前窗口 Sim(3)，并让下一窗口只使用主共识组支持区域注册。

**Architecture:** 保留现有 Anchor Cell–Leaf–Parent、TrackSegment、IRLS 和层次尺度共识；新增纯 NumPy Pose Consensus 模块，从严格多数高置信支持中产生 `window_scale`、`local_residual_mask` 和 `pose_support_mask`。引擎先做粗注册，再运行 HART，随后以 `s_final=s_coarse×g` 从原始 camera poses 重新求解 `R,t` 并原子提交 final Base/pose 状态；普通注册按需使用 final Base×residual，LC pairwise 注册按需使用 raw×residual。

**Tech Stack:** Python 3.13（本地验证环境）、NumPy、PyTorch、pytest、现有 Cython segmentation extensions、现有 streaming/loop-closure Sim(3) 接口。

## Global Constraints

- 实现基线固定为 `origin/codex/unified-hart-anchor-propagation@e776d72`，最终覆盖发布到 `origin/codex/unified-hart-anchor-propagation`。
- 不修改四种分割、merge、split、Track、IoU/包含率对应、IRLS 或 legacy IoU 权重公式。
- Pose Consensus 分组只复用 `scale_consistency_thresh`；不增加法向、RGB、距离、语义、光流或复合权重。
- 只有覆盖 overlap 全部双侧高置信有效深度像素严格超过 50% 的唯一尺度组才能产生 `window_scale != 1`。
- `pixel_count` 只选择主尺度组；`window_scale` 必须是主组 TrackSegment 尺度的普通中位数。
- `direct/leaf_fill/parent_fill` 使用 `regional_scale/window_scale`；`no_anchor/conflict` 的 local residual 固定为 1。
- Pose support 只包含主组中的 `direct` TrackSegment，不包含 Leaf/Parent fill。
- Pose support 与 mutual-confidence 交集少于 `anchor_min_pixels` 时必须回退 final Base，不能回退全部 refined geometry。
- 普通引擎当前窗口必须从 raw poses 以 final scale 重求 `R,t`；LC 必须把公共尺度写入当前 final pairwise Sim(3)。
- LC raw cache 不被覆盖，aggregation 只应用一次 optimized global 和一次 local residual。
- `none` 与 `legacy_iou` 行为、字段和测试保持不变。
- 现有正式 ATE/RPE 的 `align=True, correct_scale=True` 定义保持不变；文档必须解释统一尺度变化可能不可见。
- 状态只保存 overlap tail，不保存第二份三通道 propagated point map。

---

## File Structure

- Create `inference_engine/anchor_propagation/pose_consensus.py`: 严格多数尺度组选择、regional/local 分解和 pose-support mask 构造。
- Create `tests/test_hart_pose_consensus.py`: Pose Consensus 的独立数值与失败边界测试。
- Modify `inference_engine/anchor_propagation/types.py`: 用 final Base、local residual 和三通道输出定义替换旧 HART v1 状态。
- Modify `inference_engine/anchor_propagation/anchors.py`: 统计 overlap 全部双侧高置信有效深度像素。
- Modify `inference_engine/anchor_propagation/consensus.py`: 保留 regional mask/status 输出，重命名区域尺度 diagnostics。
- Modify `inference_engine/anchor_propagation/hart.py`: 编排 Anchor HART、Pose Consensus 和 residual decomposition。
- Modify `inference_engine/anchor_propagation/__init__.py`: 导出 v2 类型和纯函数。
- Modify `inference_engine/inference_utils.py`: 新增给定 final scale 重求相邻窗口 `R,t` 的单一入口。
- Modify `inference_engine/streaming_window_engine.py`: 纯 HART 计算、注册输入选择、final state 提交与当前窗口两阶段耦合。
- Modify `inference_engine/streaming_window_engine_lc.py`: pairwise/cumulative 两阶段耦合和 raw×residual 注册。
- Modify `tests/test_hart_anchors.py`: 有效 pose-consensus 分母计数。
- Modify `tests/test_hart_consensus.py`: regional diagnostics 语义。
- Modify `tests/test_hart_propagator.py`: v2 结果、残差和 state tail。
- Modify `tests/test_hart_streaming_integration.py`: 当前窗口 camera coupling、pose support、fallback、普通与 LC 坐标系。
- Modify `scripts/verify_anchor_propagation.py`: 四分割 smoke 输出公共尺度、残差和 support。
- Modify `tests/test_hart_smoke_script.py`: v2 smoke contract。
- Modify `README.md`: HART v2 双输出和云端文档入口。
- Modify `docs/hart-anchor-propagation-cloud-validation.md`: 克隆、构建、单测、运行、指标和诊断命令。

---

### Task 1: Pose Consensus Core and v2 Types

**Files:**
- Create: `tests/test_hart_pose_consensus.py`
- Create: `inference_engine/anchor_propagation/pose_consensus.py`
- Modify: `inference_engine/anchor_propagation/types.py`
- Modify: `inference_engine/anchor_propagation/__init__.py`

**Interfaces:**
- Consumes: `DirectAnchor`, `TrackWindow`, status constants and `complete_link_groups()`.
- Produces: `PoseConsensus`, `select_pose_consensus(...)`, `decompose_regional_scales(...)`, v2 `RegistrationState`, `AnchorPropagationState`, and `PropagationResult`.

- [ ] **Step 1: Write failing Pose Consensus tests**

Create tests covering strict majority, exactly 50%, conflict/no-candidate denominator behavior, unweighted median, unresolved residual and direct-only support. Use this concrete API:

```python
def test_strict_majority_selects_unweighted_track_median():
    anchors = (
        _anchor(segment=1, scale=1.03, pixels=30),
        _anchor(segment=2, scale=1.05, pixels=31),
        _anchor(segment=3, scale=0.80, pixels=10),
    )
    result = select_pose_consensus(
        anchors,
        {1: 1.03, 2: 1.05, 3: 0.80},
        valid_pixel_count=100,
        threshold=0.05,
    )
    assert result.accepted
    assert result.window_scale == pytest.approx(1.04)
    assert result.segment_ids == frozenset({1, 2})
    assert result.support_pixels == 61
    assert result.support_ratio == pytest.approx(0.61)


def test_exactly_half_support_is_rejected():
    result = select_pose_consensus(
        (_anchor(1, 1.02, 50),),
        {1: 1.02},
        valid_pixel_count=100,
        threshold=0.05,
    )
    assert not result.accepted
    assert result.window_scale == 1.0
    assert result.segment_ids == frozenset()


def test_unresolved_pixels_do_not_cancel_window_scale():
    regional = np.asarray([[[1.04, 1.04, 0.91, 1.0, 1.0]]])
    statuses = np.asarray([[[STATUS_DIRECT, STATUS_LEAF_FILL,
                             STATUS_DIRECT, STATUS_NO_ANCHOR,
                             STATUS_CONFLICT]]])
    segment_maps = np.asarray([[[1, 1, 2, 3, 4]]])
    residual, support = decompose_regional_scales(
        regional,
        statuses,
        segment_maps,
        window_scale=1.04,
        pose_segment_ids=frozenset({1}),
    )
    np.testing.assert_allclose(
        residual,
        [[[1.0, 1.0, 0.91 / 1.04, 1.0, 1.0]]],
    )
    np.testing.assert_array_equal(
        support,
        [[[True, False, False, False, False]]],
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_hart_pose_consensus.py -q`

Expected: collection fails because `pose_consensus` and v2 result types do not exist.

- [ ] **Step 3: Implement v2 immutable records**

Replace HART v1 state fields with the exact interfaces:

```python
@dataclass(frozen=True)
class RegistrationState:
    final_base_points_tail: torch.Tensor
    final_base_poses_tail: torch.Tensor
    pose_support_mask_tail: np.ndarray
    cumulative_sim3: tuple[Any, Any, Any] | None = None


@dataclass(frozen=True)
class AnchorPropagationState:
    local_residual_tail: np.ndarray
    confidence_tail: np.ndarray
    segments_tail: tuple[SegmentationFrame, ...]


@dataclass(frozen=True)
class PropagationResult:
    window_scale: float
    local_residual_mask: np.ndarray
    pose_support_mask: np.ndarray
    next_state: AnchorPropagationState
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PoseConsensus:
    window_scale: float
    segment_ids: frozenset[int]
    group_count: int
    support_pixels: int
    valid_pixels: int
    support_ratio: float
    accepted: bool
```

- [ ] **Step 4: Implement strict-majority selection and decomposition**

Implement `select_pose_consensus()` by sorting `(segment_id, scale)`, grouping scales with `complete_link_groups()`, summing `DirectAnchor.pixel_count` per segment, requiring `support_pixels * 2 > valid_pixel_count`, and taking `np.median()` over segment scales. Implement `decompose_regional_scales()` with resolved status membership and `STATUS_DIRECT`-only support. Validate shapes, positivity and finiteness before returning `float32` residual and bool support.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_hart_pose_consensus.py tests/test_hart_consensus.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add inference_engine/anchor_propagation/pose_consensus.py \
  inference_engine/anchor_propagation/types.py \
  inference_engine/anchor_propagation/__init__.py \
  tests/test_hart_pose_consensus.py
git commit -m "feat: add HART pose consensus core"
```

---

### Task 2: HART Orchestration Produces Window Scale and Residual

**Files:**
- Modify: `tests/test_hart_anchors.py`
- Modify: `tests/test_hart_propagator.py`
- Modify: `tests/test_hart_consensus.py`
- Modify: `inference_engine/anchor_propagation/anchors.py`
- Modify: `inference_engine/anchor_propagation/consensus.py`
- Modify: `inference_engine/anchor_propagation/hart.py`

**Interfaces:**
- Consumes: Task 1 `select_pose_consensus()` and `decompose_regional_scales()`.
- Produces: `HartAnchorPropagator.refine(...) -> PropagationResult` with pure v2 outputs and a residual-only tail state.

- [ ] **Step 1: Write failing evidence-count and propagator tests**

Add a direct-anchor test asserting diagnostics count all mutual high-confidence finite positive pixels, including pixels whose relation does not become a direct anchor. Update propagator tests to assert:

```python
assert result.window_scale == pytest.approx(2.0)
np.testing.assert_allclose(result.local_residual_mask, 1.0)
assert result.pose_support_mask.dtype == np.bool_
assert result.pose_support_mask.all()
np.testing.assert_allclose(result.next_state.local_residual_tail, 1.0)
```

Add a mixed-status test where a majority direct region has regional scale 2, a local outlier has 3, and a no-anchor cell stays residual 1 instead of 0.5.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_hart_anchors.py tests/test_hart_propagator.py -q`

Expected: failures reference missing v2 fields and missing `pose_consensus_valid_pixels` diagnostics.

- [ ] **Step 3: Count the Pose Consensus denominator at the anchor boundary**

In each overlap frame, after computing `prev_high`, `cur_high`, and `valid_depth`, add:

```python
mutual_valid = prev_high & cur_high & valid_depth
diagnostics["pose_consensus_valid_pixels"] += int(
    np.count_nonzero(mutual_valid)
)
```

Continue using relation-specific `core` for direct anchors; do not change IRLS inputs.

- [ ] **Step 4: Preserve regional status output and rename diagnostics**

Keep `build_scale_mask()` returning `(regional_scale_mask, status_maps, diagnostics)`. Rename `scale_mask_min/median/max` to `regional_scale_min/median/max` so the public diagnostics cannot be confused with residual output.

- [ ] **Step 5: Orchestrate Pose Consensus and residual decomposition**

After `aggregate_segment_scales()` and `build_scale_mask()`:

```python
pose = select_pose_consensus(
    direct_anchors,
    segment_scales,
    valid_pixel_count=int(
        anchor_diagnostics.get("pose_consensus_valid_pixels", 0)
    ),
    threshold=self.scale_consistency_thresh,
)
segment_maps = np.stack(
    [
        segment_ids[frame.anchor_labels]
        for frame, segment_ids in zip(
            current_segments.frames,
            anchor_tracks.segment_ids,
            strict=True,
        )
    ]
)
residual, pose_support = decompose_regional_scales(
    regional_scale_mask,
    status_maps,
    segment_maps,
    window_scale=pose.window_scale,
    pose_segment_ids=pose.segment_ids,
)
```

For the first window, initialize `regional=1`, `status=STATUS_NO_ANCHOR`, `window_scale=1`, `residual=1`, and bool support false. Save only `residual[-overlap:]` in `AnchorPropagationState`.

- [ ] **Step 6: Add Pose Consensus diagnostics**

Emit `window_scale`, group count, selected segment count, support/valid pixels, support ratio, accepted flag, support pixel ratio and residual min/median/max. Keep runtime diagnostics and existing anchor/track counters.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_hart_anchors.py tests/test_hart_consensus.py tests/test_hart_pose_consensus.py tests/test_hart_propagator.py -q`

Expected: all focused tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add inference_engine/anchor_propagation/anchors.py \
  inference_engine/anchor_propagation/consensus.py \
  inference_engine/anchor_propagation/hart.py \
  tests/test_hart_anchors.py \
  tests/test_hart_consensus.py \
  tests/test_hart_propagator.py
git commit -m "feat: decompose HART window and local scales"
```

---

### Task 3: Fixed-scale Pose Solver and Registration Support Selection

**Files:**
- Modify: `inference_engine/inference_utils.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `tests/test_hart_streaming_integration.py`

**Interfaces:**
- Produces: `register_adjacent_window_pose(src_cam_overlap, tgt_cam_overlap, scale, register_func=...) -> tuple[Tensor, Tensor]`.
- Produces: `StreamingWindowEngine._select_hart_registration_input(fallback_points, mutual_mask) -> tuple[Tensor, Tensor, dict]`.
- Produces: `StreamingWindowEngine._commit_hart_state(result, final_base_points, final_base_poses, cumulative_sim3=None) -> None`.

- [ ] **Step 1: Write failing helper tests**

Add tests proving the fixed-scale helper calls the camera registration function in the same direction as `register_adjacent_windows()`, and proving support selection behavior:

```python
points, mask, diagnostics = engine._select_hart_registration_input(
    fallback_points,
    mutual_mask,
)
np.testing.assert_array_equal(points[..., -1], expected_depth)
np.testing.assert_array_equal(mask, expected_mask)
assert diagnostics["registration_pose_support_used"] is expected_used
```

Cover both `candidate_count >= anchor_min_pixels` and fallback. In the accepted case expected points are `fallback_points × previous residual`; in fallback expected points are unchanged Base.

- [ ] **Step 2: Run helper tests and verify RED**

Run: `python -m pytest tests/test_hart_streaming_integration.py -k 'fixed_scale or registration_support' -q`

Expected: helper attributes/functions do not exist.

- [ ] **Step 3: Add the fixed-scale pose entrypoint**

Implement:

```python
def register_adjacent_window_pose(
    src_cam_overlap,
    tgt_cam_overlap,
    scale,
    register_func=register_camera_poses_kabsch_pytorch,
):
    scale_tensor = torch.as_tensor(scale)
    if not bool(torch.isfinite(scale_tensor)):
        raise ValueError("registration scale must be finite")
    if float(scale_tensor) <= 0:
        raise ValueError("registration scale must be positive")
    return register_func(tgt_cam_overlap, src_cam_overlap, scale=scale)
```

Make `register_adjacent_windows()` call this function after IRLS so coarse and final pose solve use one direction definition.

- [ ] **Step 4: Add pure HART run, input selection and state commit methods**

`_run_hart_propagation()` only builds segments and returns the propagator result. `_select_hart_registration_input()` converts residual/support tails to the fallback tensor device, intersects support with mutual confidence, checks `anchor_min_pixels`, and returns selected points/mask/diagnostics. `_commit_hart_state()` validates tail shapes and creates v2 `RegistrationState` only after final Sim(3) is available.

- [ ] **Step 5: Run helper tests and verify GREEN**

Run: `python -m pytest tests/test_hart_streaming_integration.py -k 'fixed_scale or registration_support' -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add inference_engine/inference_utils.py \
  inference_engine/streaming_window_engine.py \
  tests/test_hart_streaming_integration.py
git commit -m "refactor: separate HART registration state"
```

---

### Task 4: Immediate Coupling in the Ordinary Streaming Engine

**Files:**
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `tests/test_hart_streaming_integration.py`

**Interfaces:**
- Consumes: Task 2 `PropagationResult` and Task 3 helpers.
- Produces: final current-window points/poses, HART diagnostics, and v2 tail state.

- [ ] **Step 1: Replace old future-feedback tests with failing current-window tests**

Create a fake propagator returning `window_scale=2`, residual ones, and a support mask. Assert on the second window, not the third:

```python
np.testing.assert_array_equal(
    engine.prev_window_cache["camera_poses"][:, 0, 3],
    [2.0, 4.0],
)
assert (
    engine.prev_window_cache["hart_diagnostics"]["final_registration_scale"]
    == 2.0
)
```

Capture the fixed-scale pose solver and assert it receives `s_coarse * g`. Add an output identity test proving `final_base * residual == coarse_base * regional` on resolved pixels and public scale remains on unresolved pixels.

- [ ] **Step 2: Run ordinary integration tests and verify RED**

Run: `python -m pytest tests/test_hart_streaming_integration.py -k 'ordinary or current_window or unresolved' -q`

Expected: current camera poses still use coarse scale and old result fields fail.

- [ ] **Step 3: Implement the ordinary HART two-stage flow**

Keep unprojected current points and raw camera poses unchanged until final solve. For non-first HART windows:

```python
s_coarse, _, _ = register_adjacent_windows(
    previous_points,
    raw_points[:self.overlap],
    previous_final_poses,
    raw_poses[:self.overlap],
    registration_mask,
)
coarse_base = s_coarse * raw_points
result = self._run_hart_propagation(
    coarse_base,
    working_window["conf"],
    working_window.get("images"),
)
s_final = s_coarse * result.window_scale
R_final, t_final = register_adjacent_window_pose(
    previous_final_poses,
    raw_poses[:self.overlap],
    s_final,
)
final_base = s_final * raw_points
final_poses = apply_sim3_to_pose(raw_poses, s_final, R_final, t_final)
```

Convert `local_residual_mask` once, write `working_window['local_points'] = final_base * residual`, write final poses, merge diagnostics, and atomically commit state. First-window HART uses identity global scale and commits false support.

- [ ] **Step 4: Remove propagated-point semantics**

Delete all `propagated_points_tail` and `points_for_registration` references. Ordinary next registration uses Task 3 support selection over final Base and residual.

- [ ] **Step 5: Run ordinary and legacy integration tests**

Run:

```bash
python -m pytest tests/test_hart_streaming_integration.py -k 'not lc_' -q
python -m pytest tests/test_segmentation_engine_modes.py tests/test_demo.py -q
```

Expected: all ordinary HART, none, legacy and CLI tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add inference_engine/streaming_window_engine.py \
  tests/test_hart_streaming_integration.py
git commit -m "feat: couple HART scale to current window poses"
```

---

### Task 5: Immediate Pairwise Coupling in the Loop-closure Engine

**Files:**
- Modify: `inference_engine/streaming_window_engine_lc.py`
- Modify: `tests/test_hart_streaming_integration.py`

**Interfaces:**
- Consumes: base-engine support selection with LC raw fallback points.
- Produces: final pairwise `sim3`, final cumulative Base state, raw cache plus `local_residual_mask`.

- [ ] **Step 1: Write failing LC coordinate-frame tests**

Add tests that return `g=1.5` after a coarse pairwise scale of 2 and assert:

```python
assert engine.prev_window_cache["sim3"][0] == pytest.approx(3.0)
assert engine.registration_state.cumulative_sim3[0] == pytest.approx(3.0)
np.testing.assert_array_equal(
    engine.prev_window_cache["local_points"][..., -1],
    3.0,
)
```

On the next window capture registration source points and prove accepted support uses previous `raw × residual`, while fallback uses previous raw, never cumulative Base. Update aggregation tests to use `local_residual_mask` and prove optimized global×residual×raw exactly once.

- [ ] **Step 2: Run LC tests and verify RED**

Run: `python -m pytest tests/test_hart_streaming_integration.py -k 'lc_' -q`

Expected: pairwise scale remains coarse and old `local_scale_mask` semantics fail.

- [ ] **Step 3: Implement LC two-stage pairwise flow**

Use previous raw cache tail as `fallback_points` for support selection. Compute coarse pairwise and coarse cumulative only for HART evidence. After HART:

```python
final_pair_scale = coarse_pair_scale * result.window_scale
R_final, t_final = register_adjacent_window_pose(
    previous_raw_poses,
    raw_poses[:self.overlap],
    final_pair_scale,
)
final_pairwise = final_pair_scale, R_final, t_final
final_cumulative = accumulate_sim3(
    self.registration_state.cumulative_sim3,
    final_pairwise,
)
final_base = final_cumulative[0] * raw_points
final_base_poses = apply_sim3_to_pose(raw_poses, *final_cumulative)
```

Save `working_window['sim3'] = final_pairwise`, preserve raw points/poses, save `local_residual_mask`, and commit final cumulative state.

- [ ] **Step 4: Update LC aggregation field semantics**

Replace HART cache key handling with:

```python
if "local_residual_mask" in cache:
    cache["local_points"] = (
        s_d
        * cache.pop("local_residual_mask")
        * cache.pop("local_points")
    )
```

Keep legacy `scale_mask` branch byte-for-byte equivalent.

- [ ] **Step 5: Run LC and loop-related tests**

Run: `python -m pytest tests/test_hart_streaming_integration.py tests/test_demo_lc.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add inference_engine/streaming_window_engine_lc.py \
  tests/test_hart_streaming_integration.py
git commit -m "feat: couple HART scale to LC pairwise poses"
```

---

### Task 6: Smoke Script, Diagnostics, and Cloud Documentation

**Files:**
- Modify: `scripts/verify_anchor_propagation.py`
- Modify: `tests/test_hart_smoke_script.py`
- Modify: `README.md`
- Modify: `docs/hart-anchor-propagation-cloud-validation.md`

**Interfaces:**
- Consumes: final v2 result/cache field names.
- Produces: deterministic four-mode CPU smoke and copy-paste cloud clone/build/test/eval commands.

- [ ] **Step 1: Write failing smoke assertions**

Update the smoke test fixture so current points are uniformly `0.8 × previous`; assert for all four segmentation modes:

```python
assert second.window_scale == pytest.approx(1.25, rel=1e-4)
np.testing.assert_allclose(second.local_residual_mask, 1.0, rtol=1e-4)
assert second.pose_support_mask.any()
```

- [ ] **Step 2: Run smoke tests and verify RED**

Run: `python -m pytest tests/test_hart_smoke_script.py -q`

Expected: old smoke accesses `local_scale_mask` and does not verify coupling outputs.

- [ ] **Step 3: Update smoke output and validation**

Print one `[PASS]` line per segmentation mode containing direct anchor count, `g`, residual min/median/max and pose-support ratio. Fail on non-finite/non-positive residual, missing direct anchors, missing support or unexpected uniform-scale decomposition.

- [ ] **Step 4: Rewrite user-facing cloud workflow**

Document exact commands for:

```bash
git clone --recursive --branch codex/unified-hart-anchor-propagation \
  https://github.com/Cjuicy/LASER.git LASER-HART-v2
cd LASER-HART-v2
conda create -n laser-hart-v2 -y python=3.11
conda activate laser-hart-v2
pip install -r requirements.txt
pip install -e viser
python setup.py build_ext --inplace
python scripts/verify_anchor_propagation.py
python -m pytest -q
```

Keep ordinary and LC inference examples, update mask name to `local_residual_mask`, document `s_coarse/g/s_final` diagnostics and explain that official ATE/RPE use global scale correction. Include commands for `none` versus `hart` output directories without changing formal evaluator semantics.

- [ ] **Step 5: Run smoke and CLI tests**

Run:

```bash
python scripts/verify_anchor_propagation.py
python -m pytest tests/test_hart_smoke_script.py tests/test_demo.py tests/test_demo_lc.py -q
```

Expected: four `[PASS]` lines and all tests pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add scripts/verify_anchor_propagation.py \
  tests/test_hart_smoke_script.py \
  README.md \
  docs/hart-anchor-propagation-cloud-validation.md
git commit -m "docs: add HART v2 cloud validation workflow"
```

---

### Task 7: Full Verification and Publication

**Files:**
- Verify all modified files.
- Do not stage generated `inference_engine/utils/*.cpp`, compiled extensions, or `build/` artifacts.

**Interfaces:**
- Produces: verified commit series and remote target branch updated with lease protection.

- [ ] **Step 1: Run the complete verification suite**

Run fresh commands:

```bash
python setup.py build_ext --inplace
python scripts/verify_anchor_propagation.py
python -m pytest -q
git diff --check
git status --short
```

Expected: four smoke passes, zero pytest failures, no diff-check errors, and only expected generated untracked artifacts outside the commit.

- [ ] **Step 2: Audit requirements against the design**

Check every design section against implementation with:

```bash
rg -n "window_scale|local_residual|pose_support" \
  inference_engine tests scripts docs README.md
rg -n "local_scale_mask|propagated_points_tail|points_for_registration" \
  inference_engine tests scripts docs README.md
```

Expected: v2 terms cover result/state/ordinary/LC/docs; obsolete HART v1 state and cache fields have no live-code hits. Any historical mention must be explicitly labeled as old behavior.

- [ ] **Step 3: Inspect the complete committed diff**

Run:

```bash
git log --oneline origin/codex/unified-hart-anchor-propagation..HEAD
git diff --stat origin/codex/unified-hart-anchor-propagation...HEAD
git diff --name-status origin/codex/unified-hart-anchor-propagation...HEAD
```

Expected: only HART v2 code, tests, design/plan and validation documentation are included.

- [ ] **Step 4: Confirm GitHub authentication and remote lease**

Run:

```bash
gh --version
gh auth status
git fetch origin codex/unified-hart-anchor-propagation
git rev-parse origin/codex/unified-hart-anchor-propagation
```

Expected: authenticated GitHub CLI and remote target still based on the inspected lineage. If the remote moved unexpectedly, inspect the new commits before publication.

- [ ] **Step 5: Push the verified implementation over the requested branch**

Run:

```bash
git push --force-with-lease \
  origin HEAD:codex/unified-hart-anchor-propagation
git ls-remote --heads origin codex/unified-hart-anchor-propagation
git rev-parse HEAD
```

Expected: the remote hash exactly matches local `HEAD`.

- [ ] **Step 6: Hand off cloud commands**

Return the exact clone/build/smoke/test and ordinary/LC inference commands from `docs/hart-anchor-propagation-cloud-validation.md`, the final commit hash, test count, generated artifact exclusions, and the ATE scale-alignment caveat.
