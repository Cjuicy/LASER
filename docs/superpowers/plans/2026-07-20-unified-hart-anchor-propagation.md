# Unified HART-AP Anchor Propagation Implementation Plan

**Goal:** 从 `codex/auto-post-merge-split` 的设计提交创建独立分支，一次性实现 HART-AP 旁路，使 `depth / geometry / layer_atomic / layer_atomic_split` 四种分割均可选择 `none / legacy_iou / hart`，完成本地回归、普通与 LC 引擎接入、验证文档、最终提交并推送到 `origin/codex/unified-hart-anchor-propagation`。

**Architecture:** 保留 `utils/depth.py` 与 `utils/lsa.py` 的 legacy 图和 IoU 加权传播不变；新增 `inference_engine/anchor_propagation/`，把四种分割适配为严格嵌套的 Anchor Cell–Leaf–Parent 层次，以稀疏 label-pair 对应构建 Leaf/Anchor TrackSegments，只在双侧高置信核心中运行现有 IRLS，再用中位数和 complete-link 对数尺度共识生成一通道局部 mask。HART 路径逻辑分离 Base/Refined 状态；普通引擎立即输出 refined points，LC 引擎保存纯局部残差 mask 并在优化后的全局 Sim(3) 下应用一次。

**Tech Stack:** Python 3.11、NumPy、PyTorch、现有 SciPy/scikit-image 分割依赖、pytest、现有流式缓存与 loop-closure 接口。

## 执行纪律：只有一个总阶段

下面所有复选项属于同一个连续实施阶段，不是可分别交付的里程碑。执行者从基线保护开始，连续完成实现、测试、文档、最终 diff 审核、单次最终提交和推送；不得在“只完成数据结构”“只支持两种分割”“只接普通引擎”或“等待后续再接 LC”等中间状态停止，也不得把未完整通过的部分推送为交付结果。

允许在本地按 TDD 顺序多次运行小测试，但只在全部验收通过后创建最终实现提交并推送云端。

## 固定基线与不可变约束

- 工作分支：`codex/unified-hart-anchor-propagation`。
- 基线提交：`c1fc82200e2e6dcbe4a7ae8ad36c76cb3a7e929e`（设计提交；其代码基线继承自 `c4c2820c649fd612be433c0b58912cee011ddac4`）。
- 开始时测试基线：`python -m pytest -q` -> `103 passed`。
- 不修改四种分割的算法、默认参数和旧 ndarray 公共输出。
- 不修改 legacy `Vertex.cache={'iou': [], 'scale': []}`、IoU 加权公式、逐层遍历顺序或 `scale_mask` 语义。
- HART 中 IoU/包含率绝不能进入尺度数值融合。
- confidence 只筛选核心，不成为乘法权重。
- HART 只新增 `anchor_min_pixels=64` 与 `scale_consistency_thresh=0.05` 两个数值参数。
- HART 失败不得静默回退 legacy；无证据或冲突空间单元回退局部尺度 1。
- 普通/LC 两条 HART 路径都必须完整交付。
- 只保留上一窗口尾部 overlap 状态，不保存全序列 Track 图。
- 不新增第三方依赖。
- 不改动工作树中既有未跟踪生成文件 `inference_engine/utils/_segmentation_cy.cpp` 与 `fast_seg.cpp`。

## 文件映射

### 新增

- `inference_engine/anchor_propagation/__init__.py`
- `inference_engine/anchor_propagation/types.py`
- `inference_engine/anchor_propagation/segmentation.py`
- `inference_engine/anchor_propagation/correspondence.py`
- `inference_engine/anchor_propagation/anchors.py`
- `inference_engine/anchor_propagation/consensus.py`
- `inference_engine/anchor_propagation/hart.py`
- `tests/test_hart_segmentation.py`
- `tests/test_hart_correspondence.py`
- `tests/test_hart_anchors.py`
- `tests/test_hart_consensus.py`
- `tests/test_hart_propagator.py`
- `tests/test_hart_streaming_integration.py`
- `scripts/verify_anchor_propagation.py`
- `docs/hart-anchor-propagation-cloud-validation.md`

### 修改

- `inference_engine/utils/layer_atomic_geometry.py`
- `inference_engine/streaming_window_engine.py`
- `inference_engine/streaming_window_engine_lc.py`
- `demo.py`
- `demo_lc.py`
- `eval_launch.py`
- `tests/test_segmentation_engine_modes.py`
- `tests/test_demo.py`
- `tests/test_demo_lc.py`

### 明确保留原算法

- `inference_engine/utils/depth.py`
- `inference_engine/utils/lsa.py`
- `inference_engine/utils/post_merge_split.py`

除非测试暴露纯接口导出需要，否则这三处不应改动；若必须修改，只允许新增无副作用导出/适配，不得改 legacy 计算。

---

## 唯一总阶段：连续完成以下全部步骤

### 1. 锁定 legacy 行为与新路由契约

**Files:**
- Modify: `tests/test_segmentation_engine_modes.py`
- Modify: `tests/test_demo.py`
- Modify: `tests/test_demo_lc.py`
- Create: `tests/test_hart_streaming_integration.py`

- [ ] 在任何生产代码改动前增加失败测试，固定解析规则：

```python
resolve_anchor_propagation(depth_refine=False, explicit=None) == "none"
resolve_anchor_propagation(depth_refine=True, explicit=None) == "legacy_iou"
resolve_anchor_propagation(depth_refine=False, explicit="hart") == "hart"
resolve_anchor_propagation(depth_refine=True, explicit="none") == "none"
```

- [ ] 固定 `ANCHOR_PROPAGATION_MODES = ("none", "legacy_iou", "hart")`，未知值快速失败。
- [ ] 固定兼容验证：未给新参数时，`segment_mode != 'depth' and depth_refine=False` 继续报原错误；显式 `anchor_propagation='none'` 时允许非 depth 全局-only 对照。
- [ ] 参数化四种 `segment_mode`，证明每种都允许显式 `hart`。
- [ ] 记录 legacy spy/call contract：`legacy_iou` 仍调用 `_build_segment_graph()` 和 `refine_depth_segments()`，不能调用 HART builder。
- [ ] 记录 HART spy/call contract：`hart` 调用统一 segmentation builder/HART propagator，不能调用 legacy `refine_depth_segments()`。
- [ ] 记录 `none` 不运行分割或局部传播。
- [ ] 为 demo、demo_lc、eval parser 增加失败测试：`--anchor_propagation` 接受三个值，默认 `None` 以保留旧 `depth_refine` 行为。
- [ ] 运行：

```bash
python -m pytest -q \
  tests/test_segmentation_engine_modes.py \
  tests/test_demo.py \
  tests/test_demo_lc.py \
  tests/test_hart_streaming_integration.py
```

Expected：新增断言因生产接口缺失而失败；原断言仍通过。

### 2. 建立统一类型与四模式层次适配

**Files:**
- Create: `inference_engine/anchor_propagation/types.py`
- Create: `inference_engine/anchor_propagation/segmentation.py`
- Modify: `inference_engine/utils/layer_atomic_geometry.py`
- Create: `tests/test_hart_segmentation.py`

- [ ] 在 `types.py` 定义不可变数据：

```python
@dataclass(frozen=True)
class SegmentationFrame:
    leaf_labels: np.ndarray
    parent_labels: np.ndarray
    anchor_labels: np.ndarray
    leaf_to_parent: np.ndarray
    anchor_to_leaf: np.ndarray
    split_diagnostics: dict | None = None


@dataclass(frozen=True)
class SegmentationWindow:
    frames: tuple[SegmentationFrame, ...]
    segment_mode: str


@dataclass(frozen=True)
class RegistrationState:
    base_points_tail: torch.Tensor
    base_poses_tail: torch.Tensor
    cumulative_sim3: tuple


@dataclass(frozen=True)
class AnchorPropagationState:
    local_scale_tail: torch.Tensor
    confidence_tail: np.ndarray
    segments_tail: tuple[SegmentationFrame, ...]
    track_state: object


@dataclass(frozen=True)
class PropagationResult:
    local_scale_mask: torch.Tensor
    next_state: AnchorPropagationState
    diagnostics: dict
```

- [ ] `segmentation.py` 实现紧凑化、pair-label 交集和严格验证 helpers：

```python
compact_labels(labels)
compact_intersection_labels(initial_labels, leaf_labels)
label_parent_lookup(child_labels, parent_labels)
validate_segmentation_frame(frame)
```

- [ ] `compact_intersection_labels()` 对 `(initial, leaf)` 编码，不假定 initial atom 完整嵌套在 split leaf。
- [ ] 给 `layer_atomic_geometry.py` 增加私有 staged metadata 入口；旧 `segment_point_map_layer_atomic()` 与 `segment_point_map_layer_atomic_split()` 继续只返回原 labels 且逐像素不变。
- [ ] 实现 `build_segmentation_window(...)`，四模式映射严格按设计规范第 5 节。
- [ ] 单元测试覆盖：
  - 四模式 output；
  - labels 紧凑、非负、全覆盖；
  - Anchor->Leaf->Parent 唯一映射；
  - 人工构造“一个 atom 被 split leaf 切开”，Anchor Cell 数量正确增加；
  - split diagnostics 保留；
  - 旧 atomic/split 公共返回逐元素不变；
  - 空窗口和不一致 shape 快速失败。
- [ ] 运行：

```bash
python -m pytest -q \
  tests/test_hart_segmentation.py \
  tests/test_layer_atomic_geometry.py \
  tests/test_layer_atomic_split_integration.py \
  tests/test_segmentation_modes.py
```

Expected：全部通过。

### 3. 实现稀疏对应、主次关系与 TrackSegments

**Files:**
- Create: `inference_engine/anchor_propagation/correspondence.py`
- Create: `tests/test_hart_correspondence.py`

- [ ] 定义轻量记录：

```python
PairRelation(src_id, tgt_id, intersection, iou, src_coverage, tgt_coverage)
TemporalEdge(src_node, tgt_node, relation, is_primary)
TrackNode(frame_index, label_id, track_id, segment_id, lineage_parent_id)
TrackWindow(...)
```

- [ ] 实现一次 `np.bincount` 的实际相交 label pair 统计，不创建每区域完整 mask。
- [ ] 候选条件固定为 `max(iou, src_coverage, tgt_coverage) >= corr_iou_thresh`。
- [ ] 当前目标的主前驱排序固定为 `tgt_coverage, iou, intersection, -src_id`，结果确定。
- [ ] 其他候选关系保留为 secondary，不复制最终尺度。
- [ ] split 时最高分子分支延续原 segment，其余创建带 `lineage_parent_id` 的新 segment；merge 只延续一个主 segment，其他来源记 secondary。
- [ ] 先构建 Leaf lineage，再在 Leaf 候选范围内构建 Anchor Cell TrackSegments。
- [ ] 单元测试覆盖：
  - sparse IoU/coverage 与现有 `pairwise_iou()`、`pairwise_intersection_ratio()` 数值等价；
  - 无交集；
  - 一对一稳定对应；
  - 一对多 split 小 child 通过单侧包含率；
  - 多对一 merge；
  - 一帧同时多条轨迹；
  - tie-break 确定；
  - Anchor 关系不能跨 Leaf 候选；
  - secondary 边不改变 segment scale 字段。
- [ ] 运行：

```bash
python -m pytest -q tests/test_hart_correspondence.py
```

Expected：全部通过。

### 4. 实现高置信直接锚点

**Files:**
- Create: `inference_engine/anchor_propagation/anchors.py`
- Create: `tests/test_hart_anchors.py`

- [ ] 定义：

```python
DirectAnchor(
    current_frame,
    current_anchor_id,
    current_segment_id,
    previous_segment_id,
    scale,
    pixel_count,
)
```

- [ ] 每个 overlap frame、每个窗口侧独立计算 confidence 分位数阈值。
- [ ] 核心固定为 previous/current Anchor masks、两侧 high confidence、两侧有限正 depth 的交集。
- [ ] `pixel_count < anchor_min_pixels` 时不得调用 IRLS。
- [ ] 调用现有 `align_depth_irls(current_base_depth, previous_refined_depth, core)`，方向测试必须验证 `scale * current ~= previous`。
- [ ] 仅接受有限正 scale；不 clamp 一个错误的 NaN/Inf 成有效锚点。
- [ ] 主/次关系都可以提供带来源的直接证据，但不得读历史 fused scale 并直接 append。
- [ ] 测试通过 monkeypatch/spy 证明：
  - 低 confidence 像素未进入 IRLS mask；
  - 单侧 confidence 不够时不进入核心；
  - 无效/非正 depth 被排除；
  - 63/64 像素边界；
  - scale 方向正确；
  - NaN、Inf、0、负值被拒绝；
  - IoU 改变但核心 depth 相同，不改变尺度值。
- [ ] 运行：

```bash
python -m pytest -q tests/test_hart_anchors.py
```

Expected：全部通过。

### 5. 实现 Track、Leaf 与 Parent 尺度共识

**Files:**
- Create: `inference_engine/anchor_propagation/consensus.py`
- Create: `tests/test_hart_consensus.py`

- [ ] 实现 log-scale complete-link 分组：组内 `max(log_s)-min(log_s) <= threshold`，不得使用相邻链式传递合并。
- [ ] 同一 Anchor TrackSegment 只有一个一致组时取 median；多个组时标记 conflict 并返回局部尺度 1。
- [ ] Leaf 共识：
  - 唯一组补全无锚点非冲突 Anchor Cells；
  - 多组保留已锚定 cell 的组中位数；
  - 多组中的无锚点 cell 回退 1；
  - conflict cell 不得被后续 fill 覆盖。
- [ ] Parent 共识：
  - 唯一组允许 split leaves 共享尺度；
  - 多组保持 leaves/cells 分离；
  - 对 `parent==leaf` 自动退化，不写模式分支。
- [ ] 按 `anchor_labels` 查表生成 mask，不用多个布尔 mask 相加。
- [ ] 每个 cell 输出状态 `direct / leaf_fill / parent_fill / conflict / no_anchor`。
- [ ] 单元测试覆盖：
  - median 抵抗单个异常值；
  - `[1.00, 1.04, 1.08]` 在 5% threshold 下不能因链式邻近合成一组；
  - 同 cell 多来源冲突回退；
  - 唯一 Leaf 共识补全；
  - Leaf 多组冲突；
  - split leaves Parent 一致共享；
  - split leaves Parent 冲突分离；
  - conflict 不被 parent fill；
  - mask 完整、有限、正、每像素一次赋值。
- [ ] 运行：

```bash
python -m pytest -q tests/test_hart_consensus.py
```

Expected：全部通过。

### 6. 编排完整 HART 窗口传播与尾部状态

**Files:**
- Create: `inference_engine/anchor_propagation/hart.py`
- Create: `inference_engine/anchor_propagation/__init__.py`
- Create: `tests/test_hart_propagator.py`

- [ ] 实现：

```python
class HartAnchorPropagator:
    def __init__(
        self,
        corr_iou_thresh=0.3,
        anchor_min_pixels=64,
        scale_consistency_thresh=0.05,
        confidence_quantile=0.5,
    ): ...

    def refine(
        self,
        previous_registration_state,
        previous_anchor_state,
        current_base_points,
        current_confidence,
        current_segments,
        overlap,
    ) -> PropagationResult: ...
```

- [ ] 第一窗口返回全 1 mask，但仍构建 Track 与 next tail state。
- [ ] 后续窗口先构建完整当前 Track，再只在 previous tail/current head 产生直接锚点，随后一次性赋值到当前 TrackSegments。
- [ ] previous refined depth 只能按 `previous base depth × previous local scale` 重建。
- [ ] next state 只切片保存最后 `overlap` 帧。
- [ ] 汇总设计规范要求的 diagnostics 与三个分项 runtime；diagnostics 不参与决策。
- [ ] 生产代码对 shape、overlap 范围、层次、finite scale 做显式校验。
- [ ] 测试覆盖：
  - first window；
  - overlap 多帧产生 Track median；
  - 当前完整窗口一次性赋值而非逐跳加权；
  - 下一窗口重新估计；
  - previous local mask 只用于局部锚点，不修改 registration state；
  - no anchor 与 conflict 回退；
  - tail-only state；
  - diagnostics 计数和比例自洽。
- [ ] 运行：

```bash
python -m pytest -q \
  tests/test_hart_segmentation.py \
  tests/test_hart_correspondence.py \
  tests/test_hart_anchors.py \
  tests/test_hart_consensus.py \
  tests/test_hart_propagator.py
```

Expected：全部通过。

### 7. 接入普通 Streaming Engine，同时保持 legacy 原路

**Files:**
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `tests/test_segmentation_engine_modes.py`
- Modify: `tests/test_hart_streaming_integration.py`

- [ ] 构造函数新增：

```python
anchor_propagation: str | None = None
anchor_min_pixels: int = 64
scale_consistency_thresh: float = 0.05
corr_iou_thresh: float = 0.3
```

- [ ] 解析旧 `depth_refine`，日志同时打印 `segment_mode` 与 resolved propagation。
- [ ] `_registration_worker` 明确三路：
  - `none`：现有全局路径，不分割；
  - `legacy_iou`：保留现有 `_build_segment_graph -> refine_depth_segments` 与缓存行为；
  - `hart`：构造 Base state、SegmentationWindow、HART mask 和 refined output。
- [ ] HART 普通引擎保存：
  - 输出缓存 `local_points = base_points * local_scale_mask`；
  - 内存 registration tail = 未乘 local mask 的 Base points；
  - 内存 anchor tail = local mask + segmentation/conf/track；
  - 不保存第二份完整窗口三通道 points。
- [ ] `_reset_state()` 清理两类新 state 和 propagator 运行状态。
- [ ] 测试使用两个合成窗口和 monkeypatch global registration，证明：
  - 下一窗口 global register 收到 base tail；
  - local anchor 收到 refined depth；
  - 普通输出收到 base×local；
  - legacy 调用和输出 contract 不变；
  - none 不构造 segmentation；
  - 四模式 HART 均完成两窗口 CPU smoke。
- [ ] 运行：

```bash
python -m pytest -q \
  tests/test_segmentation_engine_modes.py \
  tests/test_hart_streaming_integration.py
```

Expected：全部通过。

### 8. 接入 Loop Closure Engine，消除尺度重复语义

**Files:**
- Modify: `inference_engine/streaming_window_engine_lc.py`
- Modify: `tests/test_hart_streaming_integration.py`

- [ ] LC 的 HART 在线路径维护临时 cumulative Base/pose state，用于 registration 与局部锚点；磁盘缓存仍保留 raw local points、raw poses 和 pairwise `sim3`，供 loop optimizer 使用。
- [ ] HART 字段固定为 `local_scale_mask`；legacy 保留现有 `scale_mask` 字段和聚合行为。
- [ ] `aggregate_caches()` 分支：

```python
if "local_scale_mask" in cache:
    local_points = optimized_cumulative_scale * local_scale_mask * raw_local_points
elif "scale_mask" in cache:
    # 原 legacy 语义，逐字保留现有实现
else:
    local_points = optimized_cumulative_scale * raw_local_points
```

- [ ] 测试以不等于 1 的 global scale 和 local mask 验证乘法恰好各一次。
- [ ] 测试 HART raw cache 未被 base/refined points 覆盖，loop closure 读取字段不变。
- [ ] 测试 legacy LC aggregation 数值逐元素不变。
- [ ] 测试 parse cache 去 overlap 时 `local_scale_mask` 与其他 Tensor 同步切片。
- [ ] 运行：

```bash
python -m pytest -q \
  tests/test_hart_streaming_integration.py \
  tests/test_demo_lc.py
```

Expected：全部通过。

### 9. CLI、验证脚本、诊断与云端说明一次接齐

**Files:**
- Modify: `demo.py`
- Modify: `demo_lc.py`
- Modify: `eval_launch.py`
- Modify: `tests/test_demo.py`
- Modify: `tests/test_demo_lc.py`
- Create: `scripts/verify_anchor_propagation.py`
- Create: `docs/hart-anchor-propagation-cloud-validation.md`

- [ ] 三个 CLI 增加：

```text
--anchor_propagation {none,legacy_iou,hart}
--anchor_min_pixels 64
--scale_consistency_thresh 0.05
--corr_iou_thresh 0.3
```

- [ ] `eval_launch.py` 不再用 `segment_mode` 强制决定显式传播器；未传新参数时仍复现旧 `depth_refine=args.segment_mode != 'depth'` 默认。
- [ ] 启动日志打印 resolved mode 和所有 HART 参数，避免实验配置不透明。
- [ ] `verify_anchor_propagation.py` 构造确定性的两个小窗口，对四种分割分别运行 `legacy_iou` 与 `hart`，检查：
  - 无异常；
  - mask shape；
  - finite/positive；
  - exact image coverage；
  - tail state 帧数；
  - diagnostics 必需字段。
- [ ] 云端文档提供：
  - 环境/编译准备；
  - full pytest；
  - `4 × 2` CPU smoke；
  - KITTI/TUM/Sintel 示例命令；
  - ATE/RPE、runtime、memory、no-anchor/conflict 指标提取；
  - legacy parity 与 HART 性能预算判定；
  - 失败日志收集；
  - 结果 JSON 命名，必须编码 segment/propagation/seed。
- [ ] 运行：

```bash
python -m pytest -q tests/test_demo.py tests/test_demo_lc.py
python scripts/verify_anchor_propagation.py
```

Expected：全部通过并为八个组合打印 `[PASS]`。

### 10. 全量回归、静态检查、最终一次提交并推送

**Scope:** 全部新增/修改文件。

- [ ] 运行定向测试：

```bash
python -m pytest -q \
  tests/test_hart_segmentation.py \
  tests/test_hart_correspondence.py \
  tests/test_hart_anchors.py \
  tests/test_hart_consensus.py \
  tests/test_hart_propagator.py \
  tests/test_hart_streaming_integration.py
```

- [ ] 运行所有 segmentation/CLI/engine 回归：

```bash
python -m pytest -q \
  tests/test_depth_segmentation_stages.py \
  tests/test_geometry_segmentation.py \
  tests/test_layer_atomic_geometry.py \
  tests/test_layer_atomic_integration.py \
  tests/test_layer_atomic_split_integration.py \
  tests/test_post_merge_split.py \
  tests/test_post_merge_split_evaluation.py \
  tests/test_segmentation_modes.py \
  tests/test_segmentation_engine_modes.py \
  tests/test_demo.py \
  tests/test_demo_lc.py
```

- [ ] 运行完整套件与 smoke：

```bash
python -m pytest -q
python scripts/verify_segmentation_modes.py
python scripts/verify_anchor_propagation.py
```

- [ ] 运行语法和 diff 检查：

```bash
python -m compileall -q inference_engine/anchor_propagation
git diff --check
git status --short
```

- [ ] 人工审核最终 diff，确认：
  - `depth.py` legacy 公式无变化；
  - `lsa.py` legacy 入口无变化；
  - 没有把未跟踪 C++ 生成文件加入暂存；
  - 没有数据集路径、结果文件、cache 或大数组进入提交；
  - HART 四模式、普通/LC、CLI/验证文档全部在同一交付中；
  - diagnostics 不作为权重；
  - local/global scale 在普通与 LC 都只按定义应用一次。
- [ ] 只暂存本方案文件，创建一个最终实现提交：

```bash
git add \
  inference_engine/anchor_propagation \
  inference_engine/utils/layer_atomic_geometry.py \
  inference_engine/streaming_window_engine.py \
  inference_engine/streaming_window_engine_lc.py \
  demo.py demo_lc.py eval_launch.py \
  scripts/verify_anchor_propagation.py \
  tests/test_hart_segmentation.py \
  tests/test_hart_correspondence.py \
  tests/test_hart_anchors.py \
  tests/test_hart_consensus.py \
  tests/test_hart_propagator.py \
  tests/test_hart_smoke_script.py \
  tests/test_hart_streaming_integration.py \
  tests/test_segmentation_engine_modes.py \
  tests/test_demo.py tests/test_demo_lc.py \
  README.md \
  docs/hart-anchor-propagation-cloud-validation.md \
  docs/superpowers/specs/2026-07-20-unified-hart-anchor-propagation-design.md \
  docs/superpowers/plans/2026-07-20-unified-hart-anchor-propagation.md
git commit -m "feat: add unified HART anchor propagation"
```

- [ ] 确认提交后的状态只剩实施前已存在的未跟踪生成文件，并推送目标分支：

```bash
git status --short --branch
git push origin codex/unified-hart-anchor-propagation
```

- [ ] 用远端引用核对推送成功：

```bash
git ls-remote --heads origin codex/unified-hart-anchor-propagation
git rev-parse HEAD
```

Expected：远端 hash 与本地 `HEAD` 一致。到此唯一总阶段完成；在此之前不得宣称交付完成。

## 最终验收矩阵

| 检查 | 必须结果 |
|---|---|
| 旧测试 | 基线 103 项及新增测试全部通过 |
| Legacy parity | 四分割 legacy labels/scale mask 逐元素兼容 |
| 四分割 HART | depth、geometry、layer_atomic、layer_atomic_split 全部可运行 |
| IoU 职责 | 只参与对应/事件，不参与尺度权重 |
| Confidence 职责 | 只筛选直接锚点核心 |
| Scale consensus | median + complete-link；冲突不平均 |
| Split 层次 | atom 内 split 由 Anchor Cell 交集正确处理 |
| 在线状态 | 只保留尾部 overlap |
| 普通引擎 | global registration 使用 Base，输出使用 Base×Local |
| LC 引擎 | optimized Global×HART Local×Raw，各应用一次 |
| 参数数量 | 只新增两个 HART 数值参数 |
| 性能 | propagation <=1.20× 目标/1.50× 硬上限；端到端 <=10% 硬上限 |
| 云端交付 | 最终实现提交已推送，远端 hash 等于本地 HEAD |
