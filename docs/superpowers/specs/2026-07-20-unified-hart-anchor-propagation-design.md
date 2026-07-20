# 四分割统一 HART-AP 锚点传播旁路设计

- 日期：2026-07-20
- 代码基线：`codex/auto-post-merge-split@c4c2820c649fd612be433c0b58912cee011ddac4`
- 实现分支：`codex/unified-hart-anchor-propagation`（从包含本设计的
  `codex/auto-post-merge-split@c1fc82200e2e6dcbe4a7ae8ad36c76cb3a7e929e`
  创建）
- 设计范围：锚点产生、区域/轨迹尺度共识、窗口内传播、重叠窗口传播、普通与回环引擎状态隔离
- 最终方法名：`HART-AP`（High-confidence Anchor–Region–Track Anchor Propagation，高置信锚点—区域—轨迹传播）

## 1. 结论

本轮不再修改 `depth`、`geometry`、`layer_atomic`、`layer_atomic_split` 的分割判据，而是在现有局部尺度传播旁边新增一条可选择的 HART-AP 路径。四种分割全部能够运行：

```text
segment_mode
├── depth
├── geometry
├── layer_atomic
└── layer_atomic_split

anchor_propagation
├── none
├── legacy_iou
└── hart
```

`legacy_iou` 继续调用现有 `make_sp_graph()`、`assign_overlap_window_depth_scale()` 和 `align_adjacent_windows_depth_segments()`，不得改变默认结果。`hart` 使用同一个统一接口接收四种分割的层次化标签，并遵守四条职责边界：

| 信号 | 唯一职责 |
|---|---|
| 分割层次 | 定义 Anchor Cell、Leaf 与 Parent 的空间边界 |
| IoU/包含率 | 建立区域对应、Track 身份与 split/merge 事件 |
| confidence | 筛选能够产生直接尺度锚点的像素 |
| IRLS | 在高置信核心中估计尺度 |

IoU 不再作为尺度数值的权重；confidence 也不与 IoU、法向、RGB 或距离组合成复合权重。直接尺度采用现有 IRLS，重复证据采用中位数，一致性判断只在一维对数尺度上进行。

HART-AP 是一个在线旁路。每个窗口通过上一窗口尾部与当前窗口头部的 overlap 重新估计当前窗口 TrackSegment 的尺度，然后一次性应用到当前窗口对应的 Anchor Cell Track；不等待完整序列，不做逐边 IoU 加权扩散，也不回溯修改已输出窗口。

## 2. 当前代码事实与问题边界

### 2.1 当前分割统一，传播仍固定

`inference_engine/utils/lsa.py::make_sp_graph()` 已把四种分割都压成最终二维 labels，随后固定调用 `match_segmentation_seq()`。窗口间尺度由 `assign_overlap_window_depth_scale()` 初始化，窗口内由 `align_adjacent_windows_depth_segments()` 沿 `Vertex` 边逐层传播。

这已经实现“分割模式统一”，但没有实现“传播器可替换”。`segment_mode` 与 `depth_refine` 仍耦合，`eval_launch.py` 还用 `args.segment_mode != 'depth'` 决定是否启用局部传播，使 `depth` 无法从同一入口运行论文式 `legacy_iou` 与新 HART-AP 的严格对照。

### 2.2 IoU 同时承担身份与尺度权重

现有代码先用 IoU 阈值建立多对多边，再在区域交集内运行 IRLS，并把边的 IoU 与尺度一起写入目标 `Vertex.cache`。窗口内每次传播和最终 mask 都再次用 IoU 加权平均：

```text
IoU >= threshold
    ↓
建立传播边
    ↓
区域交集 IRLS
    ↓
IoU 加权融合多个尺度
    ↓
父节点融合值继续逐帧传播
```

IoU 可以说明两个区域在像素平面上的对应强度，但不能证明某个尺度估计更可靠。几何上相同但大小变化的 split/merge 区域会降低 IoU；一个大而错误的区域也可能具有较高 IoU。因此新方法保留 IoU/包含率的身份判断作用，删除其尺度权重作用。

### 2.3 局部锚点没有使用模型 confidence

普通与回环引擎的全局 Sim(3) 已在双窗口高置信交集中估计；局部区域尺度仍使用完整区域交集。HART-AP 将同样的双侧高置信原则下沉到每个 Anchor Cell，confidence 只负责建立可靠核心，不参与尺度加权。

### 2.4 `layer_atomic_split` 丢失层次与诊断

当前 `segment_point_map_layer_atomic_split()` 已经得到：

- 初始 `atom_labels`；
- merge 后 `merged.labels`；
- split 后 `refined`；
- `SplitDiagnostics`。

但公共入口只返回 `refined`。传播阶段无法知道多个 split leaves 属于哪个父区域，也无法用尺度一致性验证 split 边界。

### 2.5 HART 结果必须进入后续全局配准

如果 HART 只修改当前输出点云，而下一窗口仍使用未细化 Base 点云做 Sim(3)，局部
尺度不会进入后续相机位姿链，因而无法改善 ATE。HART 的 refined tail 必须成为下一
窗口 registration input，使修正按窗口逐次传递。

HART 路径必须把逻辑状态分为：

- Base geometry：只完成窗口级 Sim(3)，用于重建局部锚点且防止 local mask 重复相乘；
- Refined geometry：Base geometry 乘 HART 局部 mask，供输出、局部锚点和下一窗口
  全局配准。

只额外保留尾部 overlap 的 Base/refined registration points 与一通道 local mask，
不保存两份完整序列点云。

## 3. 目标与非目标

### 3.1 目标

- 为四种 `segment_mode` 提供完全相同的 HART-AP 核心算法。
- 保留 `none` 与 `legacy_iou`，形成 `4 × 3` 可重复消融矩阵。
- 将 IoU/包含率限制为对应关系和事件检测，不作为尺度权重。
- 只在双窗口高置信 Anchor Cell 核心中运行现有 IRLS。
- 允许一帧同时存在多条区域/表面轨迹；“主前驱”只针对单个当前单元。
- 用 TrackSegment 中位数抵抗 overlap 中的单帧异常。
- 用 Leaf/Parent 内尺度共识补全低置信单元；尺度冲突时禁止平均。
- 让 split leaves 在尺度一致时共享尺度、冲突时保持独立。
- 让 HART refined tail 进入下一窗口全局 Sim(3)，形成能影响相机轨迹和 ATE 的
  逐窗口反馈链。
- 普通与 LC 引擎使用同一 HART 输出定义，保证全局尺度只应用一次。
- 状态只保留尾部 overlap，内存不随序列长度增长。

### 3.2 非目标

- 不修改四种分割的阈值、法向、merge 或 watershed 算法。
- 不引入 SAM、光流、语义、RGB 传播权重、法向传播权重或学习模块。
- 不做像素级深度补全；HART 只估计分段常数尺度。
- 不做完整序列因子图、全局 Track 优化或回溯重写旧窗口。
- 不声称局部传播可以消除无绝对尺度或无回环条件下的长期漂移。
- 不把天空、反射幻觉和无稳定表面强行修复；无可靠证据时局部尺度为 1。
- 本轮不改造未暴露四种分割配置的 `SlidingWindowEngine`；交付范围是 `StreamingWindowEngine`、`StreamingWindowEngineLC` 及其 demo/eval 入口。

## 4. 旁路与向后兼容

新增公共参数：

```text
--anchor_propagation {none,legacy_iou,hart}
```

内部统一字段名为 `anchor_propagation`。原 `depth_refine` 暂时保留为兼容别名：

```python
if anchor_propagation is None:
    resolved = "legacy_iou" if depth_refine else "none"
else:
    resolved = anchor_propagation
```

显式新参数优先，因此 `depth_refine=False, anchor_propagation='hart'` 必须运行 HART。旧调用未传 `anchor_propagation` 时保持原行为：

- demo 默认 `depth_refine=False`：`none`；
- demo 显式 `--depth_refine`：`legacy_iou`；
- eval 的 `depth`：`none`；
- eval 的另外三种模式：`legacy_iou`。

当前引擎对“非 depth 分割但未开启 `depth_refine`”会报错。该兼容行为只在新参数未显式给出时保留；要运行 `geometry + none` 等全局-only 对照，必须显式传 `anchor_propagation='none'`。这样旧调用不会从报错静默变成另一种实验，新接口又能真正解耦分割与传播。

`legacy_iou` 必须继续走原函数和原缓存语义。HART 的分割适配、状态和 mask 不能反向进入 legacy 路径。旧的 `depth_refine` 只发出一次弃用提示，不在本轮删除。

## 5. 统一分割层次

### 5.1 数据结构

```python
@dataclass(frozen=True)
class SegmentationFrame:
    leaf_labels: np.ndarray
    parent_labels: np.ndarray
    anchor_labels: np.ndarray
    leaf_to_parent: np.ndarray
    anchor_to_leaf: np.ndarray
    split_diagnostics: dict | None


@dataclass(frozen=True)
class SegmentationWindow:
    frames: tuple[SegmentationFrame, ...]
    segment_mode: str
```

所有 label map 都必须是 `[H,W]`、非负、紧凑的 `np.intp`，并完整覆盖图像。层次必须满足：

```text
每个 Anchor Cell 只属于一个 Leaf
每个 Leaf 只属于一个 Parent
```

### 5.2 Anchor Cell 的必要修正

聊天方案曾把 `layer_atomic_split` 表示成：

```text
Parent Region ⊇ Split Leaf ⊇ Atom
```

真实代码不能保证最后一层。post-merge watershed 的目的之一就是恢复落在初始 atom 内部的边界，因此一个初始 atom 可能被 split 切开。直接把原始 `atom_labels` 作为 Leaf 子节点会造成一个锚点单元跨越 split 边界。

统一方案定义：

\[
\boxed{
\text{Anchor Cell}
=
\operatorname{compact}(\text{initial label},\text{final leaf label})
}
\]

实现上对 `(initial_label, leaf_label)` 二元组编码并重新紧凑化。这样 split 能切开 atom，同时 Anchor Cell 仍保留初始细粒度几何支持。

### 5.3 四种模式映射

| 模式 | initial | leaf | parent | anchor |
|---|---|---|---|---|
| `depth` | depth Felzenszwalb labels | coarse depth labels | leaf | `compact(initial, leaf)` |
| `geometry` | geometry Felzenszwalb labels | geometry merged labels | leaf | `compact(initial, leaf)` |
| `layer_atomic` | initial depth atoms | atomic merged labels | leaf | `compact(initial, leaf)` |
| `layer_atomic_split` | initial depth atoms | split refined labels | pre-split merged labels | `compact(initial, leaf)` |

因此四种模式都具备 Anchor–Leaf–Parent 三层；没有真实 split 层次时 `parent_labels == leaf_labels`，算法无需特殊分支。

`layer_atomic_geometry.py` 新增 staged/private metadata 入口，现有公共 ndarray 返回值保持不变。`SplitDiagnostics` 随 `SegmentationFrame` 进入 HART diagnostics，但不进入算法权重。

## 6. 稀疏对应与多轨迹结构

### 6.1 一次 label-pair 计数

禁止继续为每个区域展开完整布尔 mask 并构造密集 `N × M × H × W` 关系。对相邻两张紧凑 labels：

```python
pair_code = src_labels * num_tgt + tgt_labels
intersection = np.bincount(pair_code.ravel())
```

同时用 `np.bincount` 得到源/目标面积，从实际非零 pair 计算：

\[
IoU=\frac{I}{A_s+A_t-I},\quad
C_s=\frac{I}{A_s},\quad
C_t=\frac{I}{A_t}.
\]

候选关系满足：

\[
\max(IoU,C_s,C_t)\ge\tau_{corr}.
\]

IoU 处理同尺寸稳定区域，单侧包含率保护 one-to-many split 和 many-to-one merge。三者只决定关系是否存在，不参与尺度数值。

### 6.2 Leaf 对应

每个当前 Leaf 选择一个确定的主前驱，排序键固定为：

```text
target coverage ↓, IoU ↓, intersection pixels ↓, source id ↑
```

其余合格关系保留为次级边。主边定义时间身份；次级边用于：

- split/merge 事件；
- 历史 Track 来源；
- Parent/Leaf 欠分割提示；
- 允许独立直接锚点证据进入一致性检查。

次级边不能直接复制某个已融合尺度。

### 6.3 Track 与 TrackSegment

一个父 Leaf 可以对应多个子 Leaf，一个当前 Leaf 也可以覆盖多个历史 Leaf。为避免一次 split 后把不同子分支永久绑定为一个尺度，区分：

- Track lineage：区域身份的有向谱系，可包含 split/merge；
- TrackSegment：两个 split/merge 事件之间的单一分支，是尺度中位数的聚合单位。

发生 split 时，得分最高的子分支可延续原 segment，其余子分支创建新 segment 并记录 `lineage_parent_id`。发生 merge 时，一个 segment 作为主身份，其他来源记录为次级关系。任何尺度聚合不得跨越事件边界无条件合并。

### 6.4 Anchor Cell Track

在已建立的 Leaf 候选关系内，再对 Anchor Cell 使用同一稀疏计数与主/次关系规则。Anchor Cell TrackSegment 才是尺度携带单元；Leaf Track 负责大尺度身份和事件边界。

这样即使某个最终 Region 欠分割，只要内部 Anchor Cells 有稳定空间边界和不同高置信尺度，HART 仍能保留多个尺度组。若同一个 Anchor Cell 本身收到互相冲突的来源，则没有可用空间边界，必须标记 conflict 并回退 1，不能用中位数掩盖冲突。

## 7. 重叠窗口直接锚点

对上一窗口尾部与当前窗口头部的对应帧、对应 Anchor Cell：

\[
K_{ab}=
A_a^{prev}\cap A_b^{cur}
\cap C_{prev}^{high}\cap C_{cur}^{high}\cap V.
\]

其中 `V` 要求两侧深度有限且为正。HART 对每个 overlap frame 分别计算两侧 confidence 分位数阈值，避免一个整体高置信帧饿死同一 overlap 中的其他帧。分位数参数复用现有 `top_conf_percentile`。

只有：

```text
|K_ab| >= anchor_min_pixels
```

时才调用现有 `align_depth_irls()`。方向固定为：

\[
s_{ab}\,d_{cur}^{base}\approx d_{prev}^{refined}.
\]

返回值必须有限且大于 0；否则丢弃该候选。confidence 只形成 `K_ab`，不作为 IRLS 外部权重。IoU、包含率和交集像素数也不作为尺度权重。

主/次对应均可提出独立直接锚点证据，但证据保留来源 TrackSegment。次级关系不会把历史最终尺度直接注入当前单元。

## 8. 尺度聚合与共识

### 8.1 一致性定义

唯一的一致性判据为：

\[
\left|\log\frac{s_i}{s_j}\right|\le\tau_s.
\]

尺度按 log 值排序，使用 complete-link 分组：一个组内最大与最小 log scale 的差不得超过 `tau_s`。这避免相邻链式分组把首尾明显冲突的尺度连在一起。

### 8.2 Anchor TrackSegment 汇总

同一 Anchor Cell TrackSegment 在 overlap 可得到多个直接尺度。若所有有效尺度形成唯一一致组：

\[
s_{track}=\operatorname{median}(s_1,\ldots,s_m).
\]

若形成多个冲突组，该 TrackSegment 标记 `conflict`，局部尺度为 1，不能选最大组或对多组取中位数。

### 8.3 Leaf 共识

一个 Leaf 中所有非冲突、有锚点的 Anchor TrackSegment 再做相同分组：

- 唯一组：以组内中位数补全该 Leaf 内无锚点且非 conflict 的 Anchor Cells；
- 多组：各已锚定 Anchor Cells 保持自己的组尺度，未锚定单元回退 1；Leaf 标记欠分割/冲突；
- 无组：整个 Leaf 局部尺度为 1。

### 8.4 Parent 共识

对于 `layer_atomic_split`，同一 pre-split Parent 下的多个 Leaves：

- 所有已锚定 Leaf 只有一个一致组：它们共享组中位数，并补全 Parent 中仍无锚点的非冲突 Leaves；
- 存在多个尺度组：保持 Leaves/Anchor Cells 分离，禁止父区域统一尺度；
- 无可靠组：局部尺度为 1。

另外三种模式 `parent == leaf`，自动退化为 Leaf 共识，不需要模式专用传播代码。

最终像素 mask 只按紧凑 `anchor_labels` 查表生成；每个像素恰好获得一个尺度，不允许多个 mask 求和覆盖。

### 8.5 状态枚举

每个 Anchor Cell 的最终状态必须是以下之一：

```text
direct       直接锚点/Track median
leaf_fill    Leaf 唯一共识补全
parent_fill  Parent 唯一共识补全
conflict     同一空间单元存在冲突，scale=1
no_anchor    无可靠证据，scale=1
```

这些状态只用于诊断和明确回退，不能再次转成权重。

## 9. 窗口内与窗口间传播

### 9.1 第一窗口

第一窗口没有历史锚点：

- 建立完整 Leaf/Anchor Track；
- `local_scale_mask = 1`；
- 保存尾部 overlap 的 Base points、mask、confidence、SegmentationFrame 与 TrackSegment 元数据；
- 不凭当前窗口内部的相对形状生成绝对局部尺度。

### 9.2 后续窗口

```text
窗口级全局 Sim(3)
    ↓
Current Base geometry
    ↓
四模式统一 SegmentationWindow
    ↓
当前窗口完整 Leaf/Anchor TrackSegments
    ↓
Previous tail ↔ Current head 对应
    ↓
双侧高置信直接锚点 + IRLS
    ↓
TrackSegment median
    ↓
Leaf/Parent scale consensus
    ↓
一次性为当前窗口 TrackSegments 赋值
    ↓
local scale mask
```

空间层次仍采用一次性共识，不执行像素/父子节点逐跳加权；但窗口之间必须逐个传播：
当前 refined tail 参与下一窗口 Sim(3)，下一窗口再基于新的 Base 和上一 refined
锚点重新估计 TrackSegment 尺度。尺度不是绑定完整序列的永久常量。

## 10. Base/Refined 状态与引擎语义

### 10.1 逻辑状态

```python
@dataclass
class RegistrationState:
    base_points_tail: torch.Tensor
    base_poses_tail: torch.Tensor
    propagated_points_tail: torch.Tensor
    cumulative_sim3: tuple


@dataclass
class AnchorPropagationState:
    local_scale_tail: torch.Tensor
    confidence_tail: np.ndarray
    segments_tail: tuple[SegmentationFrame, ...]
    track_state: object
```

上一窗口 refined depth 按需计算：

\[
D_{prev}^{refined}
=D_{prev}^{base}\cdot S_{prev}^{local}.
\]

Base tail 负责局部尺度数值重建，propagated tail 专供下一次 registration；两者都只
保留尾部 overlap。完整窗口输出继续写缓存后释放。

### 10.2 普通 Streaming Engine

HART 路径：

```text
raw points
→ window Sim(3)
→ base points
→ HART local mask
→ refined points（写普通输出缓存）
```

下一窗口全局注册读取 `RegistrationState.propagated_points_tail = base tail × local
mask`。下一窗口局部锚点仍从 base tail × local mask 重建 previous refined depth，
避免在同一窗口重复应用 local mask。

### 10.3 Loop Closure Engine

LC 磁盘缓存继续保留 raw local points 与 pairwise `sim3`，供 loop optimizer 使用；另外保存 HART 的纯局部残差 mask：

```text
local_scale_mask = HART residual only
```

在线处理时 LC 磁盘缓存仍保持 raw；内存中的下一次 pairwise Sim(3) 输入使用
`raw tail × local mask`，因此 HART 会改变后续 pairwise Sim(3) 与相机轨迹。
同时维护 cumulative Base state 供 HART 数值估计。回环优化后：

\[
P_{final,k}
=S_{global,k}^{optimized}
\cdot S_{local,k}^{HART}
\cdot P_{raw,k}.
\]

全局尺度只能在 aggregation 中应用一次；HART mask 不得包含窗口级全局尺度。为避免与 legacy 当前 `scale_mask` 含义混淆，新字段固定命名为 `local_scale_mask`，legacy 字段和聚合逻辑保持原状。

## 11. 模块结构

新增：

```text
inference_engine/anchor_propagation/
├── __init__.py
├── types.py
├── segmentation.py
├── correspondence.py
├── anchors.py
├── consensus.py
└── hart.py
```

- `types.py`：统一数据结构、状态、结果和 diagnostics。
- `segmentation.py`：四种分割 staged adapter、Anchor Cell 交集紧凑化和层次验证。
- `correspondence.py`：稀疏 pair count、Leaf/Anchor 主次关系、Track lineage/segment。
- `anchors.py`：逐帧高置信核心、IRLS 安全包装、直接锚点记录。
- `consensus.py`：complete-link 尺度分组、Track/Leaf/Parent 共识和 mask 查表。
- `hart.py`：只编排上述模块，返回 `PropagationResult`。

`utils/depth.py` 和 `utils/lsa.py` 的 legacy 图与传播函数留在原位。HART 模块可以复用 `align_depth_irls`，不能修改 legacy 的 cache、权重或遍历公式。

统一调用：

```python
result = hart_propagator.refine(
    previous_registration_state=prev_registration_state,
    previous_anchor_state=prev_anchor_state,
    current_base_points=current_base_points,
    current_confidence=working_window["conf"],
    current_segments=segmentation_window,
    overlap=self.overlap,
)
```

返回：

```python
PropagationResult(
    local_scale_mask,
    next_anchor_state,
    diagnostics,
)
```

## 12. 参数

HART 首版只新增两个数值参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `anchor_min_pixels` | `64` | 一个 Anchor Cell 双侧高置信核心产生直接锚点所需的最少像素 |
| `scale_consistency_thresh` | `0.05` | complete-link 组内允许的最大 log-scale 距离，约为 5.1% 比例差 |

复用：

- `top_conf_percentile`：逐 overlap frame 生成高置信 mask；
- `corr_iou_thresh`：IoU/包含率候选关系阈值，HART 默认 `0.3`。

不得再加入 `normal_weight`、`rgb_weight`、`confidence_weight`、`path_decay`、`max_hops` 或数据集特定阈值。

## 13. Diagnostics

HART 每窗口记录：

```text
leaf_track_count
anchor_track_count
primary_edge_count
secondary_edge_count
split_event_count
merge_event_count
direct_anchor_candidate_count
direct_anchor_count
rejected_small_core_count
rejected_invalid_scale_count
track_conflict_count
leaf_consensus_count
leaf_conflict_count
parent_consensus_count
parent_conflict_count
direct_pixel_ratio
filled_pixel_ratio
conflict_pixel_ratio
no_anchor_pixel_ratio
scale_mask_min/median/max
correspondence_runtime_ms
anchor_runtime_ms
consensus_runtime_ms
propagation_runtime_ms
```

Diagnostics 不改变任何决策。正常路径不保存全分辨率中间数组；需要可视化时由独立评估脚本导出少量选中帧。

## 14. 复杂度与内存

- 对应：每对相邻帧一次 label pair 扫描，约 `O(P + E)`。
- IRLS：只对 overlap 中通过对应且高置信核心至少 64 像素的候选运行。
- 一致性：每个 Leaf/Parent 的少量一维尺度排序，约 `O(A log A)`。
- 状态：当前完整窗口加上一窗口尾部 Base/propagated overlap，约
  `O((W+2O)P)`，不依赖序列长度。
- 额外常驻大数组：一通道 local scale mask、紧凑 labels、尾部 Base state 和
  尾部 propagated registration points；不保存第二份完整窗口或完整序列点云。

预算：

- HART 传播阶段 CPU 中位耗时不超过 legacy 局部传播的 `1.20×`，硬上限 `1.50×`；
- 端到端单窗口中位延迟增幅目标不超过 5%，硬上限 10%；
- 非 LC 普通输出不因状态分离永久多存一份完整 point map；
- LC 只新增一通道 `local_scale_mask` 和轻量 diagnostics；
- 无 NaN/Inf scale，scale mask 全部为有限正数。

## 15. 测试与验收

### 15.1 基线保护

设计时基线为：

```text
python -m pytest -q
103 passed
```

交付时必须证明：

- 未显式传 `anchor_propagation` 的旧构造行为不变；
- `legacy_iou` 对四种分割的 labels、图、scale mask 与当前基线逐元素一致；
- 现有 103 项测试继续通过；
- `depth.py` 的 legacy IoU 加权公式和 cache 语义未修改。

### 15.2 新单元测试

至少覆盖：

1. 四模式层次映射、紧凑标签、全覆盖和嵌套不变量；
2. split 在 atom 内切开时 Anchor Cell 被正确拆开；
3. 稀疏 pair count 与当前密集 IoU/包含率在小样例上等价；
4. 一帧多 Leaf/Anchor Track 并行；
5. split/merge 的主边、次级边和 TrackSegment 分支确定；
6. IoU/包含率不出现在尺度聚合公式；
7. 只使用双侧高置信、有效、正深度核心；
8. 核心不足 64 像素不运行 IRLS；
9. 同一 Track 多锚点使用 median；
10. complete-link 不发生链式误合并；
11. 同一 Anchor Cell 多来源冲突时回退 1；
12. Leaf 唯一共识补全低置信 Anchor Cells；
13. Leaf 多尺度冲突时保持已锚定组、未锚定回退 1；
14. split Leaves 一致时 Parent 共识共享尺度；
15. split Leaves 冲突时保持分离；
16. 第一窗口 mask 全 1；
17. mask 完整覆盖、有限、为正且每像素只赋值一次；
18. next state 只保留尾部 overlap。

### 15.3 引擎与 LC 集成

- `none / legacy_iou / hart` 路由与旧 `depth_refine` 兼容解析；
- 四种 `segment_mode` 均可显式运行 `hart`；
- 普通 HART 下一窗口 registration input 使用上一窗口 refined tail；
- HART 修正会改变后续 Sim(3) 和相机位姿，进入 ATE 计算链；
- 普通输出正确使用 `base × local`；
- LC raw cache 未被 HART 覆盖；
- LC aggregation 使用 `optimized global × HART local × raw`，全局尺度没有重复；
- demo、demo_lc、eval_launch 参数和日志完整；
- CPU 合成窗口 smoke 覆盖 `4 × {legacy_iou,hart}`；
- 缓存解析、去 overlap 和 diagnostics 不破坏既有输出字段。

### 15.4 实验矩阵

核心公平比较：

| 分割 | `legacy_iou` | `hart` |
|---|---:|---:|
| depth | ✓ | ✓ |
| geometry | ✓ | ✓ |
| layer_atomic | ✓ | ✓ |
| layer_atomic_split | ✓ | ✓ |

另保留 `none` 作为只运行窗口级 Sim(3) 的对照。记录 ATE、RPE translation、RPE rotation、传播耗时、峰值 CPU 内存、无锚点像素比例、冲突像素比例和 split/merge 事件数。性能结论必须来自相同输入、相同 seed、相同 warm-up 与相同分割参数。

## 16. 失败与回退规则

- 无历史窗口：mask=1。
- 无对应：mask=1。
- 高置信核心不足：不生成直接锚点。
- IRLS 非有限或非正：丢弃候选。
- TrackSegment 内多个冲突组：该空间单元 mask=1。
- Leaf/Parent 多组冲突：保留已有空间分组，未锚定单元 mask=1。
- 任何层次不满足嵌套或完整覆盖：快速失败并报告模式/帧索引，不静默修补。
- HART 路径异常不得自动切到 legacy，以免实验结果混合；用户必须显式选择回退。

## 17. 最终算法定义

\[
\boxed{
\text{IoU 加权逐边传播}
\Rightarrow
\text{高置信 Anchor Cell 直接锚点}
+
\text{多 TrackSegment 中位数}
+
\text{Leaf/Parent 尺度共识}
+
\text{窗口在线重估}
}
\]

职责固定为：

```text
Initial segmentation × Final leaf
→ 构造不会跨边界的 Anchor Cells

IoU / containment
→ 建立 Leaf 与 Anchor Track lineage

Confidence
→ 选择直接锚点核心

IRLS
→ 估计单个直接尺度

Median + complete-link consensus
→ 一致则共享，冲突则保持空间分离或回退

Base / Refined state
→ Base 防止重复乘尺度，Refined 逐窗口反馈到下一次 Sim(3)
```

该设计让四种分割真正共享同一个新传播器，同时通过旁路完整保留论文/现有 `legacy_iou` 基线。
