# HART v2：Pose Consensus Coupling 设计

- 日期：2026-07-21
- 实现基线：`origin/codex/unified-hart-anchor-propagation@e776d72`
- 实现分支：`codex/hart-v2-pose-consensus`
- 最终发布目标：`origin/codex/unified-hart-anchor-propagation`
- 设计目标：让区域尺度证据立即进入当前窗口相机 Sim(3)，同时隔离局部物体尺度，优先改善轨迹尺度漂移且保持点云分区域修正能力

## 1. 最终结论

HART v2 不修改现有分割、Track、直接锚点或层次共识算法，而是在区域尺度输出后增加独立的 Pose Consensus Coupling：

\[
\boxed{
\text{HART Anchor Propagation}
+
\text{Pose Consensus Coupling}
}
\]

HART v2 对每个当前窗口输出三个互相分离的结果：

1. `window_scale`：窗口公共尺度 \(g_k\)，立即耦合到当前窗口相机 Sim(3)；
2. `local_residual_mask`：区域局部残差 \(\ell_k(p)\)，只修正最终点云；
3. `pose_support_mask`：支持公共尺度的直接 Track 区域，只约束下一窗口全局注册。

对于有可靠区域尺度的 Anchor Cell：

\[
r_{k,c}=g_k\ell_{k,c}.
\]

对于没有可靠尺度的 `no_anchor` 或 `conflict` Anchor Cell，\(r_{k,c}\) 未定义，不能把实现中的单位占位值当成观测；它们固定使用 \(\ell_{k,c}=1\)，直接继承窗口公共尺度。

## 2. 已验证的根因

当前普通引擎的执行顺序是：

```text
register_adjacent_windows()
→ 固定 s_coarse, R_coarse, t_coarse
→ 更新 current local_points 和 camera_poses
→ HART 只生成 local_scale_mask
→ local_scale_mask 只乘点云
```

因此当前窗口 HART 结果不会修改已经确定的 camera poses。提交 `e776d72` 又把完整 `Base × local mask` 尾部送入下一窗口注册，使局部物体和冲突区域能够间接污染下一个全局尺度；这不是“区域尺度如何进入相机轨迹”的明确算法定义。

LC 引擎也在 HART 运行前固定 pairwise Sim(3)。HART mask 只在最终 aggregation 中乘点云，当前窗口的 pairwise/cumulative 相机链没有显式公共尺度耦合。

## 3. 保持不变的 HART 主体

以下实现和职责保持不变：

- `depth / geometry / layer_atomic / layer_atomic_split` 四种分割；
- merge、post-merge split 和 staged segmentation metadata；
- Anchor Cell–Leaf–Parent 严格层次；
- IoU/包含率只建立区域对应、主边、次级边和 split/merge 事件；
- 主边形成 TrackSegment，次级边保留多来源证据；
- 双窗口逐帧高置信有效深度核心；
- 现有 IRLS 估计直接区域尺度；
- 同一 TrackSegment 内按来源先取中位数，再做冲突检查；
- complete-link 对数尺度一致性；
- 冲突不平均；
- Leaf/Parent 共识只补全低置信空间单元。

`none` 与 `legacy_iou` 路径保持行为兼容，不读取任何 HART v2 状态。

## 4. 尺度与坐标系定义

### 4.1 普通 Streaming Engine

第 \(k\) 个窗口的原始局部点云为 \(P_k^{raw}\)，粗注册尺度为 \(s_k^{coarse}\)：

\[
P_k^{coarse}=s_k^{coarse}P_k^{raw}.
\]

HART 使用上一窗口最终 refined tail 和当前 coarse Base，在重叠帧中估计区域尺度 \(r_{k,c}\)：

\[
r_{k,c}D_{k,c}^{coarse}\approx D_{k-1,c}^{refined}.
\]

Pose Consensus 得到 \(g_k\) 后：

\[
s_k^{final}=s_k^{coarse}g_k.
\]

最终 Base 与 refined geometry 为：

\[
P_k^{base}=s_k^{final}P_k^{raw},
\qquad
P_k^{refined}=P_k^{base}\ell_k.
\]

### 4.2 Loop Closure Engine

LC 保存窗口原始局部点云和相邻窗口 pairwise Sim(3)。假设当前粗 pairwise 尺度为 \(s_{k,pair}^{coarse}\)：

\[
s_{k,pair}^{final}=s_{k,pair}^{coarse}g_k.
\]

`g_k` 写入当前 pairwise Sim(3)，随后累积成 HART 使用的 cumulative Base。磁盘只保存纯局部 `local_residual_mask`，回环优化后：

\[
P_k^{final}
=S_k^{optimized}\ell_kP_k^{raw}.
\]

`window_scale` 不得再次写入 residual mask，否则 LC 会重复应用公共尺度。

## 5. Pose Consensus

### 5.1 输入

Pose Consensus 只读取 Anchor HART 已产生的信息：

- `segment_scales: dict[int, float]`：无冲突 TrackSegment 的尺度；
- `conflict_segments: set[int]`；
- `direct_anchors: tuple[DirectAnchor, ...]`，其中 `pixel_count` 是双侧高置信直接核心像素数；
- `anchor_tracks.segment_ids`；
- `status_maps`；
- overlap 内全部双侧高置信、有限且正深度像素总数。

不新增法向、RGB、距离、语义、光流或复合权重。

### 5.2 一致性分组

对 `segment_scales` 的值复用现有 complete-link 分组：

\[
\max_{i,j\in G}
\left|\log\frac{r_i}{r_j}\right|
\le\tau_s.
\]

IoU、包含率和像素数不参与组内尺度计算。

### 5.3 支持率

尺度组 \(G_j\) 的直接支持像素数为：

\[
w_j=
\sum_{a:\,a.current\_segment\_id\in G_j}
a.pixel\_count.
\]

分母不是“当前所有无冲突 Track 的支持”，而是 overlap 内全部双侧高置信且深度有效像素：

\[
W=
\sum_p
\mathbf 1[
C_{prev}^{high}(p)
\land C_{cur}^{high}(p)
\land V(p)
].
\]

冲突 Track 的核心像素自然包含在 \(W\) 中，但冲突 Track 不属于任何候选尺度组。这防止少量一致证据在大量冲突或无对应证据存在时获得虚假的 100% 支持。

组支持率为：

\[
\rho_j=\frac{w_j}{\max(W,1)}.
\]

### 5.4 接受规则

只有某个组严格满足：

\[
\rho_j>0.5
\]

时才接受窗口公共尺度。由于各直接核心的 label-pair 互斥，最多只有一个组能严格超过 50%。

接受时：

\[
g_k=\operatorname{median}
\{r_s\mid s\in G_j\}.
\]

这里使用普通 TrackSegment 中位数；`pixel_count` 只选择组，不对 \(g_k\) 做加权。

以下任一条件成立时固定回退：

```text
第一窗口
无直接锚点
无无冲突 segment scale
W = 0
没有组严格超过 50%
g 非有限或非正
```

回退结果为：

```text
window_scale = 1
pose_support_mask = 全 False
```

### 5.5 “稳定/静态”的可观测边界

当前系统没有语义或独立运动估计。HART v2 中的 pose-support 只表示：

> 有直接锚点、Track 连续、无尺度冲突，并属于严格多数尺度组。

该定义能阻止小物体 Track 修改相机尺度，但不能保证排除占据超过半数高置信像素的大型运动物体。设计和文档不得声称已经实现语义静态区域识别。

## 6. Local Residual

Anchor HART 继续产生 `regional_scale_mask` 和 `status_maps`，但前者只是当前调用内的临时结果。

对有可靠尺度的状态：

\[
\ell_k(p)=\frac{r_k(p)}{g_k},
\quad
status(p)\in
\{direct,leaf\_fill,parent\_fill\}.
\]

对无可靠尺度的状态：

\[
\ell_k(p)=1,
\quad
status(p)\in
\{no\_anchor,conflict\}.
\]

必须验证 `local_residual_mask` 形状为 `[N,H,W,1]`，并且全部有限、严格为正。

该定义保证：

### 6.1 可靠区域保持原 HART 点云结果

\[
s_k^{coarse}g_k\frac{r_{k,c}}{g_k}P_k^{raw}
=s_k^{coarse}r_{k,c}P_k^{raw}.
\]

### 6.2 无证据区域继承公共尺度

\[
s_k^{coarse}g_k\cdot1\cdot P_k^{raw}.
\]

不能把 `no_anchor/conflict` 的单位占位值除以 \(g_k\)，否则会抵消当前窗口的相机尺度修正。

## 7. Pose-support Mask

Pose-support 只能来自被接受主组中的直接尺度 TrackSegment：

```text
pose_support(p)
= status(p) == direct
  and anchor_track_segment(p) in accepted_group
```

`leaf_fill` 和 `parent_fill` 可用于点云补全，但没有独立直接证据，不能支持相机注册。其他一致组即使尺度有效，也只保留局部残差，不能修改相机轨迹。

完整窗口返回 `[N,H,W]` bool mask；状态只保存尾部 overlap。

## 8. 当前窗口 Sim(3) Coupling

### 8.1 保留原始输入

引擎必须在任何尺度或 pose 变换前保留当前窗口：

```text
raw_local_points
raw_camera_poses
```

不能在已经应用粗 Sim(3) 的 camera poses 上再次乘 \(g_k\)。

### 8.2 两阶段求解

普通引擎固定顺序：

```text
raw points / raw poses
↓
register_adjacent_windows → s_coarse, R_coarse, t_coarse
↓
coarse Base points = s_coarse × raw points
↓
Anchor HART + Pose Consensus → g, residual, pose support
↓
s_final = s_coarse × g
↓
使用 raw current poses、previous final poses 和 s_final 重新求 R_final,t_final
↓
final Base points = s_final × raw points
↓
final poses = apply_sim3_to_pose(raw poses, s_final, R_final, t_final)
↓
refined points = final Base × residual
```

当前 `register_camera_poses_kabsch_pytorch()` 的旋转构造也使用 scale，因此必须重新计算 `R_final,t_final`，不能只修改平移或复用粗 `R,t`。

新增一个明确的“给定尺度求相邻窗口 pose”函数，`register_adjacent_windows()` 仍负责粗尺度 IRLS；legacy 调用契约保持不变。

### 8.3 第一窗口

第一窗口没有历史尺度证据：

```text
window_scale = 1
local_residual_mask = 1
pose_support_mask = False
```

第一窗口仍建立 segmentation/Track/anchor tail 状态，不凭单窗口内部结构生成公共尺度。

## 9. 下一窗口注册

### 9.1 注册掩码

上一窗口尾部支持与当前窗口头部是同一批 overlap 图像和像素。候选注册 mask 为：

\[
M_{candidate}
=M_{mutual\ confidence}
\land M_{pose\ support}^{prev}.
\]

只有当候选像素总数不少于 `anchor_min_pixels` 时才使用 pose-support。

### 9.2 安全回退

pose-support 不足时：

```text
registration points = previous final Base
registration mask = 原 mutual-confidence mask
```

不能回退到全部 refined geometry。

### 9.3 普通引擎源点

pose-support 足够时按需构造：

\[
P_{src}=P_{prev}^{base}\ell_{prev}.
\]

只在 `pose_support_mask` 选中的像素参与尺度 IRLS；其他局部区域不影响注册。

### 9.4 LC 源点

LC pairwise 注册必须与 raw camera poses 保持同一局部坐标系：

\[
P_{src}^{pair}=P_{prev}^{raw}\ell_{prev}.
\]

LC 的 final Base tail 只用于下一窗口 HART 深度锚点，不可直接作为 pairwise 注册输入。

## 10. 状态与接口

### 10.1 数据类型

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
```

旧 HART 分支尚未作为稳定公共 API 发布，因此 v2 直接使用语义明确的字段名，不保留会混淆“区域绝对尺度”和“局部残差”的 `local_scale_mask` 别名。

### 10.2 职责边界

`HartAnchorPropagator.refine()` 是纯计算过程：

- 读取上一 Registration/Anchor state；
- 产生 window scale、residual、support 和 next anchor state；
- 不创建当前 RegistrationState；
- 不修改引擎缓存。

引擎在最终 Sim(3) 求解完成后原子提交：

- final Base tail；
- final pose tail；
- support tail；
- next anchor state；
- LC cumulative Sim(3)。

这样不会把 coarse pose 与 final point scale 混入同一个状态。

### 10.3 内存

删除 `propagated_points_tail`。普通引擎通过 final Base tail 与 residual tail 按需构造 refined support；LC 复用当前已有 raw cache。只保存 overlap 尾部，状态内存不随序列长度增长。

## 11. LC Coupling

LC 后续窗口顺序固定为：

```text
previous raw × previous residual + previous raw poses
↓
pairwise coarse Sim(3)
↓
accumulate previous final cumulative × coarse pairwise
↓
构造 current coarse cumulative Base
↓
HART → g, residual, support
↓
pairwise final scale = pairwise coarse scale × g
↓
以 raw previous/current poses 和 pairwise final scale 重求 pairwise R,t
↓
重新 accumulate final cumulative Sim(3)
↓
提交 final Base/pose state
↓
磁盘保存 raw points、final pairwise Sim(3)、local residual
```

`aggregate_caches()` 只执行：

```text
optimized cumulative Sim(3) × raw points × local residual
```

全局尺度和 local residual 各应用一次。

## 12. Diagnostics

保留现有 HART diagnostics，并新增：

```text
window_scale
pose_consensus_group_count
pose_consensus_selected_segment_count
pose_consensus_support_pixels
pose_consensus_valid_pixels
pose_consensus_support_ratio
pose_consensus_accepted
pose_support_pixel_ratio
regional_scale_min/median/max
local_residual_min/median/max
registration_pose_support_pixels
registration_pose_support_used
registration_pose_support_fallback_count
coarse_registration_scale
final_registration_scale
```

Diagnostics 只记录决策和结果，不回流成为算法权重。

## 13. ATE/RPE 解释与验证

现有 `eval/vo_eval.py::eval_metrics()` 对 ATE 和 RPE 使用：

```python
align=True
correct_scale=True
```

因此一次统一的全轨迹尺度变化会被评估器主动消除。`g_k` 已进入 camera poses 并不保证报告的 Sim(3)-aligned ATE 数值变化；只有逐窗口 \(g_k\) 修正非均匀尺度漂移时，正式 ATE 才可能改善。

不得把“ATE 数字相同”单独作为 HART 未进入位姿链的证据。云端验证同时记录：

- 原有 Sim(3)-aligned ATE、RPE translation、RPE rotation，继续作为正式可比指标；
- 不做 scale correction 的补充 ATE；
- 每窗口 `s_coarse / g / s_final`；
- HART 与 `none` 的原始 camera translation 差异；
- pose consensus 接受率、support ratio、registration fallback 次数。

补充指标不能替换或改写正式基线指标。

## 14. 失败与回退规则

- 第一窗口：`g=1`、residual=1、support=False；
- 无对应或无直接锚点：`g=1`；
- 直接核心不足：不生成 anchor；
- IRLS 非有限或非正：丢弃 anchor；
- Track 冲突：不进入 pose group，local residual=1；
- Leaf/Parent 冲突：保持已解析局部组，未解析单元 residual=1；
- 无严格多数 pose group：`g=1`，所有已解析 `r` 保留为 local residual；
- support 与 mutual-confidence 交集不足：使用 final Base 和原 confidence mask；
- `s_final`、`g` 或 residual 非有限/非正：快速失败，不静默切换 legacy；
- HART 异常不得自动回退到 `legacy_iou`，避免实验结果混合。

## 15. 测试与验收

### 15.1 Pose Consensus 单元测试

- complete-link 分组复用现有尺度阈值；
- 多数支持组得到未加权 Track median；
- 相对多数但未覆盖全部有效高置信像素 50% 时拒绝；
- 恰好 50% 时拒绝；
- conflict 像素进入分母但不进入候选组；
- 单个小物体 Track 不能获得公共尺度；
- 无锚点、非法尺度和第一窗口回退。

### 15.2 Residual 单元测试

- direct/leaf_fill/parent_fill 使用 `regional/g`；
- no_anchor/conflict 固定为 1；
- accepted group 与外部局部组都保持 `g×residual=regional`；
- mask 完整、有限、严格为正；
- pose support 只包含主组 direct Track；
- next state 只保存 residual tail。

### 15.3 普通引擎集成测试

- `g` 在当前窗口改变 `s_final` 和 camera translations；
- 给定 `s_final` 重新求 `R,t`，不复用粗结果；
- 输出为 final Base × residual；
- 下一窗口只在 pose-support 足够时使用 Base × residual；
- support 不足时使用 Base，不使用全部 refined；
- 第一窗口和 `g=1` 路径稳定；
- `none/legacy_iou` 行为不变。

### 15.4 LC 集成测试

- `g` 写入当前 final pairwise Sim(3)；
- cumulative Base 在 final pairwise 下重建；
- 下一 pairwise 注册使用 raw × residual；
- raw cache 未被覆盖；
- aggregation 使用 optimized global × residual × raw；
- 公共尺度没有重复应用。

### 15.5 评估与回归

- `python setup.py build_ext --inplace`；
- `python scripts/verify_anchor_propagation.py`；
- `python -m pytest -q` 全量通过；
- HART 专项测试验证 test-first 的 red/green；
- 设计、实现、CLI 和云端验证文档字段一致；
- `git diff --check` 无格式错误；
- 远端目标 hash 与最终本地提交一致。

## 16. 非目标

- 不重写分割、merge、split 或 Track；
- 不增加语义、运动分割、光流或学习模块；
- 不把 IoU、法向、RGB、confidence 或距离组合成尺度权重；
- 不增加逐区域相机位姿；
- 不增加全序列尺度图优化或回溯修改旧窗口；
- 不保证每个数据集 ATE 必然下降；
- 不改变正式 ATE/RPE 的全局 Sim(3) 对齐定义。

## 17. 最终算法定义

```text
Previous final state
        ↓
Coarse window registration
        ↓
Current coarse Base geometry
        ↓
Anchor HART → reliable regional scales r
        ↓
Complete-link scale groups
        ↓
Strict-majority high-confidence support
        ↓
window scale g + pose-support group
        ├── current final Sim(3): s_final = s_coarse × g
        └── local residual: resolved r/g, unresolved 1
        ↓
final Base × residual → point-cloud output
        ↓
pose-support-only registration for next window
```

职责固定为：

\[
\boxed{
\text{公共尺度负责当前与后续相机轨迹，}
\quad
\text{局部残差负责分区域点云质量。}
}
\]
