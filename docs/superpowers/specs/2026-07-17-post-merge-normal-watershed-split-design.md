# Auto 后合并法向分水岭分割设计

- 日期：2026-07-17
- 基线提交：`codex/unified-segmentation-methods@98cce5f9f470599aca0cf5a6614f39409d929d58`
- 实现分支：`codex/auto-post-merge-split`
- 最终方法名：`layer_atomic_split`

## 1. 结论

当前 `layer_atomic` 的几何阶段只在初始 Felzenszwalb 原子之间执行 DSU 合并。它可以撤销 coarse depth layer 的错误组合，但不能切开任何一个初始原子。因此，只要独立物体边界落在同一个初始原子内部，后续几何判据无论多强都无法恢复该边界。

最终方案在现有 Auto 合并完成之后增加一次保守的区域内细分：

1. 点云法向突变产生候选边界和区域种子；
2. 一次 marker-controlled watershed 产生 2–4 个候选子区；
3. 用一个统一分数判断该候选划分是否值得保留；
4. RGB 边界对比或归一化三维间隙任一成立，都可以确认法向候选；
5. 每个 Auto 最终区域最多处理一次，不递归，最多四个叶子。
6. 一个布尔消融开关控制是否启用 RGB/间隙辅助确认；关闭时只使用法向一致性改善分数。

这不是另一套密集 Geometry 分割。现有 Auto 负责恢复连续主支撑面；新增步骤只在 Auto 已合并的大区域内部恢复少量、明显的独立物体边界。

## 2. 目标与非目标

### 2.1 目标

- 恢复位于同一初始原子内部、且在法向上明显不连续的独立物体边界。
- 保持 Auto 对墙、地面、道路和连续转弯面的合并能力。
- 一个区域可以直接得到 2、3 或 4 个子区，而不是被限制为二分。
- 控制区域数量和执行时间，避免 Geometry 式碎片化。
- 在不派生第二套方法和第二个阈值的前提下，直接测量辅助确认对碎片、效率和轨迹指标的影响。
- 保持结果确定、标签紧凑、像素全覆盖，并可直接进入现有 LSA 匹配和尺度传播。
- 保留原 `layer_atomic` 行为不变，新增 `layer_atomic_split` 作为可回退、可对照的最终方法。

### 2.2 非目标

- 不做逐像素通用场景分割或语义分割。
- 不追求恢复所有细小物体、纹理边缘或遮挡边缘。
- 不引入平面拟合、曲率阈值、递归切分、时序投票、SLIC 或学习模型。
- 不修改现有 atom merge、跨帧匹配、尺度估计和尺度传播公式。
- 不将 RGB 单独作为切分依据；纯纹理变化不能切开几何连续平面。
- 不修改现有前置阈值过滤；新增方法的职责只从 Auto 最终区域开始。

## 3. 依据

对 17 条 KITTI 诊断轨迹和两个 TUM 室内场景共 12 条诊断轨迹的分析显示：

- Geometry 边界落在 Auto 初始原子内部的中位比例为 KITTI 95.72%、TUM 93.54%，证明缺失边界主要无法通过 atom merge 恢复。
- 直接采用逐像素几何边界会把区域数中位放大到 KITTI 3.50 倍、TUM 6.02 倍，而边界收益很小，证明不能把 Geometry 结果直接叠加到 Auto。
- 在现有诊断代理目标上，法向角的区分能力明显高于 RGB 和归一化三维间隙；RGB 对室内接触物体边界有补充作用，三维间隙对真实空间断裂有补充作用。
- 本地 CPU 微基准估计：法向/间隙线索、RGB 梯度、连通种子和一次 watershed 合计约增加 18% 的分割阶段时间；全图 SLIC、完整 Geometry 和区域平面拟合均明显更慢，因此不进入最终方案。

这些统计只用于选择结构和预算，不把 Geometry 输出当作真实标注，也不以它的碎片数量作为优化目标。

## 4. 处理位置

`layer_atomic_split` 的执行顺序固定为：

```text
Point map
  -> depth-based Felzenszwalb atoms
  -> current threshold filtering and coarse layer construction
  -> current coarse-layer-guided atomic geometry merge
  -> Auto final regions
  -> select only regions large enough to contain at least two valid children
  -> one-pass normal watershed refinement inside those regions
  -> compact refined labels
  -> existing match_segmentation_seq
  -> existing scale estimation and propagation
```

新增分割必须发生在 Auto merge 之后。它接收 Auto 的最终标签，不修改合并过程，也不在 watershed 之后再次运行 atom merge。

## 5. 唯一的候选分割信号

### 5.1 法向图

复用当前 Geometry 工具和现有 `normal_method` 配置中的点图法向估计方式。对无效点使用无效掩码，不参与法向统计。为降低单像素噪声，对三个法向分量分别做固定的 `3 x 3` 中值滤波，然后重新单位化。

相邻像素 `p,q` 的法向边缘为：

\[
e_n(p,q)=\arccos\!\left(\operatorname{clip}(|n_p^\top n_q|,0,1)\right).
\]

绝对点积消除法向正负号不一致。每个像素的边缘强度 `E_n(p)` 是其有效四邻域法向角的最大值。

### 5.2 固定屏障与种子

使用一个固定、可解释的法向屏障：

\[
T_n=30^\circ.
\]

满足 `E_n >= T_n` 的有效像素是候选屏障。去除屏障和无效法向像素后，在 `R` 内计算四连通分量。分量按面积降序排列；满足最小面积的前四个分量成为 watershed markers。固定角度避免每个区域产生新的统计阈值，也使“什么是明显法向突变”保持一致。

区域相关的最小子区面积只由现有分割尺度推导，不增加公开调参项：

\[
A_{min}(R)=\max(\texttt{seg\_min\_size},\lceil0.02|R|\rceil).
\]

若合格 marker 少于两个，区域原样返回。marker 多于四个时只保留面积最大的四个，其余像素仍由 watershed 分配，不会被丢弃。

因此，一个区域经过一次处理即可自然得到 2–4 个叶子；叶子数量来自明显的法向连通核心，而不是递归次数或固定二分。

## 6. 一次分水岭

在区域 `R` 的裁剪包围盒内，以 `E_n` 为地形、以上述 2–4 个 marker 为种子执行一次 marker-controlled watershed。mask 只排除区域外像素。无效点不能成为 marker，其地形值设为区域内有效 `E_n` 的最大值，但仍由最近的有效 marker 获得标签，从而保证父区域像素全覆盖。评分时继续忽略无效点。

watershed 只决定候选边界的位置，不自动改变最终标签。它的输出必须满足：

- 恰好覆盖 `R` 的每一个像素；
- 子区数量为 2–4；
- 每个子区面积不小于 `A_min(R)`；
- 相同输入重复运行得到完全相同的标签。

任何约束失败时，整个区域保留原 Auto 标签，不做局部修补。

## 7. 一个统一的接受分数

候选划分 `P={R_1,...,R_k}` 只计算一个接受分数。面积和四叶上限是结构约束，不是额外的语义判断条件。

### 7.1 法向一致性改善

为同时消除法向正负号影响，令 `V_A` 为区域 `A` 内具有有效法向的像素集合，定义等权二阶矩：

\[
M(A)=\frac{1}{|V_A|}\sum_{p\in V_A}n_p n_p^\top,
\qquad
H_n(A)=1-\lambda_{max}(M(A)).
\]

所有有效法向等权。法向二阶矩或边界对比缺少有效样本时，统一分数记为 0，保留父区域。

候选划分的相对改善为：

\[
G_n(P)=\operatorname{clip}\left(
\frac{H_n(R)-\sum_i\frac{|R_i|}{|R|}H_n(R_i)}{H_n(R)+\epsilon},0,1
\right).
\]

### 7.2 RGB 或三维间隙确认

仅当 `split_aux_confirmation=True` 时，对候选子区之间的边界计算：

- `g_rgb`：把现有 `[N,3,H,W]`、`[0,1]` 图像转为 `[N,H,W,3]` 后，计算四邻域欧氏颜色差；输入只做裁剪和布局转换，不做 Lab 转换；
- `g_gap`：已有局部采样尺度归一化后的三维邻接距离。若像素 `p,q` 分别属于初始原子 `a(p),a(q)`，且 merge 阶段已经得到原子内部采样尺度 `s_a`，则

\[
g_{gap}(p,q)=\frac{\|P_p-P_q\|_2}
{\sqrt{s_{a(p)}s_{a(q)}}+\epsilon}.
\]

原子尺度在 Auto merge 与 split 之间复用，不重复统计。这一归一化保持点云全局尺度不变，也能评价落在同一初始原子内部的候选边界。

对线索 `x`，边界对比度为：

\[
r_x=\frac{\operatorname{median}(g_x\mid\partial P)}
{Q_{75}(g_x\mid R\setminus\partial P)+\epsilon},
\qquad
C_x=\operatorname{clip}\left(\frac{r_x-1}{r_x+1},0,1\right).
\]

最终确认度为：

\[
C(P)=\max(C_{rgb},C_{gap}).
\]

这表示 RGB 和点云间隙是二选一的确认信号，不是两道必须同时通过的门。RGB 缺失时 `C_rgb=0`；有效三维边界不足时 `C_gap=0`。两者都不可用时不切分。

当 `split_aux_confirmation=False` 时，不计算 RGB 和三维间隙边缘图，也不计算上述对比度，辅助因子直接取 1。该开关只用于同一方法内部的消融，不改变法向候选、marker、watershed、最小面积和四叶上限。

### 7.3 最终分数

\[
A(P)=
\begin{cases}
C(P), & \texttt{split\_aux\_confirmation=True}\\
1, & \texttt{split\_aux\_confirmation=False}
\end{cases},
\qquad
\boxed{S(P)=G_n(P)\,A(P)}
\]

仅当：

\[
S(P)\ge\texttt{split\_score\_thresh}
\]

时接受整个候选划分，否则保留父区域。默认值固定为 `0.10`。该阈值越高，切分越保守；它只控制是否接受一次候选划分，不触发递归。开关开启和关闭时共用同一个阈值，不为法向单独设置第二个阈值。

## 8. 公开接口与配置

公开接口增加一个方法值、一个数值阈值和一个布尔消融开关：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `segment_mode` | 保持现有默认 | 新增可选值 `layer_atomic_split` |
| `split_score_thresh` | `0.10` | 接受候选划分的统一分数阈值 |
| `split_aux_confirmation` | `True` | 是否用 RGB 或归一化三维间隙确认法向候选 |

`split_max_leaves=4` 是方法常量，不作为普通实验参数；`30°`、`2%` 和最小区域面积是固定算法定义。`split_aux_confirmation` 只切换同一个分数中的辅助因子，不注册新的 `segment_mode`，也不允许单独阈值。这样可以防止消融演变成另一套数据集特定方法。

CLI 使用 `--split_aux_confirmation` / `--no-split_aux_confirmation`，默认开启。两个选项映射到同一个布尔值，不增加模式名称。

现有 `layer_atomic` 的默认值、输出和调用路径不得改变。切换回 `layer_atomic` 即可得到改进前结果；切换 Git 分支可得到完整代码级回退。

## 9. 代码结构

计划新增：

- `inference_engine/utils/post_merge_split.py`
  - 法向边缘图；
  - marker 提取；
  - 单次 watershed；
  - 统一分数；
  - 区域标签替换和紧凑化。

计划最小修改：

- `inference_engine/utils/layer_atomic_geometry.py`
  - 保持现有 merge 公共签名、默认返回值和逐像素结果不变；
  - 允许把内部计算整理为私有 helper，使组合入口在 merge 之后复用 atom scale，再调用 post-merge split。
- `inference_engine/utils/lsa.py`
  - 注册 `layer_atomic_split`；
  - 仅该模式向 post-merge split 传递 RGB 和辅助确认开关；现有前置过滤调用保持不变。
- `inference_engine/streaming_window_engine.py`
  - 将已经存在的 `images` 传入 `make_sp_graph`，不重复读取和转换图像。
- 分割模式配置、命令行选项和诊断导出
  - 新增最终方法名和最小诊断字段。

不得把后处理逻辑复制到调用端；所有判据集中在 `post_merge_split.py`。

## 10. 诊断输出

每帧只记录下列聚合字段，默认不保存大数组：

- `split_parent_count`
- `split_proposed_count`
- `split_accepted_count`
- `split_added_regions`
- `split_score_mean/max`
- `split_reject_no_markers`
- `split_reject_small_child`
- `split_reject_low_score`
- `split_runtime_ms`
- `split_aux_confirmation`

需要可视化时，通过独立验证脚本额外导出少量选中帧的 parent、markers、candidate 和 accepted label PNG，不改变正常推理路径。

## 11. 效率设计

- 法向图每帧只计算一次；辅助确认开启时，RGB 梯度和归一化间隙也各计算一次并供所有区域复用。
- 关闭辅助确认时完全跳过 RGB 梯度和归一化间隙边缘图计算，用于同时验证精度影响和可节省的开销。
- 只有面积至少容纳两个 `A_min(R)` 的父区域才进入 marker 提取。
- 没有两个合格 marker 时立即退出，不运行 watershed 和评分。
- watershed 在父区域包围盒内运行，而不是重复处理整帧。
- 每个父区域最多一次 watershed，不递归，不拟合平面，不运行 SLIC。
- 最大四个 marker 限制后续 LSA 节点增长。

验收预算：

- 分割阶段 CPU 时间相对 `layer_atomic` 增幅不超过 20%；
- 端到端单窗口延迟增幅目标不超过 5%，硬上限 10%；
- 额外峰值内存不超过一组全分辨率 `float32` 边缘图、两组工作标签和 RGB 引用；
- KITTI 与 TUM 的区域数中位增长不超过 30%，任一评估轨迹不超过 50%。

预算必须在目标服务器上用相同输入、相同 warm-up、至少 30 次计时的中位数和 P90 复测。本地 CPU 数字只作为实现前的相对证据。

## 12. 测试与验收

### 12.1 单元测试

- 连续平面只有 RGB 纹理边缘：不切分。
- 连续转弯面：不因渐变法向而碎裂。
- 两表面有明显法向变化且 RGB 边界明显：切分。
- 两表面有明显法向变化且三维间隙明显、RGB 不明显：切分。
- 法向噪声但 RGB/间隙不确认：不切分。
- 对同一法向候选，辅助确认开启时因 RGB/间隙不足而拒绝，关闭时可以按 `G_n` 接受。
- 辅助确认关闭时不调用 RGB/间隙边缘构造函数。
- 一个父区内存在三个或四个明显核心：一次得到相应叶子数。
- 超过四个核心：最终最多四叶。
- 任一候选子区过小：父区整体保持不变。
- RGB 缺失、无效点和混合有效掩码：稳定退化，不崩溃。
- 点云全局缩放：间隙确认与最终标签保持不变。
- 标签紧凑、像素全覆盖、结果确定。

### 12.2 集成与回归测试

- `layer_atomic_split` 从配置、CLI、streaming engine、`make_sp_graph` 到诊断输出完整连通。
- `images` 的 NCHW/HWC、数值范围和帧索引与 point map 对齐。
- 现有 `layer_atomic` 输出逐像素不变。
- 现有分割相关测试全部通过；设计开始前的基线为 54 项通过。
- 新模式可完成最小端到端 smoke run，并进入未修改的 `match_segmentation_seq`。

### 12.3 数据验收

固定复测已分析的 17 条 KITTI 诊断轨迹和两个 TUM 室内场景的 12 条诊断轨迹：

- 人工选定的明显独立物体边界召回率相对 Auto 提升；
- 连续墙面、地面、道路和转弯面保持 Auto 的主体连通性；
- 不出现 Geometry 式密集碎片；
- 满足区域数和效率预算；
- KITTI 和 TUM 的轨迹指标不得出现系统性退化。若 ATE 中位数恶化超过 2%，或任一序列恶化超过 5%，该实现不满足交付条件。

每个固定阈值都运行两组配对实验：`split_aux_confirmation=True` 和 `False`。两组必须使用完全相同的输入、法向方法、面积常量、四叶上限和 `split_score_thresh`。报告辅助确认带来的候选接受数变化、区域数变化、运行时间变化和 ATE 变化；不得为关闭组另行调阈值。

阈值评估仅允许围绕 `split_score_thresh` 做固定网格 `{0.05, 0.075, 0.10, 0.15, 0.20}`。最终仍交付一个默认值，不保留多套方法。默认阈值只依据生产设置 `split_aux_confirmation=True` 的碎片预算和轨迹保护条件选择；关闭组只用于影响分析，不参与调参。默认 `0.10` 只有在生产设置违反保护条件时才可由该固定网格中的更保守值替换，并在诊断报告中记录唯一原因。

## 13. 删除与保留

从拟议实现中明确删除：

- 区域平面拟合和残差阈值；
- 曲率阈值；
- RGB、gap、normal 的多级硬门；
- 基于时间的确认或否决；
- 递归二分和多轮 watershed；
- 全图 SLIC；
- 多套实验实现或按数据集切换规则。

明确保留：

- 现有 Auto atom merge；
- 一次法向主导的 marker watershed；
- 一个统一接受分数；
- RGB 或 gap 的替代式确认；
- 一个默认开启的辅助确认消融开关，关闭时退化为纯法向评分；
- 最多四叶；
- 原 `layer_atomic` 基线模式和完整回退能力。

## 14. 完成定义

只有在以下条件全部满足后，`layer_atomic_split` 才算完成：

1. 上述单元、集成、回归和 smoke 测试全部通过；
2. 17 条 KITTI 与 12 条 TUM 诊断轨迹完成统一评估；
3. 区域数、延迟和内存满足预算；
4. 旧 `layer_atomic` 输出保持不变；
5. 最终代码只包含这一套后合并切分方法，不保留被删除的试验条件；
6. 诊断报告能够解释每次候选的接受或拒绝，并包含辅助确认开启/关闭的同阈值配对结果；
7. 公开数值调参项只有统一分数阈值，辅助确认只有一个默认开启的布尔开关。
