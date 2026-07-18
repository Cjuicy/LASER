# LASER Layer-Atomic Split 三方法诊断设计

**日期：** 2026-07-18  
**目标分支：** `codex/segmentation-diagnostics`  
**新方法来源分支：** `codex/auto-post-merge-split`  
**共同基线提交：** `98cce5f9f470599aca0cf5a6614f39409d929d58`

## 1. 目标

把 `codex/auto-post-merge-split` 中的 `layer_atomic_split` 正式实现集成到
`codex/segmentation-diagnostics`，并把诊断实验收敛为以下三个正式方法：

1. `depth`；
2. `geometry_baseline`；
3. `layer_atomic_split`。

诊断需要回答：

- 三种方法在 KITTI Odometry 00–10 上的 ATE/RPE 分别如何；
- 新方法的轨迹误差从哪些帧和窗口开始出现；
- 新增 post-merge split 在哪些位置改变了分割；
- 哪些 split 事件与轨迹改善、恶化或异常碎片化同时出现；
- 对应位置的 RGB、depth、三方法 labels、split 前后 labels、split 决策和局部轨迹证据是什么。

KITTI Odometry 不提供本实验所需的分割真值。因此报告中的“分得好/差”由分割结构证据
与轨迹误差证据共同定义，只报告相关关系和假设支持程度，不自动宣称 split 导致了 ATE
变化。

## 2. 固定实验配置

诊断只运行三个配置，按下列顺序串行执行：

| 配置 ID | `segment_mode` | 正式排名 | 固定参数 |
| --- | --- | --- | --- |
| `depth` | `depth` | 是 | Felzenszwalb `300 / 1.1 / 500` |
| `geometry_baseline` | `geometry` | 是 | Felzenszwalb `300 / 1.1 / 500`，`normal_method=cross` |
| `layer_atomic_split` | `layer_atomic_split` | 是 | `normal_method=cross`，`split_score_thresh=0.10`，`split_aux_confirmation=true` |

旧 `layer_atomic` 和 `geometry_legacy_reference` 从完整推理、正式排名、区间选择、案例目录、
报告图例和操作文档中删除。旧 layer-atomic labels 只保留为新方法内部的 `pre_split_labels`
中间结果，不作为第四个轨迹配置。

三个配置必须共享 checkpoint、KITTI 数据指纹、窗口大小、overlap、随机种子、设备设置和
全序列 Sim(3) 对齐口径。RPE 继续使用 `delta=1 frame, all_pairs=True`。

## 3. 集成策略

采用原生集成，而不是复制部分代码或离线重放：

1. 将新方法分支中 post-merge split、`layer_atomic_split` engine 路由、RGB 传递、CLI
   参数和相应测试完整集成到诊断分支；
2. 抽出共享的 staged split 接口，返回 pre-split labels、post-split labels、atom metadata
   和 split diagnostics；
3. 正式 `segment_point_map_layer_atomic_split(...)` 与诊断重算共同调用 staged 接口；
4. 正式路径仍只返回最终 labels，未启用诊断时不写 trace，也不引入诊断数据依赖；
5. 诊断路径用正式 engine 产生的 labels 做逐像素 parity 校验，防止观测逻辑与实际算法
   分叉。

共享 staged 接口至少提供：

```text
pre_split_labels
final_labels
atom_labels
atom_scales
split_diagnostics
```

其中 `split_diagnostics` 保留现有总计字段，并为重型案例提供足以还原 parent 级决策的紧凑
数组。Pass 1 只写标量聚合，Pass 2 才保存 dense labels 和决策图。

## 4. 两遍数据流

### 4.1 Preflight

Preflight 在加载模型前验证：

- KITTI 00–10 图像和 pose 布局；
- checkpoint SHA-256 与数据集指纹；
- 三配置清单和固定 split 参数；
- 输出 schema 与配置哈希；
- 临时空间、最终空间和可用空间阈值。

配置哈希必须包含 `normal_method`、`split_score_thresh` 和
`split_aux_confirmation`。旧四配置 manifest 或参数不同的结果不得通过 `--resume` 混入
本次运行。

### 4.2 Pass 1：全量轻量诊断

对每个配置和每条序列运行完整推理，保存：

- 预测轨迹、GT association 和一次全序列对齐后的逐帧 translation error；
- ATE RMSE、RPE translation RMSE、RPE rotation RMSE；
- 每帧和每窗口分割结构标量；
- `layer_atomic_split` 的 split 活跃度和决策标量；
- merge、scale、temporal 与资源聚合统计；
- sequence checkpoint、JSONL 和校验和。

Pass 1 不保存全帧 RGB、point map、dense labels、scale map 或决策图。

### 4.3 误差对齐和选区间

每条序列完成三种方法的全序列对齐后，计算：

```text
split_minus_depth_regret    = error(layer_atomic_split) - error(depth)
split_minus_geometry_regret = error(layer_atomic_split) - error(geometry_baseline)
```

regret 同时保留逐帧值和窗口均值、最大值、正值面积、持续长度及 change point。窗口级记录把
regret 与 split 活跃度、分割结构、scale dispersion、temporal churn、车速、转弯强度和
confidence 对齐。

删除旧 `layer_atomic` 后，Stability Guard 改为以 `depth` 为稳定性基线：

- 00–10 mean ATE 相对 depth 恶化不超过 3%；
- 00–10 median ATE 不高于 depth；
- 00、05、09 任一序列相对 depth 的 ATE 恶化不超过 10%；
- 任一预期序列缺失或无效时 Guard 失败。

02、04、10 的 Recovery 以 depth 到 geometry 的差距为参照：

```text
Recovery = (ATE_depth - ATE_layer_atomic_split)
           / (ATE_depth - ATE_geometry_baseline)
```

只有当分母为正且大于数值容差时才生成 Recovery；否则写为不可用并说明 geometry 没有形成
有效的改善参照。该定义只衡量新方法恢复了多少 depth-to-geometry ATE gap，不把 geometry
当作分割真值。

选区间器从以下来源建立 deterministic union：

- 新方法相对任一基线恶化最大的区间；
- 新方法相对任一基线改善最大的区间；
- 逐帧误差或 regret 的主要 change point；
- accepted split 数量、changed-pixel ratio 或区域增长异常的区间；
- split 活跃但轨迹误差变化很小的区间；
- 同序列、相近运动条件下没有 accepted split 的 matched control。

所有三种方法使用同一个 selected-frame union。区间数量继续受 `--max-selected` 和磁盘预算
限制，Recovery、Stability Guard 和其余异常序列仍需得到确定性覆盖。

### 4.4 Pass 2：重型案例

Pass 2 从每条被选中序列的开头完整重跑，以保留 scale propagation 和 temporal state。
只有 selected intervals 及上下文帧写入：

- RGB、depth、confidence 和 point map；
- 三方法 final labels；
- 新方法 initial、coarse、pre-split 和 final labels；
- atom labels、atom scales；
- split changed-region map、parent/child map、score map 和拒绝原因图；
- scale、temporal 和局部轨迹 trace。

## 5. Split 专项指标

`layer_atomic_split` 在原有分割指标外增加：

- split parent、proposed、accepted、added-region 数量；
- no-markers、small-child、low-score 拒绝数量；
- split score mean、quantiles 和 max；
- split runtime；
- pre/post segment count 和增加比例；
- changed-pixel count 与面积比例；
- 每个 parent 的 child count 和最小 child fraction；
- parent normal dispersion 与 child 面积加权 dispersion；
- normal-dispersion gain；
- 最大区域占比、面积熵和 effective segment count 的 pre/post delta；
- 与 `geometry_baseline` 的 boundary disagreement 和 Variation of Information 的 pre/post
  delta；
- 过度碎片化信号：单帧区域增长、tiny-child area ratio 和高频边界增长。

不可计算的标量写为 null 并附原因。不得用 0 代替缺失值。

## 6. “好/差”案例的操作定义

报告采用以下标签，标签只用于排查和排序：

- `trajectory_improvement`：新方法相对至少一个基线的局部 regret 明显下降；
- `trajectory_degradation`：新方法相对至少一个基线的局部 regret 明显上升；
- `split_structural_improvement`：accepted split 降低 parent normal dispersion 或 geometry
  boundary disagreement，同时未触发过度碎片化；
- `fragmentation_risk`：区域数、tiny-child area 或边界密度异常上升；
- `split_no_trajectory_effect`：split 活跃但局部 regret 接近该序列中位水平；
- `matched_control`：相近运动和置信度条件下无 accepted split 的对照区间。

一个案例可以同时拥有多个标签。系统列出共同出现的证据，不把结构改善自动等同于轨迹
改善，也不把轨迹恶化自动归因于单个 split。

## 7. 报告设计

Overview 包含：

- KITTI 00–10 三方法 ATE/RPE 表、排名和 heatmap；
- mean/median ATE、胜出序列数和最大单序列回退；
- Stability Guard 与 Recovery 摘要；
- 每条序列两条 regret timeline；
- split 活跃度与轨迹误差的时间对齐图；
- split score、changed-pixel ratio、区域增长与 regret 的散点图和相关性；
- improvement、degradation、fragmentation、no-effect 和 control 案例入口。

案例页包含：

- RGB、depth、confidence；
- `depth`、`geometry_baseline`、`layer_atomic_split` final segmentation；
- 新方法 pre-split、post-split 和 changed-region overlay；
- split score、parent/child、接受/拒绝原因；
- geometry boundary disagreement pre/post；
- scale/temporal 证据；
- 一次全序列对齐下的局部 error/regret timeline；
- 选中原因、证据摘要和 `UNAVAILABLE` 项。

报告名称、颜色、CSV 字段和目录统一使用 `layer_atomic_split`，不得残留把旧
`layer_atomic` 当作正式配置的图例或排名字段。

## 8. Schema、存储与失败策略

- 提升 diagnostics schema version；
- manifest 精确记录三配置和固定 split 参数；
- 正常三配置文件、checkpoint、trace 和报告继续使用原子写入与 SHA-256 校验；
- 40/50 GiB warning/hard limit 和最小可用空间保护保持有效；
- `.partial`、run ownership、stale lock 和 sequence resume 语义保持不变；
- 三种方法任一序列轨迹缺失或无效时，该序列不产生正式排名，完整验收失败；
- 缺少 RGB、point map 或必要 split trace 时，案例生成失败并明确指出文件；
- 可选 scale/temporal trace 缺失时显示 `UNAVAILABLE`，不伪造单位尺度或空热图；
- Pass 2 不允许只从选中局部帧开始运行；
- 三配置的 selected-frame union 不完整时拒绝生成比较案例。

## 9. 测试策略

### 9.1 新方法回归

- 保留 post-merge split 的候选、接受、拒绝、尺度不变性、确定性和区域预算测试；
- 保留 `layer_atomic_split` engine、CLI、RGB 传递和参数验证测试；
- 验证共享 staged 接口与原公开入口的 final labels 相同；
- 验证诊断开关关闭时，新方法行为与来源分支一致。

### 9.2 诊断单元和集成测试

- `DIAGNOSTIC_PROFILES` 严格等于三个固定配置；
- diagnostic recomputation 与正式 final labels 逐像素一致；
- pre-split、post-split、changed-region 和 split 统计正确；
- synthetic trajectory 正确产生两条 regret；
- improvement、degradation、change point、split anomaly 和 matched control 能被选中；
- 报告、CSV、颜色、目录和 summary 不出现旧正式方法；
- 旧 manifest 和不同 split 参数的 resume 被拒绝；
- 三配置 artifact union 缺失时验收失败；
- CPU 无权重 verifier 覆盖 schema、parity、selection、storage、rendering 和 report。

### 9.3 完成前验证

- 运行新方法和 diagnostics 相关测试；
- 运行完整 pytest 回归；
- 运行 CPU 无权重 verifier；
- 检查 `git diff --check`；
- 文档提供 KITTI 04 小规模 GPU 命令和 KITTI 00–10 正式命令。

本地没有 KITTI、checkpoint 和 GPU 时，不声称已经获得真实 ATE。真实 ATE 由云端正式
命令产生；本地验收保证配置、指标、选区间、trace 和报告链路可执行且一致。

## 10. 成功标准

实现完成后必须满足：

1. 目标分支可直接运行 `depth / geometry_baseline / layer_atomic_split` 三方法诊断；
2. 旧 `layer_atomic` 和 `geometry_legacy_reference` 不再作为完整配置出现；
3. 新方法正式路径与诊断重算 final labels 逐像素一致；
4. summary 和报告包含三方法 ATE/RPE、逐帧误差与两条 regret；
5. selected cases 同时覆盖改善、恶化、change point、split anomaly 和 control；
6. 案例页能定位 split 前后变化、决策依据和对应局部轨迹误差；
7. schema、断点恢复、磁盘保护和 artifact 校验仍然有效；
8. 所有本地自动化测试和 CPU verifier 通过；
9. 云端文档能先运行 KITTI 04 验证，再运行 00–10 正式诊断。

## 11. 非目标

- 不引入第四种完整轨迹方法；
- 不运行旧 geometry legacy 配置；
- 不做 split threshold 或 auxiliary confirmation 的多参数消融；
- 不把 geometry labels 当作分割真值；
- 不引入 loop closure；
- 不修改 Sim(3)、RPE、scale propagation 或 temporal matching 的算法口径；
- 不在报告中自动作因果结论。
