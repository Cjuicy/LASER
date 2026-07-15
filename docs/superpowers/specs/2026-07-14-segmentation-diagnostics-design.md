# LASER 三方法分割诊断与可视化系统设计

**日期：** 2026-07-14
**目标分支：** `codex/segmentation-diagnostics`
**基线分支：** `codex/unified-segmentation-methods`
**基线提交：** `98cce5f9f470599aca0cf5a6614f39409d929d58`

## 1. 目标

在不改变已经验证的 LASER `depth`、LASER-Geometry `geometry` 和
`layer_atomic` 三种分割方法行为的前提下，增加一套完整的诊断、指标、自动选区间、
可视化和报告系统，回答以下两个实验问题：

1. 如何保护 layer-atomic 方法在 KITTI 00–10 上已经表现出的整体稳定性；
2. 如何定位 geometry 在 02、04、10 上取得突出 ATE 的机制，并找到 layer-atomic
   可以吸收、但不会破坏 00、05、09 稳定性的改进方向。

诊断系统必须把因果链完整展开：

```text
trajectory error localization
  -> segmentation stages
  -> merge / DSU growth
  -> scale coherence and propagation
  -> temporal correspondence
  -> local trajectory regret
```

系统通过一条云端命令完成两遍自动运行、报告生成、临时空间控制和断点续跑。

## 2. 固定比较配置

四个运行配置顺序执行，禁止默认并行：

| 配置 ID | 模式 | Geometry profile | 正式排名 |
| --- | --- | --- | --- |
| `depth` | `depth` | 不适用 | 是 |
| `geometry_baseline` | `geometry` | `baseline_params` | 是 |
| `layer_atomic` | `layer_atomic` | 不适用 | 是 |
| `geometry_legacy_reference` | `geometry` | `legacy` | 否，仅机制参考 |

正式三方法对比统一使用 Felzenszwalb `scale=300`、`sigma=1.1`、
`min_size=500`。geometry legacy 保留 `200 / 1.0 / 300`，不得混入正式排名。

四个配置必须使用同一个模型 checkpoint。preflight 计算并记录 checkpoint SHA-256；
任一子进程的 hash 不一致时立即停止。固定 random seed，关闭未显式请求的
`cudnn.benchmark`，并在 manifest 记录 Python、PyTorch、CUDA、GPU 和依赖版本。

## 3. 数据集与评价分组

默认运行 KITTI Odometry 00–10 全部序列，以防只优化少数序列：

- Recovery 重点组：`02 / 04 / 10`；
- Stability Guard 重点组：`00 / 05 / 09`；
- 全局保护组：`00–10`；
- 其他序列：`01 / 03 / 06 / 07 / 08`，若出现稳定性回退或指标异常，同样进入
  Pass 2 深度诊断。

GT、预测轨迹和 frame association 必须使用同一套评价路径。逐帧误差使用与 ATE
一致的一次全序列 Sim(3) 对齐，禁止为每个窗口单独重新对齐。

本次正式目标是无回环 `StreamingWindowEngine`。loop-closure 不进入四配置诊断和排名，
避免把分割影响与回环修正混在一起。

## 4. 总体架构

新增包：

```text
inference_engine/diagnostics/
├── schema.py
├── sink.py
├── segmentation.py
├── merge.py
├── scale.py
├── temporal.py
├── trajectory.py
├── metrics.py
├── selection.py
├── storage.py
├── rendering.py
├── report.py
└── orchestrator.py
```

新增入口：

```text
scripts/run_segmentation_diagnostics.py
scripts/verify_segmentation_diagnostics.py
```

主流程：

```text
preflight
  -> Pass 1: all sequences x four configs, scalar metrics only
  -> one global trajectory alignment per sequence/config
  -> deterministic multi-signal interval selection
  -> Pass 2: rerun complete selected sequences, capture selected intervals only
  -> verify artifacts
  -> build CSV / JSON / NPZ / PNG / PLY / static HTML
  -> evaluate Stability Guard and Recovery
  -> clean run-owned temporary files
```

Pass 2 必须从序列开头完整重跑。只输入选中局部区间会丢失历史 scale propagation
状态，因此不允许作为正式诊断结果。完整重跑时只持久化 selected intervals 的重型
trace。

## 5. 算法保护边界

### 5.1 默认路径不变

- 诊断默认关闭；
- 关闭时继续调用当前三种方法的原入口；
- 不改变任何 Felzenszwalb 参数、merge 阈值、DSU union 判断、segment matching、
  scale anchor、scale propagation、Sim(3)、cache aggregation 或 loop closure 行为；
- 诊断 sink 默认是 `None`，不得进入正常路径的数据依赖图。

### 5.2 中间结果获取

- depth 使用现有 `segment_depth_felzenszwalb_rag_stages(...)`；
- geometry 使用现有 `segment_geometry_felzenszwalb_rag_stages(...)`；
- layer-atomic 诊断入口调用现有 depth stages 和 `merge_layer_atoms(...)`；
- layer-atomic merge analyzer 只读地重算 candidate、boundary 和 DSU event，必须验证
  analyzer 重建的 final labels 与正式 `merge_layer_atoms(...)` 结果逐像素一致；
- scale 和 temporal 只允许使用不修改输入的 observation hook；
- traced/untraced 结果必须通过 parity tests。

## 6. 两遍运行设计

### 6.1 Pass 1：全量轻指标

Pass 1 运行全部 00–10 和四个配置，保存：

- 预测轨迹和 GT association；
- sequence/window/frame 标量；
- segmentation counts 和面积分布；
- merge、scale、temporal 聚合统计；
- 资源使用；
- manifest 与完成状态。

Pass 1 不保存全帧 RGB、depth、confidence、point map 或 dense label trace，也不调用
`save_for_viser(...)`。

### 6.2 Pass 2：选中区间重型 trace

Pass 1 完成并执行全序列轨迹对齐后，自动选择 intervals。Pass 2 对涉及的完整序列
重新推理，但只保存选中区间及其前后上下文。

四个配置使用 selected-frame union，保证所有横向对比都在相同帧和相同窗口上完成。

## 7. 指标体系

所有标量记录必须包含 `schema_version`、`run_id`、`config_id`、`sequence_id`、
`window_id`、`global_frame_id` 和有效性标记。不可计算的指标写为空值并记录原因，
禁止用零伪装缺失数据。

### 7.1 轨迹与稳定性

- ATE RMSE；
- RPE translation RMSE：`delta=1 frame`、`all_pairs=True`；
- RPE rotation RMSE：`delta=1 frame`、`all_pairs=True`；
- 全序列一次对齐后的逐帧 translation error；
- `layer_atomic - depth`、`layer_atomic - geometry_baseline`、
  `layer_atomic - geometry_legacy_reference` 的逐帧 regret；
- positive regret 最大值、均值、积分面积和持续长度；
- error onset/change point；
- 00–10 mean ATE、median ATE、胜出序列数和最大单序列回退；
- 00/05/09 Stability Guard；
- 02/04/10 Recovery Gap；
- 对未来 candidate 配置支持 Recovery Score：

```text
Recovery = (ATE_layer_atomic - ATE_candidate)
           / (ATE_layer_atomic - ATE_geometry_reference)
```

分母非正或过小时标记不可用，不生成误导性分数。

当前四配置报告同时固定一份 layer-atomic stability reference。未来加入 candidate 时，
默认 Stability Guard 为：

- 00–10 mean ATE 相对 layer-atomic 恶化不超过 3%；
- 00–10 median ATE 不高于 layer-atomic；
- 00、05、09 任一序列 ATE 恶化不超过 10%；
- 任何未完成或无效序列直接判定 Guard 未通过。

阈值允许通过 CLI 修改，但必须写入 manifest 和报告标题，禁止修改后仍显示为默认规则。

### 7.2 分割结构

- initial atom、coarse layer、final segment count；
- Largest Segment Ratio：`max(area) / valid_pixel_count`；
- Top-1/3/5 area ratio；
- segment area quantiles；
- normalized area entropy 与 effective segment count；
- Atom Compression Ratio：`initial_atom_count / final_segment_count`；
- atoms per final segment：median、P90、max；
- component growth：`final_area / largest_constituent_atom_area`；
- boundary length / area；
- invalid pixel ratio；
- label coverage 和 partition validity。

### 7.3 Merge 与 DSU 渗透

- same-coarse/cross-coarse candidate count；
- same-coarse/cross-coarse accepted count；
- count ratio、boundary-length-weighted ratio、area-weighted ratio；
- normalized gap `G` 的 P10/P25/P50/P75/P90/P95；
- accepted/rejected pair 的 threshold margin；
- union 后 largest component growth curve；
- giant component 首次越过 25%、50%、75% 面积时的 event；
- component merge depth、最长传递链和最大 atom count；
- boundary gap mean/median/P90/P95；
- whole-atom scale 与 boundary-local scale 偏差；
- boundary normal angle 与 mean-depth difference；
- invalid scale/boundary pair 数量和原因。

### 7.4 Immutable atom 风险

- atom 内 normal angular dispersion；
- atom 内 depth-gradient P95；
- 一个 initial atom 内包含的 geometry labels 数量；
- depth atom 对 geometry boundary 的跨越比例；
- 方法间 boundary disagreement；
- 相对某参考方法的 over-merge/over-split area；
- label contingency 与 Variation of Information。

这些指标只表示方法间结构差异，不把任一方法的 labels 当成分割 GT。

### 7.5 Scale coherence 与传播

- overlap 内每个 segment 的原始 scale observations；
- `MAD/IQR/std(log(scale_ratio))`；
- high-dispersion segment count 和 area ratio；
- largest segment scale dispersion；
- `segment_area * log-scale_MAD` pollution risk；
- direct anchor、propagated、identity/fallback 的 count 和 area ratio；
- anchor observation count、有效 weight、support area 和 residual；
- propagation hop length；
- global Sim(3) scale 与 local correction 的 quantiles；
- scale map spatial gradient；
- 配置间 `log(scale_a / scale_b)` difference map。

### 7.6 Temporal matching

- matched segment ratio；
- matched area ratio；
- mean/median/area-weighted temporal IoU；
- unmatched area；
- one-to-many、many-to-one 的 count 和 area；
- correspondence degree；
- segmentation churn；
- segment lifetime；
- temporal transition 与 scale dispersion 的联合统计。

### 7.7 资源指标

- model inference、segmentation、merge、matching、scale propagation 和 rendering 耗时；
- GPU/RAM 峰值；
- temporary/final disk 峰值；
- 每类 artifact 字节数；
- diagnostics 相对关闭状态的额外耗时和空间。

### 7.8 联合分析

- trajectory regret 与 LSR、cross-coarse merge、scale dispersion、temporal IoU 的
  Pearson 和 Spearman 相关；
- 0–3 个窗口 lagged correlation；
- 样本量和缺失率；
- 每个失败区间按 robust z-score 列出最异常的五项指标；
- 报告使用“支持/不支持假设”的措辞，禁止自动生成因果结论。

## 8. 自动选区间

### 8.1 候选来源

- Recovery 组每序列 trajectory regret Top-3；
- Guard 组每序列 layer-atomic 优势 Top-2；
- error change point Top-2；
- merge、immutable-atom、scale、temporal 四个指标族各自异常 Top-2；
- 每个重点序列最多两个 matched controls；
- 其他序列若 Stability Guard 失败或任一指标 robust z-score 大于 3.5，至少加入一个
  区间。

robust z-score 使用 median/MAD；MAD 为零时回退到 IQR，再回退到带 epsilon 的标准差。

### 8.2 区间形成

- 指标首先聚合到 window；
- change point 或异常 window 前后各扩展两个窗口；
- 重叠或相隔不超过一个窗口的候选合并；
- controls 使用 GT speed、turn magnitude 和 median confidence 的最近邻匹配，仅保留
  control 原生窗口（异常案例仍前后扩展两个窗口），并要求
  trajectory regret 和综合异常分数不高于该序列中位数。

### 8.3 最大数量与多样性

默认上限为 48 个 intervals。超过上限时按以下约束做确定性选择：

- 每个 Recovery 序列至少保留一个 regret 案例和一个 control；
- 每个 Guard 序列至少保留一个 guard-success 案例和一个 control；
- merge、immutable-atom、scale、temporal 各指标族若有候选，至少保留两个；
- 发生 Stability Guard breach 的其他序列至少保留一个；
- 剩余名额按 trajectory、merge、scale、temporal robust score 的加权综合分数排序；
- 所有 tie 使用 sequence、start frame 和 reason code 排序，保证复现。

默认综合权重为 trajectory 0.40、merge/atom 0.20、scale 0.25、temporal 0.15。
报告保留各分量和 selection reason，不能只展示总分。

## 9. 可视化与报告

采用已经确认的两级静态 HTML 报告，不依赖外网资源。

### 9.1 Overview

- run/config/data/commit 信息；
- Pass 状态和 50 GiB 空间状态；
- Stability Guard 和 Recovery；
- 00–10 ATE/RPE 表与热力图；
- 全局对齐逐帧 error/regret；
- 异常时间线和 selected interval ranking；
- 指标相关性与 lag 分析；
- incomplete/invalid data 警告。

### 9.2 Case Detail

每个 selected case 包含：

- RGB、depth、confidence；
- 三方法 final segmentation；
- depth initial/coarse/final；
- geometry initial/merged；
- layer-atomic initial/coarse/final；
- boundary G、accepted/rejected、same/cross-coarse merge；
- DSU component growth 与 giant-component onset；
- atom normal dispersion、boundary normal angle、boundary-local scale；
- direct/propagated/fallback source map；
- 三方法 scale map、log-scale difference、scale dispersion；
- temporal correspondence 和 one-to-many/many-to-one；
- local trajectory 和 error/regret timeline；
- 自动诊断摘要及证据指标。

### 9.3 每案例 artifact

```text
overview.png
segmentation_stages.png
merge_diagnostics.png
scale_diagnostics.png
temporal_matching.png
pointcloud_top.png
pointcloud_side.png
pointcloud.ply
trace.npz
metrics.json
```

PLY 只保存 selected frames，并提供可配置点密度。top/side preview 使用确定性的 NumPy
投影，避免依赖云端图形桌面或 OpenGL context。

## 10. 输出结构

```text
diagnostics/<run_id>/
├── manifest.json
├── run.lock
├── logs/
├── pass1/
│   ├── trajectories/
│   ├── frame_metrics/
│   ├── window_metrics/
│   └── sequence_summary.csv
├── selection/
│   ├── selected_intervals.json
│   └── selection_reasons.csv
├── pass2/
│   ├── traces/
│   ├── panels/
│   └── comparison_cases/
├── report/
│   ├── index.html
│   ├── assets/
│   ├── metrics.csv
│   └── summary.json
└── state/
    └── checkpoints/
```

## 11. 存储设计

### 11.1 限额

- temporary hard limit：50 GiB；
- warning threshold：40 GiB；
- minimum free space reserve：10 GiB；
- 四个配置顺序运行；
- 每个 `config x sequence x pass` 完成后立即清理该阶段 run-owned temp。

### 11.2 Metrics-only cache

注册线程仍在内存保留当前和前一个窗口的完整数据，但 Pass 1 磁盘只保存 pose shard、
frame/window metrics、manifest 和必要的 scale/temporal scalars。ATE 走 pose-summary
路径，跳过 full prediction aggregation 和 `save_for_viser(...)`。

### 11.3 Pass 2 编码

- labels：`uint16`，超过范围时 `uint32`；
- RGB：`uint8`；
- bool mask：bit-packed；
- visualization-only dense float：`float16`；
- scale observations 和精确统计：`float32/float64`；
- NPZ：compressed；
- point cloud：selected frames only。

核心 metrics、selection reasons、PNG 和 JSON 不得因空间不足而静默丢弃。预计超过硬
上限时必须在 preflight 停止；运行中接近硬上限时完成当前原子写入、清理已完成 temp、
保存 checkpoint 并以可恢复错误退出。

## 12. 可靠性与错误处理

- 所有 artifact 先写 `.partial`，验证后原子 rename；
- manifest 保存 Git commit、参数、配置哈希、数据路径、帧数、schema version 和状态；
- 相同 output root 使用 lock，防止并发写入；
- `SIGINT/SIGTERM` 刷新状态并只清理当前 run-id 拥有的 temp；
- resume 校验 commit/config/data fingerprint；不一致时拒绝混用；
- 缺失 GT、空序列、无效 labels、NaN scale、artifact checksum 错误必须明确记录；
- 单配置失败不伪装为完整成功；最终命令返回非零，但保留已成功部分和 incomplete report；
- 不删除用户目录、其他 run 或无法验证 ownership 的文件。

## 13. CLI

`eval_launch.py` 增加统一参数：

```text
--segment_mode
--normal_method
--geometry_seg_profile
--diagnostics
--diagnostic_run_dir
--diagnostic_pass
--diagnostic_selected_intervals
--diagnostic_cache_policy metrics-only
--model_ckpt
```

云端总命令：

```bash
python scripts/run_segmentation_diagnostics.py \
    --dataset kitti_odometry \
    --dataset-root /path/to/KITTI_Odometry/dataset \
    --model-ckpt /path/to/weights/model.safetensors \
    --output-root /path/to/diagnostics \
    --sequences 00 01 02 03 04 05 06 07 08 09 10 \
    --max-temp-gib 50 \
    --warn-temp-gib 40 \
    --min-free-gib 10 \
    --resume
```

辅助模式：

```text
--dry-run
--pass 1
--pass 2
--report-only
--run-id
--selected-interval-limit
--device
--seed
```

总调度器按配置启动独立子进程，释放 GPU/RAM 后才运行下一配置。

## 14. 执行日志

日志必须使用明确阶段前缀：

```text
[preflight]
[pass1 02 layer_atomic]
[trajectory alignment]
[interval selection]
[pass2 02 layer_atomic]
[artifact verification]
[temp cleanup]
[report complete]
```

每阶段显示 sequence、config、窗口进度、预计/实际空间、空间峰值、清理量、剩余空间、
resume 状态和输出位置。

## 15. 测试策略

### 15.1 行为保持

- 当前完整测试套件继续通过；
- 三模式 diagnostic off labels 与基线逐像素一致；
- layer-atomic trace 重建 labels 与正式 labels 完全一致；
- traced/untraced scale mask 一致；
- observation hook 不改变 graph 边、IoU、cache 或传播结果。

### 15.2 单元测试

- schema round-trip 和 version validation；
- segmentation/merge/scale/temporal 指标公式；
- synthetic giant component、cross-coarse、boundary-local scale；
- synthetic trajectory 全局对齐、regret 和 Recovery；
- checkpoint hash、固定 seed 和 evaluation signature；
- onset、Top-K、control、interval merge、diversity selection 和 deterministic tie；
- storage warning/hard limit、lock、partial、checksum、resume、signal cleanup；
- renderer 输出尺寸、格式和缺失数据标识；
- report 中所有链接与 artifact 可读取。

### 15.3 集成测试

- CPU synthetic 一条命令完成 Pass 1、selection、Pass 2、verification 和 report；
- `scripts/verify_segmentation_diagnostics.py` 不依赖模型权重；
- 云端首先运行 KITTI 04；
- 04 成功后运行完整 00–10；
- dry-run 的空间估计必须低于 50 GiB 才允许正式启动。

## 16. 完成标准

以下条件全部满足后才可声称完成：

1. 新分支基于统一分支固定提交；
2. diagnostics off 不改变三方法和下游结果；
3. 当前测试、新 parity tests、synthetic end-to-end 全通过；
4. 一条命令支持 dry-run、两遍执行、resume、report-only；
5. 50 GiB hard limit 和 40 GiB warning 已由测试覆盖；
6. 四配置顺序运行，三条正式主线与 legacy reference 清楚区分；
7. Overview 和 Case Detail 报告完整生成；
8. 不完整数据不会显示为正式成功；
9. 云端文档包含 KITTI 04 验证和 00–10 完整命令；
10. 最终分支提交并推送到 GitHub。

## 17. 非目标

- 本次不实现 structural veto、禁止 cross-coarse merge 或其他候选算法改进；
- 不根据诊断结果自动修改阈值；
- 不把 geometry labels 当作分割 GT；
- 不声称相关性证明因果；
- 不改变 loop-closure 算法；
- 不在默认路径保存全量诊断数据。
