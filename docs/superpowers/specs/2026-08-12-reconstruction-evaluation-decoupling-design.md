# LASER 重建与评测解耦设计

- 日期：2026-08-12
- 状态：已确认
- 仓库：`Cjuicy/LASER`
- 实现分支：`codex/laser-paper-pointmap-eval`
- 设计基线：`d0fa2b9d1433185d7a71dc19504eeed92785dfe3`

## 1. 摘要

本设计将 LASER 的普通 PI3 预测、三种分割、锚点传播、三种重建模式、
ATE 评测和点云质量评测拆成职责明确的模块。

最终支持两个正式实验矩阵：

```text
ATE:
  depth / geometry / atomic
  ×
  no_loop / traditional / corrected

Point cloud:
  depth / geometry / atomic
  ×
  no_loop only
```

三种重建模式共享普通 PI3 预测缓存、分割策略接口、LASER 原始
`AnchorPropagator`、相邻配准工具、回环候选检测组件和统一产物格式，但保留各自
不同的算法时序。评测器只读取已经完成的重建产物，不允许重新执行 PI3、分割、
锚点传播、相邻配准或回环。

本次允许删除旧命令、旧配置和旧内部接口，不提供兼容适配层。架构解耦和实验
灵活性优先于向后兼容。

## 2. 背景与现有问题

当前主推理框架已经有：

- 统一的 PI3 adapter 和 ordinary prediction cache；
- `depth`、`geometry`、`atomic` 三种分割策略；
- LASER 原始锚点传播实现；
- `traditional` 和 `corrected` 两套回环实现。

主要问题集中在重建和评测边界。

### 2.1 点云 evaluator 内复制了重建流水线

`mv_recon/eval.py` 直接创建分割策略和 `AnchorPropagator`，并调用
`mv_recon/paper_streaming.py` 重新完成窗口配准、temporal graph、锚点传播和点图
聚合。因此它同时承担了：

```text
实验编排 + 重建 + 指标计算 + 结果保存
```

这形成了独立于正式 pipeline 的第二条重建路径。

### 2.2 loop off 仍然依赖回环方法

当前配置使用：

```yaml
loop:
  enabled: false
  method: traditional
```

即使关闭回环，`PipelineRunner` 仍通过所选 loop strategy 创建窗口引擎、调用
`optimize()` 和 `aggregate()`。`WindowCache` 也包含 `loop_method` 和
`loop_state`。因此 no-loop 在配置、类型和执行层都没有真正独立。

### 2.3 Traditional 和 Corrected 的时序差异不能被抹平

两种方法不仅回环约束不同，窗口阶段的修正时序也不同：

- Traditional 计算相邻 Sim(3) 和 anchor scale mask 后延迟应用，回环完成后统一
  聚合；
- Corrected 边预测边应用相邻 Sim(3) 和 anchor scale，最后对在线结果应用回环
  optimization delta。

如果把两者强行改为同一个已经立即修正的基础重建，再将回环视为简单后处理，
会改变 Traditional 算法。正确边界是三种完整重建模式共享组件，而不是共享同一
时序。

### 2.4 点云协议混入运行实现配置

当前 `mv_recon/protocol.py` 同时锁定 segmentation、anchor、prediction cache、
registration 和 loop method。点云指标配置甚至需要在 `loop.enabled=false` 时验证
`loop.method=traditional`。评测协议因此知道过多重建细节。

### 2.5 点云回环实验超出正式范围

当前分支包含 Traditional 回环点图实验及专用比较脚本，但正式目标明确规定点云
质量只评测 no-loop。继续维护这条路径会重新把回环实验耦合进点云 evaluator。

## 3. 目标

1. PI3 ordinary prediction 通过一个 provider 统一输出并支持跨实验缓存复用。
2. `depth`、`geometry`、`atomic` 是三个可自由选择的分割策略。
3. 三种分割之后使用同一份 LASER 原始锚点传播实现。
4. `no_loop`、`traditional`、`corrected` 是三种互斥的完整重建模式。
5. Traditional 保留延迟修正、回环优化后统一应用的原始逻辑。
6. Corrected 保留在线修正、回环后应用整体 delta 的原始逻辑。
7. no-loop 保留 LASER Table 4 使用的增量点图组装语义。
8. ATE 支持三种分割乘三种重建模式的九种组合。
9. 点云质量只支持三种分割乘 no-loop 的三种组合。
10. window size 和 overlap 由重建配置自由设置，只验证
    `size > overlap >= 1`。
11. 所有重建模式输出同一个强类型 `ReconstructionArtifact`。
12. ATE 和点云 evaluator 只读取产物并计算指标。
13. 删除旧回环点云实验、旧 evaluator 重建接口和被替代的旧配置。
14. 最终提供可直接复制的云端克隆、环境准备、测试和评测命令。

## 4. 非目标

- 不修改 PI3 模型或权重。
- 不设计新的 segmentation 算法。
- 不修改 LASER `AnchorPropagator` 的数学公式。
- 不修改 SALAD 候选检测语义。
- 不修改 Traditional 或 Corrected 的回环约束、优化器数学和结果应用语义。
- 不让点云评测支持任何回环模式。
- 不保留旧 CLI、旧 YAML 字段或旧 evaluator 内部 API 的兼容层。
- 不构建通用 DAG 或任意插件依赖系统。
- 不要求不同 window size/overlap 的实验结果相同。

## 5. 备选方案

### 5.1 最小搬迁

只把 `paper_streaming.py` 移入 core，继续让 loop strategy 创建窗口引擎并负责
no-loop 聚合。改动较小，但 no-loop 仍依赖回环策略，不能解决类型和执行层耦合。

### 5.2 三种完整重建模式，共享组件（选定）

将 `no_loop`、`traditional`、`corrected` 定义为三种完整模式。共享 PI3 provider、
分割策略、锚点传播和工具函数，但保留每种模式的状态机和时序。所有模式输出同一
产物，再由纯 evaluator 消费。

该方案既保留算法，又恢复清晰边界。

### 5.3 通用 DAG/插件系统

将每个阶段建成任意组合节点。灵活性最高，但会引入依赖解析、节点生命周期、
产物协商和缓存失效等额外机制，超过当前实验矩阵的需求。

## 6. 总体架构

```text
ImageManifest
     │
     ▼
Pi3PredictionProvider
     │  WindowPrediction
     ▼
SegmentationStrategy selection
depth | geometry | atomic
     │
     ▼
ReconstructionMode dispatch
┌────────────┬─────────────────┬────────────────┐
│ no_loop    │ traditional     │ corrected      │
│ incremental│ deferred        │ online         │
│ no detector│ loop correction │ loop correction│
└────────────┴─────────────────┴────────────────┘
     │
     ▼
ReconstructionArtifact
     ├───────────────┐
     ▼               ▼
TrajectoryEstimate  PointMapEstimate
     │               │
     ▼               ▼
ATE/RPE Evaluator   PointCloud Evaluator
all three modes     no_loop only
```

`LoopDetector` 是独立组件，只被 Traditional 和 Corrected 调用。它不属于 PI3
prediction、segmentation、anchor propagation 或 evaluation。

## 7. 核心数据契约

### 7.1 WindowPrediction

普通 PI3 provider 的公共输出：

```python
@dataclass(frozen=True)
class WindowPrediction:
    spec: WindowSpec
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    images: torch.Tensor
    prediction_key: str
```

必须满足：

- `local_points`: `(window_frames, H, W, 3)`；
- `camera_poses`: `(window_frames, 4, 4)`；
- `confidence`: `(window_frames, H, W)`；
- `images`: `(window_frames, 3, H, W)`；
- 所有数值有限；
- 帧数严格匹配 `WindowSpec`。

ordinary prediction cache identity 只包含影响 PI3 普通预测的输入，不包含：

- segmentation method；
- reconstruction mode；
- anchor 参数；
- loop 参数；
- evaluation 参数。

因此十二个正式组合可以共享同一组普通预测。

### 7.2 ReconstructionArtifact

三个模式的统一公共输出：

```python
@dataclass(frozen=True)
class ReconstructionArtifact:
    schema_version: int
    frame_ids: tuple[int, ...]
    local_points: torch.Tensor
    global_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_method: SegmentationMethod
    reconstruction_mode: ReconstructionMode
    prediction_key: str
    diagnostics: ReconstructionDiagnostics
```

产物必须严格覆盖每个输入帧一次。`frame_ids`、点图、相机位姿和 confidence 的
首维必须一致。

模式专属的窗口 trace、回环候选、约束和解只允许存在于模式内部或单独诊断产物，
不能成为 evaluator 的输入要求。

### 7.3 Evaluator 视图

```python
@dataclass(frozen=True)
class TrajectoryEstimate:
    frame_ids: tuple[int, ...]
    camera_poses: torch.Tensor


@dataclass(frozen=True)
class PointMapEstimate:
    frame_ids: tuple[int, ...]
    global_points: torch.Tensor
    confidence: torch.Tensor
    reconstruction_mode: ReconstructionMode
```

ATE evaluator 只接受 `TrajectoryEstimate`。点云 evaluator 只接受
`PointMapEstimate`，并验证 mode 为 `no_loop`。

## 8. 共享预测流

`prediction_stream.py` 负责：

```text
canonical WindowSpec order
→ ordinary prediction provider
→ validated WindowPrediction stream
```

它不负责：

- segmentation；
- temporal graph；
- adjacent registration；
- anchor propagation；
- loop detection；
- aggregation。

当前后台 inference queue、bounded backpressure、worker exception propagation 和
prediction-store entry lock 可保留并下沉为这一公共层的实现细节。三个重建模式不
再复制 provider 调度和线程生命周期。

## 9. 分割与锚点传播

分割方法保持现有 registry：

```text
depth
geometry
atomic
```

公开策略只产生逐帧 labels 和诊断。temporal graph 构建和锚点传播由重建模式在
正确的窗口状态上调用。

不同模式不能预先共享一份 labels，因为它们调用 segmentation 时的 point maps
可能不同：Traditional 使用尚未最终应用修正的窗口状态，Corrected/no-loop 使用
在线修正后的窗口状态。共享的是策略实现和接口，不是错误地共享计算结果。

三个模式都必须使用同一个 `inference_engine/anchor_propagation.py` 中的 LASER
`AnchorPropagator`。不得复制、改写或在 evaluator 内重新调用该算法。

## 10. 三种重建模式

### 10.1 no_loop

no-loop 使用 LASER Table 4 的增量顺序：

```text
读取当前窗口普通 PI3 prediction
→ 使用第一窗口估计的参考内参反投影深度
→ 与已经修正的上一窗口做 adjacent Sim(3)
→ 立即应用相邻尺度和位姿
→ 调用选定 SegmentationStrategy
→ 构建 temporal graph
→ 调用 AnchorPropagator
→ 立即应用 layer scale mask
→ 修正后的窗口成为下一窗口的参考
→ 裁掉重复 overlap
→ 每帧只聚合一次
→ 生成 global points 和 trajectory
```

no-loop 不 import 或构造：

- LoopDetector；
- JointAlignmentEstimator；
- LoopConstraint；
- Sim3LoopOptimizer；
- Traditional/Corrected 方法。

现有 `mv_recon/paper_streaming.py` 的算法语义迁入该模式。Depth baseline 的
窗口顺序、相邻置信度选择、分割、anchor scale 和 overlap 裁剪需要通过数值回归
保持一致。

### 10.2 traditional

Traditional 保留原有延迟修正逻辑：

```text
逐窗口读取普通 PI3 prediction
→ 计算 adjacent Sim(3)，记录但暂不应用
→ 调用选定 SegmentationStrategy
→ 构建 temporal graph
→ 计算 anchor scale mask，记录但暂不应用
→ 所有窗口处理完成
→ LoopDetector 检测候选
→ joint A/B alignment
→ traditional common-frame loop constraint
→ Sim(3) graph optimization
→ 最终聚合时统一应用 optimized transform 和 depth scale
→ 输出 trajectory 和 point map
```

`TraditionalWindowEngine` 的状态机迁入
`reconstruction/modes/traditional.py`，不保留独立引擎框架。以下数学语义是硬性
回归要求：

- candidate-to-window 映射和去重；
- common-frame constraint 计算；
- 原始相邻 transform 组成；
- optimizer 输入和输出解释；
- anchor scale 与最终变换的应用顺序；
- overlap 聚合顺序。

### 10.3 corrected

Corrected 保留在线修正逻辑：

```text
读取当前窗口普通 PI3 prediction
→ 与已经修正的上一窗口做 adjacent Sim(3)
→ 立即应用相邻尺度和位姿
→ 调用选定 SegmentationStrategy
→ 构建 temporal graph
→ 调用 AnchorPropagator
→ 立即应用 layer scale mask
→ 修正后的当前窗口成为下一窗口参考
→ 所有窗口处理完成
→ LoopDetector 检测候选
→ joint A/B alignment
→ corrected local loop constraint conversion
→ sequential-edge optimization
→ 计算 optimized/original delta
→ 将 delta 应用于在线结果
→ 输出 trajectory 和 point map
```

`CorrectedWindowEngine` 的状态机迁入
`reconstruction/modes/corrected.py`，不保留独立引擎框架。以下语义是硬性回归
要求：

- `sim3_abs` 和 `sim3_edge` 的定义；
- joint alignment 到 local constraint 的坐标转换；
- sequential edge optimizer 输入；
- optimized absolute transform 的恢复；
- correction delta 只应用一次；
- overlap 聚合顺序。

## 11. 回环检测与 evidence 边界

回环检测接口：

```python
class LoopDetector(Protocol):
    def detect(
        self,
        manifest: ImageManifest,
        images: torch.Tensor,
    ) -> tuple[LoopCandidate, ...]: ...
```

调用关系：

```text
no_loop      → no call
traditional  → one detector call after window processing
corrected    → one detector call after window processing
```

ordinary PI3 prediction 只生成并缓存一次。回环约束需要的 joint A/B inference 由
独立 `LoopEvidenceProvider` 提供，只服务回环约束构造：

```text
OrdinaryPredictionProvider → reconstruction windows
LoopEvidenceProvider       → loop constraints only
```

joint evidence 不允许修改 ordinary cache 或重新执行 segmentation/anchor propagation。

## 12. 配置设计

旧的 `loop.enabled + loop.method` 被一个互斥模式字段替代：

```yaml
version: 2

model:
  name: pi3
  checkpoint: weights/model.safetensors
  inference_device: cuda
  process_device: cpu
  dtype: bfloat16

prediction_cache:
  root: inference_cache/predictions
  mode: auto

window:
  size: 20
  overlap: 5

segmentation:
  method: depth
  confidence_keep_ratio: 0.5
  depth_merge_threshold: 0.1
  temporal_iou_threshold: 0.3
  felzenszwalb:
    scale: 300
    sigma: 1.1
    min_size: 500
  geometry:
    normal_method: cross
    normal_threshold_degrees: 20.0
  atomic:
    split_mode: conservative
    split_score_threshold: 0.10

anchor_propagation:
  enabled: true
  correspondence_iou_threshold: 0.4

registration:
  confidence_keep_ratio: 0.5

reconstruction:
  mode: no_loop

loop:
  detection: {}
  constraint: {}
  optimizer: {}
```

规则：

- `reconstruction.mode` 只能是 `no_loop`、`traditional`、`corrected`；
- `window.size > window.overlap >= 1`；
- window size 和 overlap 不被任何 evaluation profile 锁死；
- `loop` 整段配置对 no-loop 可省略；
- no-loop 不验证、不加载回环 checkpoint；
- Traditional/Corrected 在模型加载前验证所需回环配置和权重；
- unknown、missing 和旧字段在模型加载前报错；
- 不提供旧字段兼容别名。

配置分层：

```text
configs/
├── reconstruction/
│   └── pi3_laser.yaml
├── evaluation/
│   ├── ate.yaml
│   └── pointcloud.yaml
└── experiments/
    ├── ate_matrix.yaml
    └── pointcloud_matrix.yaml
```

重建参数不重复写进 evaluation profile。点云 evaluation profile 只描述数据集、
裁剪、对齐、ICP 和指标定义。

## 13. 重建、评测与实验命令

重建单个配置：

```bash
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set segmentation.method=geometry \
  --set reconstruction.mode=corrected \
  --set window.size=20 \
  --set window.overlap=5
```

评测已有产物：

```bash
python evaluate_ate.py \
  --artifact outputs/reconstruction/<run-id> \
  --config configs/evaluation/ate.yaml
```

```bash
python evaluate_pointcloud.py \
  --artifact outputs/reconstruction/<run-id> \
  --config configs/evaluation/pointcloud.yaml
```

ATE matrix：

```yaml
evaluation: ate
matrix:
  segmentation.method: [depth, geometry, atomic]
  reconstruction.mode: [no_loop, traditional, corrected]
```

Point-cloud matrix：

```yaml
evaluation: pointcloud
matrix:
  segmentation.method: [depth, geometry, atomic]
  reconstruction.mode: [no_loop]
```

矩阵运行器必须在执行前展开并验证全部组合。点云矩阵包含任何回环模式时立即
报错。

```bash
python run_experiment_matrix.py \
  --config configs/experiments/ate_matrix.yaml \
  --set window.size=15 \
  --set window.overlap=5
```

```bash
python run_experiment_matrix.py \
  --config configs/experiments/pointcloud_matrix.yaml \
  --set window.size=20 \
  --set window.overlap=10
```

## 14. Artifact 存储

每个重建 run 写入独立目录：

```text
outputs/reconstruction/<run-id>/
├── manifest.json
├── resolved_reconstruction.yaml
├── trajectory.pt
├── pointmap.pt
├── confidence.pt
└── diagnostics.json
```

`manifest.json` 至少记录：

- artifact schema version；
- 数据集、序列和 frame IDs；
- segmentation method；
- reconstruction mode；
- window size 和 overlap；
- PI3 checkpoint SHA256；
- ordinary prediction key；
- resolved configuration SHA256；
- git commit；
- tensor 文件名、shape、dtype 和 digest。

ATE 只加载 trajectory。点云 evaluator 只加载 pointmap、confidence 和必要
manifest 字段。两者不加载模式专属 trace。

## 15. Evaluation 边界

### 15.1 ATE/RPE

Trajectory evaluator 保留现有 ATE、RPE translation 和 RPE rotation 的数学
定义、帧顺序和 GT 对齐语义。它不读取 segmentation、window trace、loop
constraints 或 point maps。

### 15.2 点云质量

Point-cloud evaluator 保留：

```text
predicted point map + GT point map
→ center crop and GT valid mask
→ Umeyama Sim(3)
→ point-to-point ICP
→ nearest neighbours and normals
→ Accuracy / Completion / Normal Consistency
→ Chamfer / Precision / Recall / F-score diagnostics
```

LASER Table 4 主统计仍为 Accuracy、Completion、Normal Consistency 的 mean 和
median。扩展指标不改变这六个主统计量的定义。

Point-cloud evaluator 在加载产物时强制
`reconstruction_mode == no_loop`。它不通过修改配置来关闭回环。

## 16. 模块与目录

目标结构：

```text
pipeline/
├── config.py
├── runner.py
├── manifest.py
└── artifacts.py

reconstruction/
├── prediction_stream.py
├── registry.py
├── shared.py
└── modes/
    ├── base.py
    ├── no_loop.py
    ├── traditional.py
    └── corrected.py

inference_engine/
├── prediction_cache/
├── segmentation/
└── anchor_propagation.py

loop_closure/
├── detection.py
├── evidence.py
└── methods/
    ├── traditional.py
    └── corrected.py

evaluation/
├── trajectory/
│   ├── evaluator.py
│   └── results.py
└── pointcloud/
    ├── evaluator.py
    ├── geometry_metrics.py
    ├── protocol.py
    └── results.py

experiments/
├── matrix.py
├── ate.py
└── pointcloud.py
```

依赖方向：

```text
pipeline → reconstruction → inference/loop components
experiments → pipeline + evaluation
evaluation → artifact views
```

禁止：

```text
evaluation → reconstruction
evaluation → segmentation
evaluation → anchor propagation
evaluation → loop closure
loop closure → evaluation
```

## 17. 删除与职责回迁

明确删除回环点云实验：

```text
mv_recon/loop_experiment.py
mv_recon/compare_loop_results.py
configs/evaluation/mv_recon_laser_nrgbd_depth_loop_traditional.yaml
tests/mv_recon/test_loop_experiment.py
tests/mv_recon/test_compare_loop_results.py
```

新路径通过回归后删除被取代内容：

```text
mv_recon/eval.py
mv_recon/paper_streaming.py
mv_recon/protocol.py
旧 point-cloud comparison 配置
旧 evaluator orchestration 测试
```

职责回迁：

- `paper_streaming.py` 的数学行为迁入 no-loop mode；
- Traditional/Corrected WindowEngine 状态机迁入对应 mode；
- 回环候选、约束转换和优化数学留在 `loop_closure`；
- point-cloud 纯指标、协议和结果写入迁入 `evaluation/pointcloud`；
- ATE/RPE 包装为 `evaluation/trajectory` 的纯 evaluator；
- `pipeline/runner.py` 只装配配置、provider、策略、mode 和 artifact writer。

## 18. 错误处理

模型加载前拒绝：

- unknown、missing 或 legacy 配置字段；
- 非法 segmentation method；
- 非法 reconstruction mode；
- `window.size <= window.overlap` 或 overlap 小于 1；
- point-cloud matrix 包含回环模式；
- Traditional/Corrected 缺少所需回环权重；
- no-loop 路径请求构造回环组件。

重建错误必须包含 mode、window index 和 frame range：

- prediction shape 或 frame count 不匹配；
- 非有限点、pose 或 confidence；
- adjacent overlap 没有有效置信度交集；
- Sim(3) 非有限或 scale 非正；
- segmentation labels shape 错误；
- anchor scale mask shape 错误或非有限；
- 最终 artifact 未严格覆盖每个输入帧一次。

artifact loader 拒绝：

- schema version 不匹配；
- manifest/tensor digest 不匹配；
- frame IDs 和 tensor 帧数不一致；
- tensor shape、dtype 或有限性不满足契约；
- point-cloud evaluator 收到非 no-loop artifact。

## 19. 测试策略

### 19.1 组件测试

- 三种 SegmentationStrategy；
- 原始 AnchorPropagator；
- adjacent registration；
- LoopDetector；
- LoopEvidenceProvider；
- Traditional/Corrected constraint 和 optimizer 数学。

### 19.2 模式时序测试

使用固定的 cached `WindowPrediction` 验证：

- no-loop 立即应用 adjacent Sim(3) 和 anchor scale；
- Traditional 在最终聚合前不应用记录的修正；
- Corrected 使用修正后的当前窗口作为下一窗口参考；
- no-loop 从不调用 LoopDetector；
- Traditional/Corrected 在窗口处理完成后各调用一次 LoopDetector；
- 每个窗口只调用一次选定 segmentation 和 AnchorPropagator；
- 每种模式严格去重 overlap 并覆盖所有 frame IDs。

### 19.3 数值回归

改造前用固定 ordinary predictions 生成回归 fixtures。改造后验证：

- no-loop Depth 与当前 Table 4 incremental implementation 一致；
- Traditional trajectory 和 point map 与原实现一致；
- Corrected trajectory 和 point map 与原实现一致；
- ATE、RPE translation、RPE rotation 在规定浮点容差内一致；
- Accuracy、Completion、Normal Consistency 在规定浮点容差内一致；
- Geometry/Atomic 只改变 segmentation 结果，不改变所选 mode 的时序。

如果纯 CPU/GPU 浮点执行不能逐位一致，测试使用记录并说明的绝对与相对容差；
不得通过扩大容差隐藏帧顺序、坐标系或算法时序变化。

### 19.4 矩阵测试

ATE 九种组合全部能够解析、装配并执行：

```text
depth / geometry / atomic
×
no_loop / traditional / corrected
```

Point-cloud 三种组合全部能够解析、装配并执行：

```text
depth / geometry / atomic
×
no_loop
```

显式验证 point-cloud matrix 拒绝 Traditional 和 Corrected。
ATE 与 point-cloud 同时请求相同 no-loop reconstruction 配置时，矩阵编排器复用同一
artifact，不重复执行重建。

### 19.5 window 参数测试

至少覆盖：

```text
window 10 / overlap 5
window 20 / overlap 5
window 20 / overlap 10
```

这些测试证明 evaluation 没有锁死 window 参数，不要求三组配置产生相同数值。

### 19.6 架构边界测试

使用 Python AST 检查：

- evaluation 不 import reconstruction、segmentation、anchor 或 loop closure；
- no-loop mode 不 import loop closure；
- ordinary prediction fingerprint 不包含 segmentation/mode/evaluation；
- ATE 只消费 trajectory view；
- point-cloud metrics 只消费 pointmap/confidence view；
- 删除的回环点云脚本和配置不再被正式入口引用。

## 20. 实施顺序

1. 为现有三个模式和点云指标建立固定 prediction 数值回归 fixtures。
2. 新增强类型 artifact、artifact writer/loader 和 evaluator views。
3. 提取公共 prediction stream，保持缓存和 worker 行为。
4. 先实现 no-loop mode，并对比现有 `paper_streaming.py`。
5. 迁移 Traditional 状态机，保持延迟修正和回环数学。
6. 迁移 Corrected 状态机，保持在线修正和 delta 数学。
7. 改造 pipeline config、registry 和 runner。
8. 迁移 ATE 和 point-cloud evaluator，使其只消费 artifact。
9. 实现并验证两个 experiment matrix runner。
10. 删除旧点云回环实验和被替代的 `mv_recon` 路径。
11. 运行组件、时序、数值、矩阵、参数和架构边界测试。
12. 更新 README 和云端验证文档。

每一步必须先写失败测试并观察预期失败，再写最小实现。

## 21. 云端交付

实现完成后，README 或独立验证文档必须提供可直接复制的命令，至少包括：

1. 克隆并检出最终分支；
2. 初始化 submodules；
3. 创建 Python 3.11 环境并安装依赖；
4. 编译 Cython 模块；
5. 下载或放置 PI3、SALAD 和 DINO 权重；
6. 配置数据集路径；
7. 运行 CPU/unit regression；
8. 运行单个 no-loop、Traditional、Corrected ATE 配置；
9. 运行完整 ATE 九组合矩阵；
10. 运行 point-cloud 三组合矩阵；
11. 修改 window size 和 overlap 的示例；
12. 检查 artifact、ATE 汇总和点云汇总的位置。

如果实现尚未推送到 GitHub，不能给出一个不可克隆的分支作为完成交付。最终云端
命令必须引用已经成功推送的实际分支和 commit。

## 22. 验收标准

- PI3 ordinary prediction 只有一个公共 provider 和 cache contract；
- 三种 segmentation 可与三个 ATE mode 任意组合；
- no-loop、Traditional、Corrected 保留各自算法时序；
- Traditional/Corrected 回环检测仍在窗口处理完成后独立执行；
- 三种模式使用同一份 LASER AnchorPropagator 实现；
- point-cloud evaluation 只接受三种 segmentation 的 no-loop artifact；
- window size 和 overlap 可由 reconstruction/experiment 配置自由设置；
- evaluator 不执行任何重建步骤；
- evaluator 内不存在第二份 reconstruction；
- 旧回环点云实验从分支中删除；
- 固定 prediction 的模式和指标数值回归通过；
- ATE 九组合和 point-cloud 三组合矩阵测试通过；
- 最终提供引用已推送分支的云端克隆与验证命令。
