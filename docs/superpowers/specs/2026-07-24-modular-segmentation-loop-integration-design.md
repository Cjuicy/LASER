# LASER 模块化分割与回环整合设计

- 日期：2026-07-24
- 状态：已确认
- 仓库：`Cjuicy/LASER`
- 实现基线：`codex/loop-closure-corrected-pipeline`
- atomic split 来源：`codex/auto-post-merge-split`
- traditional 回环来源：`codex/loop-closure-cloud-updated-20260714`

## 1. 目标

将三个现有开发分支整合为一套结构清晰、参数统一、可独立测试的 LASER
实验流水线。最终系统包含：

- 三种公开分割方法：`depth`、`geometry`、`atomic`；
- 一种保持现有算法不变的锚点传播方法；
- 两种公开回环方法：`traditional`、`corrected`；
- 一套严格、唯一的 YAML 配置；
- 一个统一实验入口和一套六组合成验证工具。

公开的 `atomic` 代表最新的 layer-atomic merge 加 guarded post-merge
split。无 split 的旧 atomic 只作为 `atomic.split_enabled: false` 消融，
不注册为第四种分割方法。

默认选择：

- `segmentation.method: atomic`；
- `atomic.split_enabled: true`；
- `loop.method: corrected`。

## 2. 非目标

本次整合不包含：

- 新的锚点传播算法；
- `auto-post-merge-split` 分支中尚未实现的 HART-AP 设计；
- 新的 SALAD 描述子、候选排序或候选接受算法；
- 对现有 `Sim3LoopOptimizer` 数学形式的重写；
- geometry 历史 profile 或旧 CLI 参数兼容；
- silent fallback 到其他分割或回环方法；
- 为旧缓存格式提供长期兼容读取器。

## 3. 固定来源与整合边界

| 最终模块 | 固定来源 |
|---|---|
| `depth` | `codex/loop-closure-corrected-pipeline` |
| `geometry` | `codex/loop-closure-corrected-pipeline` |
| `atomic` merge | `codex/loop-closure-corrected-pipeline` |
| atomic guarded split | `codex/auto-post-merge-split` |
| 锚点传播 | corrected 分支现有实现 |
| `traditional` 回环 | `codex/loop-closure-cloud-updated-20260714` |
| `corrected` 回环 | `codex/loop-closure-corrected-pipeline` |

实现以 corrected 分支为底座。atomic 分支只移植 split 实现、保护、诊断和
测试；traditional 分支只移植论文基线回环的算法语义。公共图像清单、严格配置、
置信度选择和诊断使用新的共享实现。

traditional 保留：

- 延迟窗口矫正；
- 传统缓存状态；
- 传统回环约束构建；
- 传统优化结果应用与聚合顺序。

corrected 保留：

- 当前窗口立即前向矫正；
- `sim3_abs` 和 `sim3_edge`；
- 全局对齐结果到窗口局部约束的坐标转换；
- 优化前后增量只应用一次；
- 非有限或非正尺度 Sim(3) 的拒绝与诊断。

两者共享同一个 SALAD 候选列表和同一套公共参数，但不共享方法专属的约束及
聚合实现。

## 4. 总体架构

采用统一编排器与策略模块：

```text
run_laser.py
  -> strict configuration
  -> canonical image manifest
  -> window scheduling and Pi3 inference
  -> SegmentationStrategy
       depth | geometry | atomic
  -> unchanged AnchorPropagator
  -> LoopClosureStrategy
       traditional | corrected
  -> aggregation, output, diagnostics
```

模块边界：

```text
pipeline/
  config.py
  runner.py
  manifest.py

inference_engine/segmentation/
  base.py
  depth.py
  geometry.py
  atomic.py

inference_engine/
  anchor_propagation.py

loop_closure/methods/
  base.py
  traditional.py
  corrected.py
  shared.py
```

职责如下：

- `pipeline/config.py`：解析、验证和冻结配置；拒绝未知字段及旧字段；
- `pipeline/manifest.py`：唯一的图像发现、自然排序和采样实现；
- `pipeline/runner.py`：模型加载、窗口调度、策略装配和结果保存；
- `inference_engine/segmentation/*`：只产生逐帧整数 labels；
- `anchor_propagation.py`：包装现有尺度锚点和传播实现，不改变公式；
- `loop_closure/methods/shared.py`：SALAD、置信度 mask 和 Sim(3) 基础工具；
- 两个回环策略：各自拥有窗口状态、约束、优化应用和聚合。

## 5. 唯一配置

唯一规范配置为：

```yaml
version: 1

input:
  image_dir: data/00/image_2
  sample_stride: 1

output:
  scene_name: kitti_00
  cache_dir: inference_cache/kitti_00
  result_dir: viser_results
  save_diagnostics: true

model:
  checkpoint: weights/model.safetensors
  inference_device: cuda
  process_device: cpu
  dtype: bfloat16

window:
  size: 10
  overlap: 5

segmentation:
  method: atomic
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
    split_enabled: true
    split_score_threshold: 0.10
    auxiliary_confirmation: true

anchor_propagation:
  enabled: true
  correspondence_iou_threshold: 0.4

loop:
  enabled: true
  method: corrected

  registration:
    confidence_keep_ratio: 0.30

  detection:
    method: salad
    salad_checkpoint: weights/dino_salad.ckpt
    dino_checkpoint: weights/dinov2_vitb14_pretrain.pth
    image_size: [336, 336]
    batch_size: 32
    similarity_threshold: 0.7
    top_k: 5
    nms_enabled: true
    nms_frame_radius: 25

  constraint:
    chunk_size: 20

  optimizer:
    implementation: cpp
    use_sim3: true
    max_iterations: 30
    initial_damping: 1.0e-6
```

字段规则：

- 所有 confidence 字段都使用 `keep_ratio` 正向语义；
- `0.30` 永远表示保留有限置信度像素中最高约 30%；
- geometry 不再提供 `profile`，Felzenszwalb 参数直接显式配置；
- unknown、missing、legacy 字段在模型加载前报错；
- `window.size > window.overlap >= 1`；
- 所有 keep ratio 满足 `0 < ratio <= 1`；
- 非 depth 分割不再依赖旧的 `depth_refine` 开关；
- `anchor_propagation.enabled` 是是否应用统一尺度传播的唯一开关。

统一入口只接受：

```bash
python run_laser.py --config configs/pipeline/default.yaml
```

实验覆盖使用同一字段路径：

```bash
python run_laser.py \
  --config configs/pipeline/default.yaml \
  --set segmentation.method=geometry \
  --set loop.method=traditional
```

旧入口中的方法参数和旧字段不保留兼容别名。启动时保存
`resolved_config.yaml`，并打印最终方法选择、关键参数和配置摘要哈希。

## 6. 分割策略

公共返回类型：

```python
@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    diagnostics: Mapping[str, float | int | bool]
```

公共策略接口：

```python
class SegmentationStrategy(Protocol):
    name: Literal["depth", "geometry", "atomic"]

    def segment(
        self,
        point_maps: np.ndarray,
        confidence: np.ndarray | None,
        images: np.ndarray | None,
    ) -> list[SegmentationResult]: ...
```

三种策略只负责 labels 和方法诊断。`match_segmentation_seq`、graph 构建和
锚点传播在策略外只执行一次。

### 6.1 depth

- Felzenszwalb 输入为标量 depth；
- 使用显式公共 Felzenszwalb 参数；
- 使用 `depth_merge_threshold`；
- confidence 选择使用正向 `confidence_keep_ratio`。

### 6.2 geometry

- Felzenszwalb 输入为归一化 depth 加三个 normal 通道；
- 保留 `cross` 和 `sobel`；
- 保留固定来源的区域 descriptor 与合并规则；
- 使用显式 `normal_threshold_degrees`；
- 不再保留 `legacy`/`baseline_params` profile 选择。

### 6.3 atomic

执行顺序：

```text
depth Felzenszwalb atoms
-> coarse depth layers
-> coarse-guided atomic geometry merge
-> optional guarded post-merge split
-> compact labels
```

split 开启时保留固定来源的保护：

- 30 度法向屏障；
- 2 至 4 个有效 marker；
- 子区面积至少为 `max(min_size, ceil(0.02 * parent_area))`；
- 单次 watershed，不递归，最多四叶；
- 候选必须完整覆盖父区域；
- 统一接受分数 `normal_gain * auxiliary_confirmation`；
- 默认接受阈值 `0.10`；
- RGB 或归一化三维 gap 任一可确认法向候选；
- marker、面积、覆盖、有限值或得分检查失败时保留父区域。

`split_enabled: false` 保留 merge 后 labels，用于同一 atomic 方法的消融。

## 7. 锚点传播

公共接口：

```python
class AnchorPropagator:
    def propagate(
        self,
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap: int,
    ) -> torch.Tensor: ...
```

实现继续调用现有：

- overlap 区域分割对应；
- IoU 阈值过滤；
- 区域尺度估计；
- IoU 加权的空间与时间传播；
- scale mask 生成。

除参数对象和模块包装外，不改变公式、传播顺序、默认
`correspondence_iou_threshold=0.4` 或失败行为。

## 8. 回环公共契约

SALAD 产生：

```python
@dataclass(frozen=True)
class LoopCandidate:
    frame_a: int
    frame_b: int
    similarity: float
```

回环策略接口：

```python
class LoopClosureStrategy(Protocol):
    name: Literal["traditional", "corrected"]

    def prepare_window(...) -> WindowCache: ...
    def build_constraints(...) -> list[LoopConstraint]: ...
    def optimize(...) -> LoopSolution: ...
    def aggregate(...) -> ReconstructionResult: ...
```

共享模块负责：

- 规范图像清单；
- SALAD descriptor、相似度、top-k 和 NMS；
- `select_top_confidence_mask(confidence, keep_ratio)`；
- Sim(3) 校验、组合、求逆和诊断指标；
- optimizer 创建。

策略模块负责：

- 前向窗口矫正时机；
- 方法专属缓存状态；
- 候选到回环约束的转换；
- 优化结果应用；
- 聚合。

## 9. 缓存契约

所有缓存使用带版本的公共外壳：

```text
schema_version
loop_method
window_index
frame_start
frame_end
local_points
camera_poses
confidence
segmentation_labels
anchor_scale_mask
loop_state
```

`loop_state` 是按 `loop_method` 区分的 tagged 数据。

traditional state 保存传统相对 Sim(3) 和延迟应用状态。corrected state 保存：

- `sim3_abs`；
- 首窗口之外的 `sim3_edge`；
- anchor scale 已经应用的状态。

加载时验证：

- schema 版本；
- loop 方法；
- 窗口编号与帧范围；
- 方法必需字段；
- 缓存数量与规范图像清单推导的窗口数量。

traditional 和 corrected 缓存不可交叉读取。旧缓存需要重新生成，不提供隐式
迁移。

## 10. 数据流

```text
strict config validation
-> canonical natural-sorted image manifest
-> one-time sampling
-> sliding windows
-> Pi3 inference
-> selected segmentation strategy
-> common graph construction
-> unchanged anchor propagation
-> selected loop strategy window state
-> shared SALAD candidates
-> method-specific constraints and optimization application
-> method-specific aggregation
-> common output and diagnostics
```

同一次运行中，流式推理、SALAD 和联合 Pi3 使用同一个图像清单对象。任何下游
模块不得重新扫描、排序或采样。

## 11. 错误处理

模型加载前拒绝：

- 配置版本错误、unknown、missing 或 legacy 字段；
- 不存在的输入目录、配置或权重；
- 空图片清单；
- 非法窗口、比例、枚举或数值范围；
- CUDA 配置与实际环境不匹配；
- atomic split 参数不合法。

运行时规则：

- 分割异常不得 silent fallback 到 depth；
- atomic 单个 split 候选拒绝时保留父区域并记录原因；
- 无 SALAD 候选属于合法结果；
- 数学无效的单条回环约束记录后跳过；
- 全部约束无效时返回前向轨迹；
- 缓存 schema、帧索引或数量不一致时立即停止；
- 两种回环的算法差异不得通过共享 helper 被隐式抹平。

## 12. 诊断与实验产物

每次运行保存：

```text
resolved_config.yaml
run_summary.json
loop_candidates.json
loop_constraints.json
segmentation_diagnostics.json
trajectory and reconstruction artifacts
```

`run_summary.json` 至少包含：

- 配置摘要哈希；
- Git 提交；
- segmentation 和 loop 方法；
- 图像数量与首尾文件；
- 窗口数量；
- 候选和有效约束数量；
- split parent/proposed/accepted/rejection 计数；
- 各阶段运行时间；
- 是否走无回环路径。

## 13. 测试设计

### 13.1 算法单元测试

- 三种分割分别直接测试；
- labels 全覆盖、紧凑、确定且不修改输入；
- atomic 覆盖正常 split 和全部保护拒绝原因；
- split 关闭时输出与 merge-only 固定结果一致。

### 13.2 公共契约测试

- 三种分割满足同一个策略契约；
- 两种回环满足同一个策略契约；
- 两种回环接收同一图像清单、候选列表和最高 30% mask；
- 两种缓存互相拒绝；
- 锚点传播固定输入的输出与整合前逐元素一致；
- strict config 拒绝旧参数、未知字段和非法组合。

### 13.3 六组合成测试

使用 mock Pi3 和 mock SALAD 参数化运行：

```text
depth    x traditional
depth    x corrected
geometry x traditional
geometry x corrected
atomic   x traditional
atomic   x corrected
```

每组验证：

- 图像清单、窗口数和帧顺序；
- 分割与回环策略调用；
- 唯一锚点传播实现；
- 有回环与无回环路径；
- 方法专属缓存和约束；
- 解析后配置、哈希和诊断输出。

### 13.4 真实序列验证

在具备本地权重、GPU 和数据的环境，对相同输入运行六组实验。传统与 corrected
结果允许不同，但必须共享相同的帧清单和 SALAD 候选。同一配置重复运行必须
确定。

验证入口：

```bash
python setup.py build_ext --inplace
pytest -q
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/test.yaml
```

## 14. 验收标准

完成必须同时满足：

1. 公开分割枚举恰好为 `depth`、`geometry`、`atomic`；
2. 公开回环枚举恰好为 `traditional`、`corrected`；
3. atomic 默认启用 guarded split；
4. 锚点传播固定回归输出不变；
5. 两种回环共享候选与公共参数但保留独立算法语义；
6. 旧参数和未知字段被严格拒绝；
7. 六组合成测试全部通过；
8. 现有相关算法测试迁移到新接口后通过；
9. 真实序列运行命令、配置和输出诊断完整；
10. 实现代码不包含 HART-AP 新传播逻辑。
