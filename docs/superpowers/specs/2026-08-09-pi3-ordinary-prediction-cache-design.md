# LASER Pi3 普通窗口预测共享缓存设计

- 日期：2026-08-09
- 状态：待书面复核
- 目标分支：`codex/pi3-ordinary-prediction-cache`
- 精确基线：`origin/codex/modular-segmentation-loop-integration@3f8a3b8277f22ef1c6d820987cab1ff707b0741d`
- 参考实现：`origin/codex/pi3x-ordinary-prediction-cache@b930a9c933457199ea5d2e0b15ccc2182353a1fa`

## 1. 决策摘要

本分支以模块化 segmentation/loop 集成分支为唯一基线，只移植参考分支中与
ordinary sliding-window prediction cache 有关的能力。第一阶段只支持现有 Pi3，
不移植 Pi3X runtime、Pi3X 权重流程、multimodal 输入或 Surface-SfM。

共享边界固定在模型普通窗口预测层：

```text
canonical images
      |
      v
Pi3Adapter + LazyModelHandle
      |
      v
OrdinaryPredictionProvider
      |
      +-- miss --> Pi3 ordinary forward --> validated atomic store
      |
      +-- hit  ----------------------------> replay
      |
      v
depth + confidence + poses + reference intrinsic
      |
      v
reconstructed local XYZ + current in-memory RGB
      |
      +----------+-----------+
      |          |           |
    depth     geometry     atomic
      |          |           |
      +----------+-----------+
                 |
          anchor propagation
                 |
        +--------+--------+
        |                 |
  traditional        corrected
        |                 |
 method-specific WindowCache
        |
  uncached SALAD + uncached joint A/B Pi3 forward
```

核心约束是：共享模型事实，不共享分割派生状态、anchor 状态、回环方法状态或最终
优化结果。

## 2. 方案比较

评估了三种实施方式：

1. 整体合并参考分支，再删除 Pi3X。该方式会先引入大量无关模型 runtime 和
   许可证/权重处理，再依赖删除来恢复目标边界，回归风险最高。
2. 逐提交 cherry-pick 参考分支。参考分支的配置、adapter、Pi3X 和缓存提交彼此
   交织，选择性 cherry-pick 会产生大量语义冲突，难以证明没有遗漏隐式依赖。
3. 以参考实现为行为蓝本，在精确基线上做 Pi3-only 的测试驱动移植。先移植并
   精简行为测试，再逐层实现 WindowSpec、adapter/lazy handle、fingerprint、store、
   provider 和流水线接线。

采用方案 3。它保留已经验证过的缓存语义，同时让 diff 只包含本阶段需要的代码。

## 3. 目标

第一阶段必须实现：

1. 相同数据、Pi3 checkpoint、预处理和窗口调度下，depth、geometry、atomic 共享
   同一组 ordinary predictions。
2. traditional 与 corrected 共享 ordinary predictions，但继续生成并校验各自的
   `WindowCache`。
3. 缓存支持 `auto`、`refresh`、`readonly`、`off` 四种模式。
4. 冷缓存允许逐窗口生成并在中断后恢复；热缓存不构造 Pi3、不把普通窗口图像提前
   移到 CUDA。
5. 缓存内容损坏、输入改变或模型/runtime 改变时，不能静默读取旧制品。
6. ordinary forward 与 joint A/B forward 分开统计；热 ordinary cache 不意味着
   有回环候选时整个进程不构造模型。
7. cache-off、cold-cache 和 warm-cache 三条路径保持原基线数学结果。
8. 现有 legacy evaluation/streaming 入口在需要时通过同一准备接口接线，不改变
   ATE、RPE、可视化或数据顺序语义。

## 4. 明确不做

本阶段不包含：

- Pi3X、Pi3X layer runtime、Pi3X checkpoint 下载或 multimodal 输入；
- Surface-SfM 或其他新优化后端；
- SALAD descriptor、loop candidate 或 joint A/B prediction 缓存；
- cached-to-joint alignment、loop constraint、优化结果或最终点云缓存；
- 修改 depth、geometry、atomic segmentation 算法；
- 修改 anchor propagation 公式、顺序或失败行为；
- 修改 traditional/corrected Sim(3)、优化器或聚合语义；
- 跨实验共享 labels、diagnostics、temporal graph 或 anchor scale mask；
- 旧 prediction cache 或旧 `WindowCache` schema 的兼容读取；
- tensor 二次量化、自动淘汰或后台清理；
- 在模型、dtype、设备或窗口配置错误时静默回退。

## 5. 配置契约

配置增加显式 Pi3 标识和独立 prediction cache：

```yaml
model:
  name: pi3
  checkpoint: weights/model.safetensors
  inference_device: cuda
  process_device: cpu
  dtype: bfloat16

prediction_cache:
  root: inference_cache/predictions
  mode: auto
```

`ModelName` 第一阶段只有 `PI3 = "pi3"`。保留 model name 是为了指纹、诊断和
方法缓存 provenance 明确，不代表本阶段支持 Pi3X。

```python
class PredictionCacheMode(str, Enum):
    AUTO = "auto"
    REFRESH = "refresh"
    READONLY = "readonly"
    OFF = "off"
```

`output.cache_dir` 继续保存方法专属 `WindowCache`；`prediction_cache.root` 只保存
普通模型预测。矩阵脚本不得按 segmentation 或 loop method 改写 prediction root。

## 6. 显式窗口调度

新增不可变窗口类型：

```python
@dataclass(frozen=True)
class WindowSpec:
    index: int
    frame_start: int
    frame_end: int

    @property
    def frame_count(self) -> int: ...
```

`build_window_specs(image_count, window_size, overlap)` 是窗口 tensor slice、缓存键、
方法缓存绝对帧范围和诊断的唯一调度来源。`validate_window_specs()` 拒绝索引不连续、
范围越界、窗口长度不符或重叠不符的序列。下游不再用
`cache_id * (window_size - overlap)` 反推范围。

## 7. Pi3 adapter 与惰性模型

新增 `inference_engine/models/`：

```text
__init__.py
adapters.py
lazy.py
loader.py
```

`Pi3Adapter` 接收 `(N,3,H,W)` 或 `(B,N,3,H,W)`，统一输出：

```text
local_points: (B,N,H,W,3)
camera_poses: (B,N,4,4)
conf:         (B,N,H,W)
images:       (B,N,3,H,W)
```

adapter 验证必需 key、rank、batch/帧/空间尺寸和 finite 值，只返回四个标准字段。
Pi3 checkpoint 继续 strict load；不改动 `pi3/models/` 的数学实现。

`LazyModelHandle` 持有 adapter factory，并提供：

```python
predict(images, *, kind=ModelForwardKind.ORDINARY | ModelForwardKind.JOINT)
```

同一进程最多构造一次模型，并分别累计 `ordinary_forward_count`、
`joint_forward_count` 和 `model_constructed`。ordinary 全命中且没有 loop candidate
时 factory 调用次数必须为零；有 loop candidate 时 joint 路径可触发模型构造。

## 8. Prediction fingerprint

指纹以规范 JSON 序列化后做 SHA-256。参与键的字段固定为：

- prediction schema 与 adapter contract 版本；
- `model.name`；
- checkpoint 文件内容 SHA-256；
- 实际参与 Pi3 构造/forward 的 runtime source digest；
- 模型 dtype；
- 按顺序排列的逐图像内容 digest；
- 预处理实现/版本、模式和输出 shape；
- sample stride；
- window size、overlap 和完整有序 `WindowSpec` 序列。

以下字段明确不参与键：

- segmentation 方法及参数、atomic split mode；
- anchor propagation 参数；
- loop on/off、traditional/corrected、SALAD 和 optimizer 参数；
- joint constraint chunk size；
- inference/process device；
- scene name、输出路径、Git commit 和硬件信息。

Git/Python/PyTorch/CUDA/GPU 等环境信息只写 provenance，不决定是否命中。

## 9. 缓存制品与存储

每个 ordinary window 只保存：

```text
depth:        (N,H,W)
confidence:   (N,H,W)
camera_poses: (N,4,4)
```

序列额外保存第一窗口得到的 `reference_intrinsic: (3,3)`。不保存完整 XYZ、RGB、
global points 或方法派生状态。tensor 按模型实际输出 dtype 存储，不二次量化。

缓存 schema 使用 `v2` 布局并记录 canonical tensor digest：

```text
<prediction_cache.root>/v2/<prediction-key>/
  manifest.json
  sequence.json
  complete.json
  windows/000000.pt
  windows/000001.pt
  invalid/
  locks/entry.lock
```

写入使用同目录临时文件、flush/fsync、重新读取验证和原子 replace。读取验证 schema、
prediction key、WindowSpec、shape、dtype、finite 值与 payload digest。

模式语义：

- `auto`：命中则读取；miss 则计算；损坏条目隔离到 `invalid/` 后重算；
- `refresh`：忽略已有普通窗口并重算，以原子替换发布；
- `readonly`：任何 miss、损坏或元数据不一致立即失败；
- `off`：不读写持久制品，但仍走 provider 的同一规范化/反投影路径。

已完成窗口在中断后保留。整个 sequence 使用 entry lock 防止两个实验同时发布相同
prediction key 的不完整状态；第一阶段不自动删除旧键或 `invalid/`。

## 10. OrdinaryPredictionProvider

provider 是缓存唯一入口：

```python
def get(spec: WindowSpec, images: torch.Tensor) -> ModelPrediction: ...
```

执行流：

1. 读取并验证 sequence metadata 与窗口 artifact；
2. miss 时通过 `LazyModelHandle.predict(..., kind=ORDINARY)` 调用 Pi3；
3. 第一窗口使用基线已有函数估计 reference intrinsic；
4. 从 local points 提取 depth，与 confidence/poses 一起原子写入；
5. hit 和 miss 都用 `depth + reference_intrinsic` 重建 local XYZ；
6. 附加当前内存中的 images，不缓存 RGB；
7. 返回独立 tensor，避免 traditional/corrected 的原地变换污染 store 制品。

该前移保持原基线数学语义：基线 worker 本来就以第一窗口内参把 depth 反投影成
后续使用的 local XYZ。provider 接管后，两个 worker 必须删除重复的内参估计和
反投影，避免执行两次。

## 11. StreamingWindowEngine 与 runner

队列元素从裸 tensor 变为：

```python
@dataclass(frozen=True)
class WindowRequest:
    spec: WindowSpec
    images: torch.Tensor
```

推理线程调用 `delegate.get(request.spec, request.images)`；registration 线程继续
执行原有 confidence intersection、相邻注册、segmentation、temporal graph、anchor
传播和方法专属状态更新。

runner 先加载 CPU images 并生成 specs，再构建 fingerprint、store、lazy handle、
provider 和 engine。`run_windows()` 按 `WindowSpec` slice，且不在 runner 中提前
`.to(cuda)`；只有 ordinary miss 或 uncached joint forward 才把输入移到模型设备。

## 12. 方法专属 WindowCache

`WindowCache` schema 升级，并增加：

```text
prediction_key
model_name
checkpoint_digest
```

traditional/corrected 继续保留各自的 `loop_method` tag、分割状态、anchor mask 和
回环状态。`from_payload(..., expected_method=...)` 继续拒绝跨方法读取，同时验证
prediction provenance；不提供旧 schema 兼容读取。

两个 worker 只接收显式 `WindowSpec`、消费 provider 已恢复的 local points 并保存
provenance。不得修改 `compute_sim3_ab`、`build_local_loop_constraint`、相邻注册、
优化器或聚合公式。

## 13. Joint loop inference

ordinary cache 不替代 loop candidate 两侧 neighborhood 的 joint Pi3 forward。
`JointPi3AlignmentEstimator` 可重命名为通用 `JointAlignmentEstimator`，但仍使用
同一个 `LazyModelHandle` 并以 `kind=JOINT` 调用模型。

候选局部几何/验证 `ValueError` 继续只跳过当前候选。模型异常、CUDA OOM 和未知
`RuntimeError` 必须传播。traditional/corrected 继续调用各自现有 constraint 转换。

## 14. 诊断

`run_summary.json` 增加：

```text
model_name
checkpoint_digest
ordinary_prediction_key
prediction_cache_mode
model_constructed
ordinary_hits
ordinary_misses
ordinary_forward_count
joint_forward_count
corrupt_count
prediction_cache_read_ms
prediction_cache_write_ms
saved_window_count
stored_bytes
```

readonly miss 必须报告窗口 index 和绝对帧范围；损坏重算必须报告原路径、隔离路径
和原因。`model_constructed=true` 不能单独解释为 ordinary cache miss。

## 15. 错误处理

启动或运行时明确拒绝：

- 非 `pi3` model name 或无效 cache mode；
- 不存在/不可读的 checkpoint、图像或 cache root；
- 不支持的 dtype/device，或请求 CUDA 但 CUDA 不可用；
- adapter 非 mapping 输出、缺 key、shape/rank 不符或 NaN/Inf；
- WindowSpec 与 artifact frame range/shape 不符；
- readonly miss、损坏、schema/digest/key 不符；
- sequence reference intrinsic 缺失或不合法。

错误不触发模型、dtype、设备、窗口大小或缓存模式的 silent fallback。

## 16. 测试策略

实现严格遵循 red-green-refactor。优先从参考分支移植行为测试，删除 Pi3X 专属断言，
每组测试先在目标基线上因功能缺失而失败，再写最小实现。

必须覆盖：

1. `WindowSpec` 构建/校验和非整除尾窗口；
2. Pi3 adapter 4D/5D 输入、标准输出、shape/key/finite 拒绝；
3. lazy factory 只构造一次以及 ordinary/joint 独立计数；
4. 指纹的包含项与排除项；
5. 四种 store mode、atomic write、digest/schema 校验、损坏隔离和部分恢复；
6. provider miss/hit、reference intrinsic 恢复、RGB 不落盘和返回值隔离；
7. 三种 segmentation、两种 loop method 命中同一 prediction key；
8. 方法 `WindowCache` 继续跨方法拒绝并验证 provenance；
9. ordinary 全 hit、存在 joint candidate 时 ordinary forward 为零且 joint forward
   大于零；
10. cache-off、cold-auto、warm-readonly 的 ordinary tensor 与方法结果回归；
11. legacy evaluation/streaming 入口和矩阵脚本接线；
12. 完整 pytest suite。

## 17. 可量化验收

设规范窗口数为 `W`：

```text
第一次 cold auto/refresh： ordinary_forward_count = W
同键 warm readonly：       ordinary_forward_count = 0
后续 segmentation/loop：   ordinary_forward_count = 0
```

同一 prediction key 下：

- depth、geometry、atomic 使用相同 ordinary artifact digest；
- traditional、corrected 使用不同方法缓存且可产生不同最终轨迹；
- cold/warm provider 输出在严格容差内一致；
- cache-off 与 cache-on 不改变 ordinary 数学结果；
- joint forward 次数由每次实验的有效候选决定，不纳入 ordinary cache 命中率。

CPU 自动测试必须全部通过。具备 Pi3 checkpoint 与 CUDA 的环境再执行 15 帧两窗口
cold/warm smoke、三种 segmentation replay、两种 loop replay 和完整实验矩阵。

## 18. 文件边界

预计新增：

```text
inference_engine/models/__init__.py
inference_engine/models/adapters.py
inference_engine/models/lazy.py
inference_engine/models/loader.py
inference_engine/prediction_cache/__init__.py
inference_engine/prediction_cache/types.py
inference_engine/prediction_cache/fingerprint.py
inference_engine/prediction_cache/store.py
inference_engine/prediction_cache/provider.py
tests/test_model_adapters.py
tests/test_model_loader.py
tests/test_prediction_types.py
tests/test_prediction_fingerprint.py
tests/test_prediction_store.py
tests/test_prediction_provider.py
tests/test_prediction_cache_matrix.py
```

预计修改：

```text
pipeline/config.py
pipeline/preflight.py
pipeline/runner.py
pipeline/diagnostics.py
configs/pipeline/default.yaml
configs/pipeline/test.yaml
inference_engine/streaming_window_engine.py
loop_closure/constraint_estimation.py
loop_closure/methods/base.py
loop_closure/methods/traditional.py
loop_closure/methods/corrected.py
scripts/verify_pipeline_matrix.py
eval_launch.py
utils/interfaces.py
相关既有测试与用户文档
```

缓存功能不得修改以下算法实现：

```text
pi3/models/pi3.py
inference_engine/segmentation/depth.py
inference_engine/segmentation/geometry.py
inference_engine/segmentation/atomic.py
inference_engine/anchor_propagation.py
loop_closure/methods/shared.py
loop_closure/utils/sim3loop.py
```

如果接口接线被迫触及该列表，必须先增加数值回归测试，并在实现说明中单独解释。

## 19. 基线与实施约束

新分支必须保持以远端 `3f8a3b8` 为祖先，不使用已经被本地快进的同名 baseline ref。
参考分支相对精确基线 ahead 16 commits，但不会整体 merge 或 cherry-pick。

当前隔离工作区已执行：

```text
python setup.py build_ext --inplace
python -m pytest -q
```

基线结果为 `160 passed, 1 warning`。Cython 生成的 `.cpp`、`.so` 和 `build/` 制品不
进入提交。当前 Python 3.13 环境无法从包索引安装 `open3d`，但基线测试不依赖该
缺失包并已完整通过；GPU/真实数据验收留给具备项目运行环境的机器执行。
