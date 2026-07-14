# LASER 三种分割方法统一设计

**日期：** 2026-07-14

**目标分支：** `codex/unified-segmentation-methods`

**分支基线：** `feature/layer-atomic-geometry` 的 `7dca624`

## 目标

在同一份 LASER 代码中运行三种已经验证过的分割方法，使后续实验能够在完全相同的模型、置信度筛选、滑动窗口、segment graph、尺度估计、尺度传播、缓存和回环流程下进行公平对比：

- `depth`：原始 LASER 深度分割；
- `geometry`：`Cjuicy/LASER-Geometry` 中的 geometry-aware 方法；
- `layer_atomic`：`Cjuicy/LASER` 的 `feature/layer-atomic-geometry` 分支中的新方法。

本次改动只统一方法选择与运行入口，不增加对比指标或实验报表。

## 固定来源版本

实现固定使用以下已经验证过的源码版本：

- LASER depth 基线：`7dca624` 所继承的基线实现；
- LASER-Geometry：`Cjuicy/LASER-Geometry/main` 的 `340c599`；
- layer-atomic geometry：`Cjuicy/LASER` 的 `7dca624`。

三种分割算法仍然可以被独立调用。不得为了统一路由而改写它们的公式、阈值、区域合并判断或各自已经存在的函数签名。

## 范围

### 本次包含

- 增加 `depth`、`geometry` 和 `layer_atomic` 三种分割模式。
- 保持当前 depth 和 layer-atomic 实现不变。
- 移植 LASER-Geometry 的分割核心及其必需的 normal/geometry 辅助函数。
- 普通流式推理与回环流式推理共用一个轻量的 graph 构建路由。
- 在 `demo.py` 和 `demo_lc.py` 中暴露完全相同的模式选择。
- 记录并说明实际生效的参数与运行命令。
- 增加特征保持测试、路由测试、CLI 测试、Engine 测试和 smoke test。

### 本次不包含

- 新的重建、位姿、深度或分割指标。
- 新的实验面板、播放器或可视化报告。
- LASER-Geometry 中的 alignment debug 与置信度加权尺度锚点实验。
- 对 segment matching、scale anchor estimation、scale propagation、Sim(3)、缓存聚合或回环算法的修改。
- 阈值调优或对任何分割方法作性能提升结论。

## 总体架构

三种模式都输出逐帧整数 labels。只有 labels 的生成方法不同，所有下游操作完全共用。

```text
point maps + confidence
          |
          v
轻量 segmentation-mode 路由
  |             |               |
  v             v               v
depth        geometry       layer_atomic
labels        labels           labels
  |             |               |
  +-------------+---------------+
                |
                v
     match_segmentation_seq
                |
                v
      共用尺度锚点与尺度传播
                |
                v
   共用 streaming/cache/loop-closure
```

统一分支以 layer-atomic 分支为基础，因为该分支已经包含验证过的新算法和尺度不变性修复。LASER-Geometry 仓库与当前 LASER 仓库没有可以直接合并的共同 Git 历史，因此只移植固定提交中的分割核心，不整体合并其仓库历史。

## 三种分割方法

### `depth`

路由从 `point_map[..., -1]` 提取 depth，并调用现有的 `segment_depth_felzenszwalb_rag(...)`。任何 geometry 专属参数都不能进入该路径。

### `geometry`

路由调用固定版本的 LASER-Geometry 实现，并保持以下行为不变：

- Felzenszwalb 的四通道输入由归一化 depth 和三个 normal 通道组成；
- 保留 `cross` 与 `sobel` 两种 normal 估计方法；
- 保留原有 region descriptor；
- 保留原有深度、法向夹角、置信度和 union-find 合并规则；
- 保留 `legacy` 与 `baseline_params` 两种参数档位。

LASER-Geometry 的原始底层函数继续保留其源码默认值，以保证源码级兼容。统一 Engine 和 CLI 默认选择 `baseline_params`，使正式对比时 geometry 与 depth、layer-atomic 使用相同的 Felzenszwalb 参数。`legacy` 仍可用于复现历史 geometry 实验。

### `layer_atomic`

现有 `segment_point_map_layer_atomic(...)` 和 `merge_layer_atoms(...)` 保持不变。该模式继续：

- 复用 depth 分割产生的初始 atoms 和 coarse layers；
- 计算局部 atom 尺度与三维边界间隙；
- 使用弱 coarse-layer 先验；
- 保持全像素覆盖、确定性的紧凑 labels、无效边界处理和全局尺度不变性。

## Felzenszwalb 参数契约

三种模式的正式对比统一使用以下参数：

| 参数 | 数值 |
| --- | ---: |
| `scale` | 300 |
| `sigma` | 1.1 |
| `min_size` | 500 |

depth、layer-atomic 和 geometry 的 `baseline_params` 档位已经提供这组参数。路由必须原样传递，不做任何换算。geometry 的 `legacy` 档位继续保留历史参数 `200 / 1.0 / 300`，并且绝不会被隐式选择。

参数相同并不意味着算法相同：depth 与 layer-atomic 的 Felzenszwalb 输入是标量 depth，而 geometry 的输入是归一化 depth 与 normals。

## 接口与路由

三种方法各自的入口保持不变：

```python
segment_depth_felzenszwalb_rag(depth_map, ...)
segment_geometry_felzenszwalb_rag(depth_map, point_map=..., ...)
segment_point_map_layer_atomic(point_map, ...)
```

`make_sp_graph(...)` 作为统一集成边界。它接收 point maps 和 `segment_mode`，选择对应的底层入口，然后只调用一次现有 `match_segmentation_seq(...)`，把得到的 labels 构建成统一 graph。

路由层只负责：

- 校验模式；
- 在需要时提取标量 depth；
- 原样转发 confidence 和公共 Felzenszwalb 参数；
- 只向 geometry 转发 normal/profile 参数；
- 调用现有 batch 图像处理包装器；
- 使用返回的 labels 构建公共时序 segment graph。

路由层不得包含任何分割公式或区域合并逻辑。

## Engine 与 CLI 配置

`StreamingWindowEngine` 保存：

- `segment_mode`，默认 `depth`；
- `normal_method`，默认 `cross`；
- `geometry_seg_profile`，默认 `baseline_params`；
- 固定的 Felzenszwalb 三参数。

`StreamingWindowEngineLC` 将同一配置转发给父类并使用同一个 graph 路由，不能维护另一套独立的模式判断。

两个 demo 都暴露：

```text
--segment_mode depth|geometry|layer_atomic
--normal_method cross|sobel
--geometry_seg_profile baseline_params|legacy
```

原有命令不传 `--segment_mode` 时继续选择原始 depth 基线。非 depth 模式必须同时开启 `--depth_refine`；否则在启动时直接报错，避免用户以为某种方法已生效但实际上没有运行。

启动时打印实际生效的 segmentation mode、geometry 专属选项以及 Felzenszwalb 三参数。这样后续指标实验能够留下明确配置记录，同时本次不引入指标系统。

## 错误处理

- 推理开始前拒绝未知 `segment_mode`。
- geometry 模式下拒绝未知 profile 或 normal method。
- 未开启 depth refinement 时拒绝 `geometry` 和 `layer_atomic`。
- 保留 layer-atomic 已有的 point-map shape 校验。
- geometry 的逐帧 point map 必须为 `(H, W, 3)`；仅在没有 point map 时要求提供 intrinsic。
- 保留三个固定来源实现当前的无效值处理行为。
- 方法专属错误发生后，禁止静默回退到 depth 模式。

## 测试与验证

### 特征保持测试

- 不修改现有任何 layer-atomic 测试。
- 验证暴露出来的 depth stages 与原 depth wrapper 输出完全一致。
- 从 `340c599` 移植 LASER-Geometry 的核心 geometry segmentation 测试。
- 覆盖 geometry 的 `cross`、`sobel`、`legacy`、`baseline_params`、batch 辅助输入选择、区域合并和紧凑 labels。

### 路由测试

- 验证每个模式只调用对应的固定入口。
- 验证 point-map/depth 选择和 confidence 转发。
- 验证正式对比参数 `300 / 1.1 / 500` 原样到达三种方法。
- 验证 geometry 专属参数不会进入 depth 或 layer-atomic 函数。
- 验证三种方法返回的 labels 进入同一个 graph builder。

### 集成测试

- 验证普通 streaming 和 loop-closure streaming 都接受三种模式。
- 验证两个 CLI parser 暴露相同的选项与默认值。
- 验证非 depth 模式必须开启 depth refinement。
- 为三种模式运行确定性的 CPU segmentation smoke test。

### 完成门槛

- 重新构建两个 Cython 扩展。
- 从干净分支状态运行完整 pytest。
- 确认所有现有 layer-atomic 测试继续通过，且没有修改它们的预期行为。
- 对照 `feature/layer-atomic-geometry` 审查完整 diff，确认算法改动只包含移植进来的 geometry 实现；depth 和 layer-atomic 算法本体不得出现行为修改。
- 如果本机存在兼容的 CUDA 硬件和本地权重，为三种模式分别运行短序列端到端测试；否则明确记录为外部 GPU 验证项，不能声称已经在本机运行。

## 成功标准

当同一份代码可以在普通 streaming 与 loop-closure streaming 中选择 `depth`、`geometry` 或 `layer_atomic`，三者共用同一套 LASER 下游流程，正式对比的 Felzenszwalb 参数固定为 `300 / 1.1 / 500`，固定来源算法行为都有测试保护，并且没有增加指标实现时，该分支达到本次目标。
