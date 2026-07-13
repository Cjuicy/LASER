# 回环检测云端可运行设计

## 目标

让 AutoDL 上现有的 LASER 仓库能够对 `data/00/image_2` 这类平铺图片序列可靠地执行回环检测推理，并默认使用当前分支中的 layer-atomic 新分割方法。Pi3、SALAD 和 DINO 权重已经位于 `weights/`，本次改动不下载、不复制、也不重新打包这些权重。

实现基于提交 `7dca6248b58ebbeacdda9cdf108aab439c527479`。在这个提交中，只要开启深度细化，layer-atomic 几何分割就已经接入 `StreamingWindowEngineLC`。

## 改动范围

本次会完成：

- 创建唯一的图片清单，统一自然排序和采样；
- 流式推理、SALAD 检索和回环约束构建复用完全相同的图片清单；
- `demo_lc.py` 默认开启 layer-atomic 深度细化；
- 保留显式的 `--no-depth-refine` 关闭选项；
- 默认使用 `configs/loop_config.yaml`；
- 在昂贵的模型推理开始前检查 AutoDL 本地输入；
- 回环检测开始前，检查实际生成的窗口缓存数量是否符合预期；
- 输出简洁的运行模式、帧数、窗口数和回环对诊断日志；
- 给出可直接复制的 AutoDL 拉取、测试和运行命令；
- 增加针对性的单元测试与回归测试。

本次不会：

- 修改 layer-atomic 分割算法或阈值；
- 修改 SALAD 相似度阈值、NMS、Sim(3) 估计或优化算法；
- 增加 Docker、Conda、数据集下载或权重下载功能；
- 把 `demo.py` 和 `demo_lc.py` 合并成新的统一入口；
- 修改缓存张量格式或可视化输出格式。

## 唯一图片清单

新建 `utils/image_paths.py`，作为仓库中唯一的图片发现与排序实现。它提供以下接口：

```text
natural_sort_key(path: str | os.PathLike[str]) -> list[tuple[int, str | int]]
discover_images(data_path: str | os.PathLike[str], sample_interval: int = 1) -> list[str]
```

`discover_images` 的行为：

1. 要求 `data_path` 是已经存在的目录；
2. 要求 `sample_interval >= 1`；
3. 不区分大小写地接受 `.png`、`.jpg` 和 `.jpeg`；
4. 对文件名中的数字按数值排序，例如 `000002.png`、`frame2.png` 都排在 `000010.png`、`frame10.png` 之前；
5. 先完成自然排序，再且仅再应用一次 `sample_interval`；
6. 返回可供现有图片加载器直接使用的路径字符串列表。

`demo.py` 和 `demo_lc.py` 都会导入这个公共实现，不再各自保存一份排序代码。导入后的函数名仍然可以通过 `demo` 和 `demo_lc` 模块访问，从而保持现有测试和调用方式兼容。

`demo_lc.py` 只扫描一次输入目录并生成图片清单，然后把同一个有序列表传给流式推理和 `LoopClosureEngine`。回环引擎再把这个列表传给 `LoopDetector`。只要调用方提供了显式清单，回环引擎和 SALAD 检测器都不得重新扫描目录，也不得再次采样。

为了保持向后兼容，`LoopClosureEngine` 和 `LoopDetector` 仍然支持原有的“只传目录”调用方式。只有没有收到显式图片清单时，它们才会调用公共图片发现函数。

## 回环检测入口

`demo_lc.py` 继续作为云端运行入口。命令行参数调整如下：

```text
--config_path 默认值为 configs/loop_config.yaml
--depth-refine 显式开启深度细化
--no-depth-refine 显式关闭深度细化
深度细化默认开启
```

因此，普通回环命令不需要额外参数就会调用 `segment_point_map_layer_atomic`。启动日志会明确输出：

```text
Loop closure: enabled
Layer-atomic depth refinement: enabled|disabled
```

场景名默认取输入图片目录名。对于 `data/00/image_2`，默认场景名是 `image_2`；仍可通过 `--scene_name kitti_00` 指定更清楚的名字。

## 启动检查与错误处理

构造模型之前，入口会检查：

- CUDA 可用，因为当前入口使用 CUDA 推理和计时；
- `window_size > overlap >= 1`；
- Pi3 主权重存在；
- 回环配置文件存在且可以成功加载；
- 此回环专用入口中的 `Model.loop_enable` 为 `true`；
- 配置中指定的 SALAD 和 DINO 权重存在；
- 图片清单非空，而且图片数量大于 `overlap`。

任何检查失败都会在 GPU 推理开始前直接报错，并指出具体缺失或非法的值。

运行器只计算一次滑动窗口并记录预期窗口数。流式推理结束后，按缓存编号进行数值排序，并要求实际缓存数与预期窗口数完全一致。这样，后台线程异常或缓存输出不完整不会静默进入回环优化，而会得到清楚的错误信息。

没有检测到回环对属于正常结果，不应导致程序崩溃。此时日志会显示回环对数量为零，程序保留相邻窗口产生的 Sim(3) 变换，继续聚合缓存并保存可视化结果。

## 数据流

```text
data/00/image_2
    -> 公共自然排序和唯一一次采样
    -> 唯一图片清单
       -> StreamingWindowEngineLC
          -> 相邻窗口配准
          -> layer-atomic 分割和深度细化（默认开启）
          -> 带编号的窗口缓存
       -> 使用相同帧索引的 SALAD LoopDetector
       -> LoopClosureEngine 构建回环窗口约束
       -> Sim(3) 优化
       -> 修正后的缓存聚合
       -> viser_results/kitti_00
```

整个流程必须保持一个核心不变量：任意阶段中的帧索引 `i` 都对应完全相同的图片路径。

## 运行日志

云端日志会包含：

- 输入图片目录；
- 采样后的帧数以及第一张、最后一张图片文件名；
- 窗口大小、重叠帧数和预期窗口数；
- 回环检测和深度细化是否开启；
- 已完成的缓存窗口数量；
- SALAD 检测到的回环对数量；
- 是否执行了 Sim(3) 优化，或者因没有回环而保留原始变换；
- 最终输出目录和场景名。

日志不会输出完整图片清单或大型张量。

## 测试设计

针对性测试会覆盖：

1. 混合大小写的 `.png`、`.jpg`、`.jpeg` 图片筛选；
2. 文件名数字的自然排序，以及排序完成后才执行采样；
3. 非法目录和非法采样间隔的错误；
4. `demo.py` 和 `demo_lc.py` 都使用公共图片工具；
5. 回环入口默认使用标准配置并开启深度细化；
6. `--no-depth-refine` 可以关闭新分割路径；
7. 显式图片清单可以不经重新扫描或重复采样，原样传递给 `LoopClosureEngine` 和 `LoopDetector`；
8. 缓存数量不符合预期时给出明确错误；
9. 零回环对时保留已有变换；
10. 现有分割测试、demo 测试和完整测试集继续通过。

本地验证命令：

```bash
python setup.py build_ext --inplace
pytest -q
```

本地仓库没有大型权重和 AutoDL GPU，因此无法在本地复现完整模型推理。最终交付会包含一个 AutoDL 轻量冒烟命令和 KITTI 00 完整运行命令。

## AutoDL 交付方式

实现分支推送后，在云端执行：

```bash
cd ~/autodl-tmp/LASER
git fetch origin codex/loop-closure-cloud-ready
git switch codex/loop-closure-cloud-ready
python setup.py build_ext --inplace
pytest -q
```

使用已有权重完整运行 KITTI 00：

```bash
python demo_lc.py \
  --data_path data/00/image_2 \
  --scene_name kitti_00 \
  --cache_path inference_cache/kitti_00 \
  --output_path viser_results \
  --window_size 10 \
  --overlap 5
```

因为 layer-atomic 深度细化是回环模式默认值，所以不需要添加 `--depth-refine`。只有需要进行基线对比时才添加 `--no-depth-refine`。
