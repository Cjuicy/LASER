# 三种分割方法的云端执行与验证

本文档用于验证分支 `codex/unified-segmentation-methods`。同一份代码可以通过
`--segment_mode` 选择以下三种已经验证过的分割方法：

- `depth`：原始 LASER 深度分割；
- `geometry`：LASER-Geometry geometry-aware 分割；
- `layer_atomic`：layer-atomic geometry 分割。

三种模式只替换 labels 的生成方法。后续 segment graph、尺度估计、尺度传播、
Sim(3)、缓存聚合和回环流程共用同一实现。

## 1. 拉取统一分支

新环境：

```bash
git clone --recursive git@github.com:Cjuicy/LASER.git
cd LASER
git checkout codex/unified-segmentation-methods
git submodule update --init --recursive
```

已有仓库：

```bash
git fetch origin
git checkout codex/unified-segmentation-methods
git pull --ff-only origin codex/unified-segmentation-methods
git submodule update --init --recursive
```

## 2. 安装与编译

```bash
conda create -n laser-unified -y python=3.11
conda activate laser-unified
pip install -r requirements.txt
pip install -e viser
python setup.py build_ext --inplace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

准备模型权重，使 `weights/model.safetensors` 存在。如果使用其他位置，下面命令的
`--model_ckpt` 必须统一替换为同一个权重文件。

## 3. 不依赖 GPU 和模型权重的快速验证

```bash
python scripts/verify_segmentation_modes.py
```

成功时必须看到三行：

```text
[PASS] mode=depth frames=2 scale=300 sigma=1.1 min_size=500
[PASS] mode=geometry frames=2 scale=300 sigma=1.1 min_size=500
[PASS] mode=layer_atomic frames=2 scale=300 sigma=1.1 min_size=500
```

然后运行完整回归测试：

```bash
python -m pytest -q
```

## 4. 同条件运行三种方法

先设置同一份输入和权重。三个命令中的 `sample_interval`、`window_size`、`overlap`
和 `depth_refine` 完全一致，只有分割模式及独立输出目录不同。

```bash
export MODEL_CKPT="weights/model.safetensors"
export DATA_PATH="examples/titanic"
```

原始 LASER depth：

```bash
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./comparison_cache/depth" \
    --output_path "./comparison_results/depth" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --depth_refine \
    --segment_mode depth
```

LASER-Geometry（正式对比档位）：

```bash
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./comparison_cache/geometry" \
    --output_path "./comparison_results/geometry" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --depth_refine \
    --segment_mode geometry \
    --normal_method cross \
    --geometry_seg_profile baseline_params
```

Layer-atomic geometry：

```bash
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./comparison_cache/layer_atomic" \
    --output_path "./comparison_results/layer_atomic" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --depth_refine \
    --segment_mode layer_atomic
```

每次启动时检查日志中的 `[segmentation]` 行。正式对比的三种模式都应显示：

```text
scale=300, sigma=1.1, min_size=500
```

geometry 还应显示 `profile=baseline_params, normal_method=cross`。历史 geometry
参数 `200 / 1.0 / 300` 仅用于复现旧实验，必须显式传入
`--geometry_seg_profile legacy`，不应用于正式对比。

## 5. 回环版本

`demo_lc.py` 接受完全相同的三个分割参数。在原有回环命令中加入：

```bash
--depth_refine \
--segment_mode geometry \
--normal_method cross \
--geometry_seg_profile baseline_params
```

把示例中的 `geometry` 分别替换成 `depth` 或 `layer_atomic` 即可运行另外两种方法。
仍建议为每种模式使用独立的 `--cache_path` 和 `--output_path`。

## 6. 成功判定

满足以下条件即可确认统一成功：

1. CPU smoke test 的三种模式全部输出 `[PASS]`；
2. 完整 pytest 无失败；
3. 三次真实序列推理均完成，且日志显示所选 mode；
4. 正式对比时三种日志均显示 `300 / 1.1 / 500`；
5. 三种结果分别写入独立目录，没有复用其他模式的缓存。
