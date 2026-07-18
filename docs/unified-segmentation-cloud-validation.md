# 四种分割方法的云端执行与验证

本文档用于验证分支 `codex/auto-post-merge-split`。同一份代码可以通过
`--segment_mode` 选择以下四种分割方法：

- `depth`：原始 LASER 深度分割；
- `geometry`：LASER-Geometry geometry-aware 分割；
- `layer_atomic`：layer-atomic geometry 分割。
- `layer_atomic_split`：在 layer-atomic Auto merge 后，对合格大区域执行一次最多
  4 叶的法向 watershed 恢复；RGB 或归一化三维间隙只负责辅助确认。

四种模式只替换 labels 的生成方法。后续 segment graph、尺度估计、尺度传播、
Sim(3)、缓存聚合和回环流程共用同一实现。

## 1. 拉取统一分支

新环境：

```bash
git clone --recursive git@github.com:Cjuicy/LASER.git
cd LASER
git checkout codex/auto-post-merge-split
git submodule update --init --recursive
```

已有仓库：

```bash
git fetch origin
git checkout codex/auto-post-merge-split
git pull --ff-only origin codex/auto-post-merge-split
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

成功时必须看到四行：

```text
[PASS] mode=depth frames=2 scale=300 sigma=1.1 min_size=500
[PASS] mode=geometry frames=2 scale=300 sigma=1.1 min_size=500
[PASS] mode=layer_atomic frames=2 scale=300 sigma=1.1 min_size=500
[PASS] mode=layer_atomic_split frames=2 scale=300 sigma=1.1 min_size=500
```

然后运行完整回归测试：

```bash
python -m pytest -q
```

## 4. 同条件运行四种方法

先设置同一份输入和权重。所有命令中的 `sample_interval`、`window_size`、`overlap`
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

Layer-atomic + Auto 后分割（正式方案，辅助确认开启）：

```bash
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./comparison_cache/layer_atomic_split" \
    --output_path "./comparison_results/layer_atomic_split" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --depth_refine \
    --segment_mode layer_atomic_split \
    --normal_method cross \
    --split_score_thresh 0.10 \
    --split_aux_confirmation
```

法向单独判断的配对消融（方法和阈值完全相同，只关闭辅助确认）：

```bash
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./comparison_cache/layer_atomic_split_no_aux" \
    --output_path "./comparison_results/layer_atomic_split_no_aux" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --depth_refine \
    --segment_mode layer_atomic_split \
    --normal_method cross \
    --split_score_thresh 0.10 \
    --no-split_aux_confirmation
```

这个开关不引入另一套分割方法或阈值。开启时评分为
`法向收益 × max(RGB 对比, 归一化三维间隙对比)`；关闭时直接使用法向收益，且
不会计算 RGB/间隙边缘。

每次启动时检查日志中的 `[segmentation]` 行。正式对比的四种模式都应显示：

```text
scale=300, sigma=1.1, min_size=500
```

geometry 还应显示 `profile=baseline_params, normal=cross`。历史 geometry
参数 `200 / 1.0 / 300` 仅用于复现旧实验，必须显式传入
`--geometry_seg_profile legacy`，不应用于正式对比。

## 5. 回环版本

`demo_lc.py` 接受完全相同的分割参数。在原有回环命令中加入：

```bash
--depth_refine \
--segment_mode geometry \
--normal_method cross \
--geometry_seg_profile baseline_params
```

把示例中的 `geometry` 替换成其他 mode 即可运行对应方法。`layer_atomic_split`
还应加入 `--split_score_thresh 0.10 --split_aux_confirmation`。
仍建议为每种模式使用独立的 `--cache_path` 和 `--output_path`。

## 6. 成功判定

满足以下条件即可确认统一成功：

1. CPU smoke test 的四种模式全部输出 `[PASS]`；
2. 完整 pytest 无失败；
3. 四种真实序列推理均完成，且日志显示所选 mode；
4. 正式对比时四种日志均显示 `300 / 1.1 / 500`；
5. 各结果分别写入独立目录，没有复用其他模式的缓存。

## 7. 保存 trace 的配对评估

同时评估 17 条 KITTI trace 和 12 条 TUM trace（包括
`freiburg1_360`、`freiburg1_desk`）：

```bash
python scripts/evaluate_post_merge_split.py \
    --trace-glob '/tmp/laser-case-*-trace.npz' \
    --trace-glob '/tmp/laser-tum-*-trace.npz' \
    --thresholds 0.10 0.15 0.20 \
    --aux-states on off \
    --repeats 30 \
    --output-dir ./post_merge_split_evaluation
```

脚本生成 `summary.json`、`per_trace.json` 和四个固定场景的 PNG 对比图。生产阈值
只根据辅助确认开启状态选择；关闭状态始终使用同一阈值，仅作为消融对照。
