# HART 锚点传播：云端执行与验证

本文档对应分支 `codex/unified-hart-anchor-propagation`。HART 是独立旁路，统一支持
`depth / geometry / layer_atomic / layer_atomic_split` 四种分割；旧 `--depth_refine`
行为保持兼容。

## 1. 拉取与 CPU 快速验收

已有 LASER 仓库时，可直接复制执行：

```bash
git fetch origin codex/unified-hart-anchor-propagation && \
git switch codex/unified-hart-anchor-propagation && \
git pull --ff-only origin codex/unified-hart-anchor-propagation && \
git submodule update --init --recursive && \
python setup.py build_ext --inplace && \
python scripts/verify_anchor_propagation.py && \
python -m pytest -q
```

全新环境先安装项目依赖：

```bash
git clone --recursive --branch codex/unified-hart-anchor-propagation \
    https://github.com/Cjuicy/LASER.git LASER-HART
cd LASER-HART
conda create -n laser-hart -y python=3.11
conda activate laser-hart
pip install -r requirements.txt
pip install -e viser
python setup.py build_ext --inplace
python scripts/verify_anchor_propagation.py
python -m pytest -q
```

冒烟脚本无需 GPU 和模型权重，成功时为四种模式各输出一行 `[PASS]`，并报告直接
锚点数和尺度 mask 的最小值、中位数、最大值。

## 2. 单条普通推理命令

把 `MODEL_CKPT` 和 `DATA_PATH` 改成云端真实路径；`SEGMENT_MODE` 可替换为四种模式
中的任意一种。

```bash
MODEL_CKPT="weights/model.safetensors" \
DATA_PATH="examples/titanic" \
SEGMENT_MODE="layer_atomic_split" \
python demo.py \
    --model_ckpt "$MODEL_CKPT" \
    --data_path "$DATA_PATH" \
    --cache_path "./inference_cache_hart_${SEGMENT_MODE}" \
    --output_path "./viser_results_hart_${SEGMENT_MODE}" \
    --sample_interval 1 \
    --window_size 30 \
    --overlap 10 \
    --anchor_propagation hart \
    --anchor_min_pixels 64 \
    --scale_consistency_thresh 0.05 \
    --segment_mode "$SEGMENT_MODE" \
    --normal_method cross \
    --geometry_seg_profile baseline_params \
    --split_score_thresh 0.10 \
    --split_aux_confirmation
```

显式 `--anchor_propagation hart` 已经决定新旁路，不需要同时传
`--depth_refine`。如果不传新参数，旧映射仍然成立：带 `--depth_refine` 等价于
`legacy_iou`，不带则等价于 `none`。

## 3. LC 推理

把上面的 `python demo.py` 替换为：

```bash
python demo_lc.py --config_path configs/loop_config.yaml ...
```

其余 HART 和四模式参数完全相同。LC 缓存保留原始窗口点云和 pairwise Sim(3)，
HART 仅新增纯局部 `local_scale_mask`；回环优化后按“优化后的全局尺度 × 局部
mask × 原始点云”应用一次。

在线传播不是只改当前输出：普通引擎把 `Base × local mask` 的尾部点云送入下一
窗口 Sim(3)；LC 在内存中把 `raw × local mask` 送入下一 pairwise Sim(3)。因此
HART 会逐窗口改变后续相机位姿并进入 ATE 计算链，同时磁盘 raw cache 仍不被覆盖。

## 4. 对照实验路由

- `--anchor_propagation none`：只做全局窗口配准，不运行分割或局部传播；
- `--anchor_propagation legacy_iou`：严格使用原始图结构和 IoU 加权传播；
- `--anchor_propagation hart`：使用层次锚点、TrackSegment 与冲突隔离传播。

正式四方法对比时，仅改变 `--segment_mode` 和输出目录，保持窗口、overlap、置信度
比例、`anchor_min_pixels` 与 `scale_consistency_thresh` 相同。
