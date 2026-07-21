# HART v2：云端执行与验证

本文档对应分支 `codex/unified-hart-anchor-propagation`。HART v2 保留原有分割、
Anchor Cell–Leaf–Parent、TrackSegment、IRLS 和层次尺度共识，在其后增加 Pose
Consensus Coupling：

```text
regional scale r = window scale g × local residual l
final current-window scale = coarse registration scale × g
```

`g` 立即进入当前窗口相机 Sim(3)；`local_residual_mask` 只修正局部点云；
`pose_support_mask` 只允许被严格多数公共尺度组支持的直接锚点参与下一窗口注册。

## 1. 全新云端环境：克隆、构建和验收

以下命令可以直接复制：

```bash
git clone --recursive --branch codex/unified-hart-anchor-propagation \
  https://github.com/Cjuicy/LASER.git LASER-HART-v2
cd LASER-HART-v2
conda create -n laser-hart-v2 -y python=3.11
conda activate laser-hart-v2
pip install -r requirements.txt
pip install -e viser
python setup.py build_ext --inplace
python scripts/verify_anchor_propagation.py
python -m pytest -q
```

已有仓库时，更新该分支并重新构建扩展：

```bash
git fetch origin codex/unified-hart-anchor-propagation
git switch codex/unified-hart-anchor-propagation
git pull --ff-only origin codex/unified-hart-anchor-propagation
git submodule update --init --recursive
python setup.py build_ext --inplace
python scripts/verify_anchor_propagation.py
python -m pytest -q
```

冒烟脚本不需要 GPU 或模型权重。四种分割模式都应输出一行 `[PASS]`，其中确定性
测试输入为 `current = 0.8 × previous`，预期：

```text
g=1.2500 residual_median=1.0000 pose_support_ratio>0
```

这同时验证公共尺度被提取、均匀尺度没有残留在局部 mask 中，并且存在可用的
pose-support。

## 2. 普通引擎：固定配置的 none/HART 对照

先设置云端路径；两次运行除传播策略和输出目录外保持相同：

```bash
export MODEL_CKPT="weights/model.safetensors"
export DATA_PATH="/path/to/sequence/images"
export SEGMENT_MODE="layer_atomic_split"
```

基线：

```bash
python demo.py \
  --model_ckpt "$MODEL_CKPT" \
  --data_path "$DATA_PATH" \
  --cache_path "./inference_cache_none" \
  --output_path "./viser_results_none" \
  --sample_interval 1 \
  --window_size 30 \
  --overlap 10 \
  --anchor_propagation none \
  --segment_mode "$SEGMENT_MODE"
```

HART v2：

```bash
python demo.py \
  --model_ckpt "$MODEL_CKPT" \
  --data_path "$DATA_PATH" \
  --cache_path "./inference_cache_hart_v2" \
  --output_path "./viser_results_hart_v2" \
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
  --split_aux_confirmation \
  2>&1 | tee hart_v2.log
```

`SEGMENT_MODE` 可设为 `depth`、`geometry`、`layer_atomic` 或
`layer_atomic_split`。显式 `--anchor_propagation hart` 不需要再传
`--depth_refine`；省略新参数时仍保持历史映射：传 `--depth_refine` 为
`legacy_iou`，否则为 `none`。

## 3. Loop Closure 引擎

使用相同的模型、数据、窗口和 HART 参数运行 LC：

```bash
export LOOP_CONFIG="configs/loop_config.yaml"

python demo_lc.py \
  --config_path "$LOOP_CONFIG" \
  --model_ckpt "$MODEL_CKPT" \
  --data_path "$DATA_PATH" \
  --cache_path "./inference_cache_lc_hart_v2" \
  --output_path "./viser_results_lc_hart_v2" \
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
  --split_aux_confirmation \
  2>&1 | tee hart_v2_lc.log
```

LC 磁盘缓存保留 raw points、raw poses、最终 pairwise Sim(3) 和
`local_residual_mask`。回环优化后的聚合只执行一次：

```text
optimized cumulative global scale × local residual × raw points
```

在线 pairwise 注册只在 pose-support 足够时使用上一窗口的
`raw × local residual`；证据不足时回退 raw，不会把 cumulative Base 当作 raw
再次缩放。

## 4. 必须检查的 HART v2 诊断

每个 HART 窗口缓存的 `hart_diagnostics` 至少应检查：

- `coarse_registration_scale`：HART 前的粗注册尺度；
- `window_scale`：Pose Consensus 输出的公共尺度 `g`；
- `final_registration_scale`：当前普通窗口或当前 LC pairwise 的最终尺度；
- `pose_consensus_accepted`、`pose_consensus_support_ratio`：公共组是否满足严格多数；
- `registration_pose_support_used`：下一窗口是否实际采用 pose-support；
- `local_residual_min/median/max`：局部残差范围；
- `conflict_pixel_ratio`：冲突区域比例。

核心恒等式应满足：

```text
final_registration_scale ~= coarse_registration_scale * window_scale
```

首窗口固定为 `coarse=1`、`g=1`、`final=1`。当 Pose Consensus 不满足严格多数时，
`g=1`、pose-support 为空，注册安全回退，不会由小物体 Track 改变整窗相机尺度。
普通和 LC 引擎都会把同一组关键值输出为 `[HART v2]` 日志，可直接提取：

```bash
grep '^\[HART v2\]' hart_v2.log
grep '^\[HART v2\]' hart_v2_lc.log
```

## 5. ATE/RPE 的正确解释

仓库正式位姿评估在 `eval/vo_eval.py` 中对 ATE、RPE translation 和 RPE rotation
均使用 `align=True, correct_scale=True`。因此，如果 HART 只造成整条预测轨迹的统一
全局缩放，评估会把该差异对齐掉，最终 ATE 数字可以完全相同；这不能用来判断 HART
是否进入了位姿链。

云端比较时必须同时记录：

1. `coarse_registration_scale / window_scale / final_registration_scale`；
2. `pose_consensus_support_ratio` 与 `registration_pose_support_used`；
3. `none` 与 `hart` 输出的原始 `pred_traj.txt` 相机 translation；
4. 正式 scale-corrected ATE/RPE；
5. 点云或深度质量指标。

只有第 3 项能直接证明当前窗口相机轨迹是否发生变化；第 4 项回答的是经过全局尺度
对齐后的轨迹形状误差。两者含义不同，都应保留。

设置两个输出轨迹后，可以直接量化未经对齐的相机 translation 差异：

```bash
export NONE_TRAJ="./viser_results_none/SCENE_NAME/pred_traj.txt"
export HART_TRAJ="./viser_results_hart_v2/SCENE_NAME/pred_traj.txt"

python - "$NONE_TRAJ" "$HART_TRAJ" <<'PY'
import sys
import numpy as np

none = np.loadtxt(sys.argv[1])
hart = np.loadtxt(sys.argv[2])
if none.shape != hart.shape:
    raise SystemExit(f"trajectory shape mismatch: {none.shape} != {hart.shape}")
delta = np.linalg.norm(none[:, 1:4] - hart[:, 1:4], axis=1)
print(f"raw translation delta: mean={delta.mean():.9f}, max={delta.max():.9f}")
PY
```

如需对同一 TUM 格式真值补充“不校正尺度”的 ATE，保留正式指标并额外运行第二条：

```bash
export GT_TRAJ="/path/to/groundtruth_tum.txt"
evo_ape tum "$GT_TRAJ" "$HART_TRAJ" --align --correct_scale
evo_ape tum "$GT_TRAJ" "$HART_TRAJ" --align
```

第一条与仓库正式 scale-corrected 语义一致；第二条仅是诊断补充，不能替换正式基线。

## 6. 路由边界

- `none`：只做全局窗口注册，不运行分割或 HART；
- `legacy_iou`：保留原图结构和 IoU 加权传播；
- `hart`：Anchor HART + Pose Consensus Coupling。

HART 无证据或发生冲突时只执行其定义内的单位尺度回退，不会静默切换到
`legacy_iou`。正式对比时固定窗口、overlap、置信度比例、
`anchor_min_pixels` 和 `scale_consistency_thresh`，并为每个路由使用独立输出目录。
