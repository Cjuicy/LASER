# 三方法分割诊断：云端执行与验收

本文档是三方法分割诊断的唯一云端操作流程。诊断入口是只读观测：不会改变默认
推理路径，也不包含回环、参数 sweep 或额外对比配置。

## 1. 固定实验口径

每次运行严格按以下顺序串行执行三项配置；不得并行占用多份 GPU 或诊断磁盘：

1. `depth`：深度基线，Felzenszwalb `300 / 1.1 / 500`；
2. `geometry_baseline`：几何基线，Felzenszwalb `300 / 1.1 / 500`，固定
   `normal_method=cross`；
3. `layer_atomic_split`：分层原子分割，Felzenszwalb `300 / 1.1 / 500`，固定
   `normal_method=cross`、`split_score_thresh=0.10`、
   `split_aux_confirmation=true`。

三项配置共享同一个 checkpoint；入口会把 checkpoint SHA-256、git commit、数据指纹、
窗口参数、按顺序排列的三项完整有效参数、序列清单和评估签名写入
`manifest.json` 的 `experiment_contract`。轨迹使用一次全序列 Sim(3) 对齐，
RPE 使用 `delta=1 frame, all_pairs=True`。引擎为无回环的
`StreamingWindowEngine`。

默认重点分组为 Recovery：02、04、10；Stability Guard：00、05、09。选择器默认最多
48 个区间；在 50 GiB 硬上限下全量 00–10 使用 `--max-selected 12`，以保留重点序列的
事件与 matched control 覆盖。

## 2. 获取代码和准备环境

```bash
git fetch origin codex/segmentation-diagnostics
git checkout codex/segmentation-diagnostics

conda create -n laser-diag -y python=3.11
conda activate laser-diag
pip install -r requirements.txt
python setup.py build_ext --inplace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

KITTI Odometry 根目录必须包含图像和 pose：

```text
/data/KITTI_Odometry/dataset/
├── sequences/
│   ├── 00/image_2/000000.png ...
│   └── 10/image_2/...
└── poses/
    ├── 00.txt
    └── 10.txt
```

```bash
export KITTI_ROOT=/data/KITTI_Odometry/dataset
export LASER_CKPT=$PWD/weights/model.safetensors
export DIAG_OUT=$PWD/results/segmentation-diagnostics
export DIAG_TMP=$PWD/cache/segmentation-diagnostics
```

## 3. 先做 CPU 无权重验收

此命令不加载模型，也不需要 KITTI 或 GPU。它真实验证 schema 2.0、staged/public
split parity、三方法 dense 渲染、两个 regret、所有 selection reason、存储上下限与离线
HTML 报告。

```bash
python scripts/verify_segmentation_diagnostics.py \
  --output-dir /tmp/laser-split-diagnostics-verify
```

成功输出包含：

```text
[PASS] schema 2.0
[PASS] parity
[PASS] storage
[PASS] selection
[PASS] rendering
[PASS] report
```

报告位于 `/tmp/laser-split-diagnostics-verify/report/index.html`。这只证明 CPU
诊断链路；没有 KITTI、checkpoint 和 GPU 的真实运行时，不能据此声明 ATE 结果。

## 4. 全量 00–10 预检（不加载模型）

先以最终输出目录运行 dry-run：

```bash
python scripts/run_segmentation_diagnostics.py \
  --dataset-root "$KITTI_ROOT" \
  --model-ckpt "$LASER_CKPT" \
  --output-dir "$DIAG_OUT" \
  --temp-root "$DIAG_TMP" \
  --device cuda \
  --max-selected 12 \
  --dry-run
```

它校验 00–10 图像/pose 布局、checkpoint SHA-256、数据指纹、窗口参数及磁盘空间，
并估算两遍 trace、case 和报告的空间上界，不执行推理。预测超过 50 GiB 会在加载模型前
拒绝。dry-run 会写入 manifest；随后同一目录的正式运行必须使用 `--resume`。

## 5. 先跑 KITTI 04 小规模 GPU 验证

为避免和全量结果混用，使用独立输出目录：

```bash
python scripts/run_segmentation_diagnostics.py \
  --dataset-root "$KITTI_ROOT" \
  --model-ckpt "$LASER_CKPT" \
  --output-dir "$PWD/results/segmentation-diagnostics-04" \
  --temp-root "$PWD/cache/segmentation-diagnostics-04" \
  --sequences 04 \
  --window-size 20 \
  --overlap 5 \
  --device cuda \
  --max-selected 12 \
  --max-temp-gib 50 \
  --warn-temp-gib 40 \
  --min-free-gib 10
```

日志必须按三项配置串行出现：

```text
[phase pass1 1/3] depth
[phase pass1 2/3] geometry_baseline
[phase pass1 3/3] layer_atomic_split
...
[phase complete] report: .../report/index.html
```

该命令完成 Pass 1、自动选区间、三方法 Pass 2、PNG/PLY 渲染和离线报告。完成 KITTI 04
不代表全量 ATE 已验证；全量结论只能来自下一节的实际 00–10 结果。

## 6. 正式运行 KITTI 00–10

若第 4 节使用相同 `$DIAG_OUT` 成功完成 dry-run，执行：

```bash
python scripts/run_segmentation_diagnostics.py \
  --dataset-root "$KITTI_ROOT" \
  --model-ckpt "$LASER_CKPT" \
  --output-dir "$DIAG_OUT" \
  --temp-root "$DIAG_TMP" \
  --window-size 20 \
  --overlap 5 \
  --top-conf-percentile 0.3 \
  --device cuda \
  --max-temp-gib 50 \
  --warn-temp-gib 40 \
  --min-free-gib 10 \
  --max-selected 12 \
  --resume
```

没有先运行 dry-run 时，使用一个新的空输出目录并去掉 `--resume`。Pass 1 只写 pose
shard 和标量 JSONL；Pass 2 从每个选中序列开头完整重跑以保留尺度传播历史，但仅为
selected-frame union 保存 RGB、point map、labels 和 scale trace。

## 7. 中断、恢复和空间保护

- 40 GiB 时进入 warning；任何写入若预测超过 50 GiB 会立即停止；可用空间必须至少保留
  10 GiB。
- `.partial` 仅存在于原子写入期间；临时目录必须带本次 run-id ownership marker，清理
  只允许删除已验证属于本次运行的目录。
- SIGTERM 会将 manifest 标为 `interrupted` 并释放锁。进程被强杀后，`--resume` 只会回收
  同 run-id、同主机且 PID 已不存在的 stale lock。
- 每个 sequence checkpoint 保存轨迹、JSONL、dense trace 的大小与 SHA-256。缺失或篡改
  会使对应 sequence 自动重跑；commit、checkpoint、配置或数据指纹变化时拒绝混用旧结果。

中断后，重复原正式命令并附带 `--resume`。只重建报告、不运行 worker 时使用：

```bash
python scripts/run_segmentation_diagnostics.py \
  --dataset-root "$KITTI_ROOT" \
  --model-ckpt "$LASER_CKPT" \
  --output-dir "$DIAG_OUT" \
  --temp-root "$DIAG_TMP" \
  --report-only
```

## 8. 输出与成功标准

```text
$DIAG_OUT/
├── manifest.json
├── summary.json
├── selected_intervals.json
├── selection_records.json
├── checkpoints/pass1|pass2/<config>/<sequence>.json
├── trajectory/<config>/<sequence>.json|npz
├── trajectory/regret/<sequence>.json|npz
├── artifacts/<config>/<sequence>/pass1|pass2/
├── cases/<sequence>/<interval>/<config>/
│   ├── PNG、segments.ply、rendering.json、metrics.json
│   └── （三个 config 均存在）
├── cases/<sequence>/<interval>/comparison-rendering.json
├── cases/<sequence>/<interval>/trajectory-timeline.json
└── report/
    ├── index.html
    ├── metrics.csv
    └── cases/*.html
```

正式成功必须同时满足：

1. `manifest.json` 的 `status` 为 `complete`，schema 为 `2.0`；
2. manifest 的 `experiment_contract` 精确记录按顺序排列的三项完整配置（包括 geometry
   `normal_method=cross`）、00–10、窗口/overlap 和全序列 Sim(3) + RPE 评估签名；
3. `summary.json` 仅含 `depth`、`geometry_baseline`、`layer_atomic_split` 三种方法的
   ATE/RPE、Stability Guard 和 Recovery；
4. `trajectory/regret` 持久化两条逐帧 regret；窗口记录同时包含两条 regret 的
   mean/max/正面积/正持续长度/最长持续段/change point，缺失比较保持 `null`；
5. `selected_intervals.json` 可追溯每个区间的 score/reasons，`summary.json` 对六种选择原因
   给出明确的 available/selected/reason coverage；
6. `report/index.html` 可离线打开，包含每条请求序列的两条 regret timeline、split activity、
   正式排名/aggregate、split scatter，以及案例的 parent/child、pre/post geometry 和局部
   三误差/两 regret 证据；缺失的可选
   scale/temporal 证据必须显示 `UNAVAILABLE`，不能以单位尺度或伪造热图代替；
7. 实际 00–10 GPU 运行完成前，不报告或推广任何 ATE 结论。

建议先检查 overview 的 Stability Guard 和 02/04/10 Recovery，再按
`Segmentation → Merge → Scale → Trajectory` 查看 selected case。相关性只支持排查，
不表示因果关系。
