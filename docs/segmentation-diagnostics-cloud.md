# 三方法分割诊断：云端执行与验收

本文档对应分支 `codex/segmentation-diagnostics`。系统不会修改已经验证的
`depth`、`geometry`、`layer_atomic` 默认算法，只在显式运行诊断入口时增加只读观测。

## 1. 固定实验口径

一条命令会按以下顺序串行运行，不并行占用四份 GPU/磁盘：

1. `depth`：正式基线，Felzenszwalb `300 / 1.1 / 500`；
2. `geometry_baseline`：正式旧 Geometry 对比，`300 / 1.1 / 500`；
3. `layer_atomic`：当前方法，`300 / 1.1 / 500`；
4. `geometry_legacy_reference`：机制参考，历史参数 `200 / 1.0 / 300`，不进入正式排名。

目标是无回环 `StreamingWindowEngine`。四项配置共享一个 checkpoint；入口会计算
SHA-256 并写入 manifest，worker 载入前再次核验。轨迹统一使用一次全序列 Sim(3)
对齐，RPE 使用 `delta=1 frame, all_pairs=True`。

默认分组：

- Recovery：02、04、10；
- Stability Guard：00、05、09；
- 全局保护：00–10；
- 选择器能力上限默认 48；在 50 GiB 硬上限下，全量 00–10 推荐 `--max-selected 12`，
  仍强制覆盖六个重点序列各一个事件与一个 matched control。若希望 48 个区间，必须提高
  空间上限并先通过 dry-run，不能绕过预检。

## 2. 获取分支和准备环境

```bash
git fetch origin codex/segmentation-diagnostics
git checkout codex/segmentation-diagnostics

conda create -n laser-diag -y python=3.11
conda activate laser-diag
pip install -r requirements.txt
python setup.py build_ext --inplace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

KITTI Odometry 目录必须是：

```text
/data/KITTI_Odometry/dataset/
├── sequences/
│   ├── 00/image_2/000000.png ...
│   └── 10/image_2/...
└── poses/
    ├── 00.txt
    └── 10.txt
```

下文用变量避免把路径写错：

```bash
export KITTI_ROOT=/data/KITTI_Odometry/dataset
export LASER_CKPT=$PWD/weights/model.safetensors
export DIAG_OUT=$PWD/results/segmentation-diagnostics
export DIAG_TMP=$PWD/cache/segmentation-diagnostics
```

## 3. 先做 CPU 无权重验收

此命令不加载模型、不需要 KITTI/GPU，会真实执行 schema、layer-atomic parity、存储阈值、
选区间、PNG/PLY 渲染与 HTML 报告链路：

```bash
python scripts/verify_segmentation_diagnostics.py \
  --output-dir /tmp/laser-diagnostics-verify
```

成功时必须依次看到：

```text
[PASS] schema
[PASS] parity
[PASS] storage
[PASS] selection
[PASS] rendering
[PASS] report
```

报告位于 `/tmp/laser-diagnostics-verify/report/index.html`。

## 4. 预检（不加载模型）

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

它验证 00–10 图像/pose 布局、checkpoint SHA-256、数据指纹、窗口参数和磁盘空间，
输出两遍 trace、案例和报告的空间上界，不执行推理；预测超过 50 GiB 会在加载模型前拒绝。
异常候选的合并跨度也被限制在 dry-run 使用的同一上下文上界内，避免相邻候选链式合并
后突破估算。
完成后正式运行应在同一命令后加 `--resume`，复用
已经写入的 manifest：

## 5. 先跑 KITTI 04 小规模 GPU 验证

为避免与全量输出混用，使用独立目录：

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

虽然只写一条命令，它内部仍完成四配置 Pass 1 → 自动选区间 → 四配置完整序列 Pass 2 →
PNG/PLY/HTML。日志中的配置必须严格串行：

```text
[phase pass1 1/4] depth
[phase pass1 2/4] geometry_baseline
[phase pass1 3/4] layer_atomic
[phase pass1 4/4] geometry_legacy_reference
...
[phase complete] report: .../report/index.html
```

## 6. 正式运行 KITTI 00–10

如果第 4 节已经用相同 `$DIAG_OUT` 做过 `--dry-run`，使用：

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

如果未做 dry-run，去掉最后的 `--resume`。Pass 1 只写 pose shard 和标量 JSONL；Pass 2
从每个选中序列开头完整重跑以保留尺度传播历史，但只为 selected-frame union 保存重型
RGB、point map、labels、scale trace。

## 7. 中断、空间保护和恢复

- 临时/诊断写入到 40 GiB 进入 warning 状态；
- 写入前预测会超过 50 GiB 时立即停止，不继续制造 partial 文件；
- 可用空间不得低于 10 GiB；
- `.partial` 文件只在原子写入过程中存在；
- 清理只允许删除带本次 run-id ownership marker 的目录；
- 四配置不并行；metrics-only engine cache 每个序列结束即删除。
- SIGTERM 会先把 manifest 写成 `interrupted` 并释放锁；若进程被强杀，`--resume` 只会
  回收同 run-id、同主机且 PID 已不存在的 stale lock。
- 每个 sequence checkpoint 保存轨迹、JSONL、dense trace 的大小与 SHA-256；缺失或篡改
  会使对应 sequence 自动重跑，不能仅靠 manifest 的 wildcard 状态跳过。

中断后原命令加 `--resume`。系统核验 commit、checkpoint、配置和数据指纹；任一发生变化
会拒绝混用旧结果：

```bash
python scripts/run_segmentation_diagnostics.py \
  --dataset-root "$KITTI_ROOT" \
  --model-ckpt "$LASER_CKPT" \
  --output-dir "$DIAG_OUT" \
  --temp-root "$DIAG_TMP" \
  --device cuda \
  --max-selected 12 \
  --resume
```

只重建报告、不调用 worker：

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
├── artifacts/<config>/<sequence>/pass1|pass2/
├── cases/<sequence>/<interval>/<config>/
│   ├── 14 类 PNG
│   ├── segments.ply
│   ├── rendering.json
│   └── metrics.json
├── cases/<sequence>/<interval>/trace.npz
├── cases/<sequence>/<interval>/artifact-manifest.json
└── report/
    ├── index.html
    ├── metrics.csv
    └── cases/*.html
```

正式成功必须同时满足：

1. `manifest.json` 的 `status` 为 `complete`；
2. commit、checkpoint SHA-256、四配置、00–10 和 50/40/10 GiB 阈值正确；
3. `summary.json` 有三正式方法的 ATE/RPE、Stability Guard 与 Recovery；
4. `selected_intervals.json` 有可追溯 reasons/score；
5. `report/index.html` 可离线打开，案例页面能访问 15 类 PNG/PLY；缺失的可选 scale/
   temporal 证据必须显示 `UNAVAILABLE`，不得用单位尺度或伪造热图代替；
6. 正常三方法文件和 Felzenszwalb 参数未发生变化。

建议先看 overview 的 Stability Guard 和 02/04/10 Recovery，再打开 selected case，按
`Segmentation → Merge → Scale → Trajectory` 顺序观察。报告中的相关性只用于支持排查，
不把相关关系写成因果结论。
