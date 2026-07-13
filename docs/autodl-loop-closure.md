# AutoDL 回环检测运行手册

本文档用于在现有 AutoDL 目录 `~/autodl-tmp/LASER` 中运行带回环检测的 LASER。数据和权重保持在现有位置，不需要重新下载。

## 1. 拉取实现分支

```bash
cd ~/autodl-tmp/LASER
git fetch origin codex/loop-closure-cloud-ready
git switch codex/loop-closure-cloud-ready
```

确认当前版本：

```bash
git branch --show-current
git log -1 --oneline
```

分支名应为 `codex/loop-closure-cloud-ready`。

## 2. 编译并运行测试

```bash
python setup.py build_ext --inplace
pytest -q
```

如果 AutoDL 环境尚未安装项目依赖，再执行：

```bash
pip install -r requirements.txt
pip install faiss-gpu-cu12 numpy==1.26.4
```

## 3. 检查 KITTI 00 和权重

```bash
ls data/00/image_2 | head
ls weights/model.safetensors \
   weights/dino_salad.ckpt \
   weights/dinov2_vitb14_pretrain.pth
```

`demo_lc.py` 会在模型推理前检查 CUDA、窗口参数、配置文件、三份权重和图片数量。任意输入不完整时会直接给出对应路径，不会先占用 GPU 跑完整序列。

## 4. 先做冒烟运行

冒烟运行每 10 帧取一张，用于先验证“流式推理 → 新分割 → SALAD 回环检测 → Sim(3) 优化 → 保存结果”的完整链路：

```bash
python demo_lc.py \
  --data_path data/00/image_2 \
  --scene_name kitti_00_smoke \
  --cache_path inference_cache/kitti_00_smoke \
  --output_path viser_results \
  --sample_interval 10 \
  --window_size 10 \
  --overlap 5
```

## 5. 完整运行 KITTI 00

冒烟运行成功后执行完整序列：

```bash
python demo_lc.py \
  --data_path data/00/image_2 \
  --scene_name kitti_00 \
  --cache_path inference_cache/kitti_00 \
  --output_path viser_results \
  --window_size 10 \
  --overlap 5
```

默认配置已经是 `configs/loop_config.yaml`，默认 Pi3 权重已经是 `weights/model.safetensors`，因此上面的命令不需要重复指定。

## 6. 新分割开关

回环入口现在默认开启 layer-atomic 深度细化，不需要添加 `--depth-refine`。启动日志应显示：

```text
Loop closure: enabled
Layer-atomic depth refinement: enabled
```

只有进行旧基线对比时才关闭：

```bash
python demo_lc.py \
  --data_path data/00/image_2 \
  --scene_name kitti_00_without_refine \
  --cache_path inference_cache/kitti_00_without_refine \
  --output_path viser_results \
  --window_size 10 \
  --overlap 5 \
  --no-depth-refine
```

## 7. 检查日志

运行日志会打印：

- 输入目录；
- 自然排序并采样后的图片数量；
- 第一张和最后一张图片文件名；
- 窗口大小、重叠帧数和预期窗口数；
- 是否开启回环和 layer-atomic 深度细化；
- 实际完成的缓存窗口数；
- SALAD 检测到的回环对数量；
- 最终结果目录。

推理、SALAD 和回环约束会复用同一份图片清单，`sample_interval` 只在自然排序完成后执行一次。

如果日志显示 `Detected loop pairs: 0`，这不是程序错误。程序会保留相邻窗口的 Sim(3) 变换并继续保存结果。

如果出现 `Window cache count mismatch`，表示后台推理没有生成全部窗口缓存。应查看该错误之前的 CUDA 或模型异常，不要继续使用不完整缓存做回环优化。

## 8. 结果与可视化

完整运行结果位于：

```text
viser_results/kitti_00
```

可视化命令：

```bash
python viser/visualizer_monst3r.py --data viser_results/kitti_00
```
