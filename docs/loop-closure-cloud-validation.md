# 回环检测改进版：云端克隆与验证

本文档用于验证分支 `codex/loop-closure-corrected-pipeline`。该分支基于
`codex/unified-segmentation-methods`，只修改回环检测及其前后数据链路：

- 推理、SALAD 和联合 Pi3 使用同一份自然排序、统一采样后的图像清单；
- 回环链路的配准点统一保留最高置信度 `30%`，即内部使用 `0.7` 分位点；
- 第一阶段立即应用窗口 Sim(3) 和已有的深度分段尺度；
- 优化器只接收相邻边与有效回环约束；
- 最终聚合只应用回环优化增量，不重复应用分割尺度。

分割方法、区域匹配、锚点传播和 Sim(3) 优化器数学实现没有改变。

## 1. 克隆指定分支

新环境直接执行：

```bash
git clone \
  --branch codex/loop-closure-corrected-pipeline \
  --single-branch \
  --recursive \
  https://github.com/Cjuicy/LASER.git
cd LASER
git submodule update --init --recursive
```

已有仓库执行：

```bash
git fetch origin
git switch codex/loop-closure-corrected-pipeline
git pull --ff-only origin codex/loop-closure-corrected-pipeline
git submodule update --init --recursive
```

确认当前版本：

```bash
git branch --show-current
git log -1 --oneline
```

## 2. 安装依赖和编译扩展

推荐使用 Python 3.11：

```bash
conda create -n laser-loop-closure -y python=3.11
conda activate laser-loop-closure

pip install -r requirements.txt
pip install -e viser
pip install faiss-gpu-cu12

python setup.py build_ext --inplace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

如果云端 CUDA/PyTorch 版本不适配 `faiss-gpu-cu12`，应安装与该环境对应的
FAISS GPU 包；仅做 CPU 单元测试时可以使用 `faiss-cpu`。

## 3. 准备权重

回环运行至少需要：

```text
weights/model.safetensors
weights/dino_salad.ckpt
```

`weights/model.safetensors` 是 Pi3 本地权重，请上传或挂载到该路径；也可以通过
`--model_ckpt /absolute/path/to/model.safetensors` 指定其他位置。

SALAD 权重可以执行：

```bash
bash scripts/download_weights.sh
```

该脚本还会下载 DINO 权重。运行前建议确认：

```bash
python -c "import os, torch; print('cuda=', torch.cuda.is_available()); print('pi3=', os.path.isfile('weights/model.safetensors')); print('salad=', os.path.isfile('weights/dino_salad.ckpt'))"
```

三个结果中的 `pi3` 和 `salad` 必须为 `True`，真实回环推理还应确认
`cuda=True`。

## 4. 代码回归验证

先构建两个 Cython 扩展，再运行完整测试：

```bash
python setup.py build_ext --inplace
pytest -q
```

还可以单独运行回环主线测试：

```bash
pytest -q \
  tests/test_loop_image_manifest.py \
  tests/test_registration_confidence.py \
  tests/test_streaming_window_engine_lc_pipeline.py \
  tests/test_loop_closure_pipeline.py \
  tests/test_demo_lc.py
```

全部测试必须无失败。

## 5. 真实序列运行

把 `/path/to/sequence` 替换为待测图像目录：

```bash
python demo_lc.py \
  --config_path configs/loop_config.yaml \
  --model_ckpt weights/model.safetensors \
  --data_path /path/to/sequence \
  --cache_path ./inference_cache \
  --output_path ./viser_results \
  --sample_interval 1 \
  --window_size 10 \
  --overlap 5 \
  --depth_refine \
  --segment_mode depth \
  --registration_top_confidence_ratio 0.3
```

`0.3` 表示保留最高置信度 30%，不是把置信度值与 `0.3` 做绝对阈值比较。
若分位点处存在相同置信度，实际保留比例可以略高于 30%。

正式实验可继续使用已统一的三种分割模式：

- `--segment_mode depth`
- `--segment_mode geometry --normal_method cross --geometry_seg_profile baseline_params`
- `--segment_mode layer_atomic`

本次回环验证建议先固定为 `depth`，避免同时改变实验变量。

## 6. 无回环对照

复制配置，并把 SALAD 相似度阈值提高到 `1.1`，以强制不生成有效回环：

```bash
cp configs/loop_config.yaml /tmp/loop_config.no_loop.yaml
python -c "import yaml; p='/tmp/loop_config.no_loop.yaml'; c=yaml.safe_load(open(p)); c['Loop']['SALAD']['similarity_threshold']=1.1; open(p,'w').write(yaml.safe_dump(c, sort_keys=False))"
```

然后只替换配置和输出目录，其他参数保持完全一致：

```bash
python demo_lc.py \
  --config_path /tmp/loop_config.no_loop.yaml \
  --model_ckpt weights/model.safetensors \
  --data_path /path/to/sequence \
  --cache_path ./inference_cache_no_loop \
  --output_path ./viser_results_no_loop \
  --sample_interval 1 \
  --window_size 10 \
  --overlap 5 \
  --depth_refine \
  --segment_mode depth \
  --registration_top_confidence_ratio 0.3
```

无回环对照与有效回环实验必须保持以下条件一致：

1. 图像目录和文件内容；
2. `sample_interval`、`window_size` 和 `overlap`；
3. Pi3、SALAD 权重；
4. 分割模式及其参数；
5. 配准置信度比例；
6. CUDA、PyTorch、随机种子和其他运行环境。

两次实验只允许改变回环是否被接受，以及各自的缓存和结果目录。

## 7. 成功判定

满足以下条件即可确认云端验证通过：

1. `python setup.py build_ext --inplace` 退出码为 0；
2. `pytest -q` 无失败；
3. 日志中的图像数量与自然排序、采样后的清单一致；
4. 启动日志显示预期分割模式，运行命令明确传入回环配准比例 `0.3`；
5. 无有效回环时不进入 Sim(3) 优化，结果仍保留第一阶段校正；
6. 有效回环时出现 `Loop SIM(3) estimating...` 和优化器迭代日志；
7. 两次运行分别写入独立缓存和结果目录，没有交叉复用缓存。
