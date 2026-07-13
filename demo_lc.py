"""
读取图像
  → 滑动窗口推理
  → 读取窗口缓存
  → 回环检测与位姿修正
  → 合并窗口结果
  → 保存可视化结果
"""
import torch

from pi3.models.pi3 import Pi3
from inference_engine import StreamingWindowEngineLC
from vggt.utils.load_fn import load_and_preprocess_images
from eval.save_func import save_for_viser
# 📒 导入回环模块
from loop_closure.loop_closure import LoopClosureEngine
from loop_closure.utils.config_utils import load_config

import os
import argparse
from tqdm import tqdm
import glob
import shutil
from pathlib import Path

from utils.image_paths import discover_images, natural_sort_key

# 初始化与设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


# 函数说明
def get_args_parser():
    parser = argparse.ArgumentParser('Depth metric evaluation', add_help=False)
    parser.add_argument('--config_path', default=None, type=str, help='loop closure config')
    parser.add_argument('--model_ckpt', default='weights/model.safetensors', type=str,
                        help='local checkpoint to load model')
    parser.add_argument('--data_path', type=str, help='sequence data path')
    parser.add_argument('--scene_name', default=None, type=str, help='scene_name')
    parser.add_argument('--cache_path', default='./inference_cache', type=str,
                        help='output inference cache')
    parser.add_argument('--output_path', default='./viser_results', type=str,
                        help='output visualization results')
    parser.add_argument('--sample_interval', default=1, type=int, help='sequence sample interval')
    parser.add_argument('--window_size', default=10, type=int, help='sliding window size')
    parser.add_argument('--overlap', default=5, type=int, help='sliding window overlap size')
    parser.add_argument('--depth_refine', action='store_true', help='enable depth refine')

    return parser


# 加载模型
def load_model(args):
    # model
    if not os.path.isfile(args.model_ckpt):
        raise FileNotFoundError(f'Model checkpoint not found: {args.model_ckpt}')

    model = Pi3().to(device)
    print('Loading checkpoint: ', args.model_ckpt)
    if args.model_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        ckpt = load_file(args.model_ckpt)
    else:
        ckpt = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    print(model.load_state_dict(ckpt, strict=True))
    del ckpt

    return model, StreamingWindowEngineLC(
        model,
        inference_device=device,
        dtype=dtype,
        window_size=args.window_size,
        overlap=args.overlap,
        cache_root=args.cache_path,
        depth_refine=args.depth_refine
    )

# 手动将列表划分为多个重叠窗口
def sliding_window(lst, window_size, overlap):
    step = window_size - overlap
    assert step > 0

    windows = []
    for i in range(0, len(lst), step):
        window = lst[i: i + window_size]
        if len(window) > overlap:
            windows.append(window)
        else:
            break
    return windows


# 执行滑动窗口推理，并统计时间和显存
def run_model(image_names):
    image_name_windows = model.img_sliding_window(image_names)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    # 1️⃣ 初始化推理
    model.begin()

    start_ev.record()
    # 2️⃣ 逐窗口推理
    for sample in tqdm(image_name_windows, 'Window inference'):
        imgs = load_and_preprocess_images(sample).to(device)
        model(imgs)
    # 3️⃣ 结束推理
    model.end()
    end_ev.record()

    # save_dict = model.parse_inference_cache_summary()
    # for key in save_dict.keys():
    #     if isinstance(save_dict[key], torch.Tensor):
    #         save_dict[key] = save_dict[key].cpu().numpy().squeeze(0)
    #
    # save_for_viser(save_dict, scene_name, output_path, inverse_extrinsic=False)

    torch.cuda.synchronize()  # make sure the event timestamps are set
    # 4️⃣ 性能统计
    duration = start_ev.elapsed_time(end_ev)
    gpu_mem_usage = torch.cuda.max_memory_allocated()

    summary_text = f"""
    Summary:
        Inference sec: {duration / 1000}
        Peak GPU memory usage (GB): {gpu_mem_usage / (1024 ** 3)} 
    """
    print(summary_text)

    # save_cache_to_viser(model.cache_dir, scene_name, output_path, overlap)

# 读取场景图像并调用模型推理
def run_dynamic_scene(args):
    data_path = args.data_path
    scene_name = data_path.split('/')[-2] if args.scene_name is None else args.scene_name

    # 获取图像文件并按编号自然排序，再按间隔采样
    img_names = discover_images(data_path, args.sample_interval)
    print(f'Found {len(img_names)} images.')
    run_model(img_names)
    return scene_name

# 主程序解析
if __name__ == "__main__":
    # 1️⃣ 解析参数并加载模型
    args = get_args_parser()
    args = args.parse_args()
    pi3_model, model = load_model(args)

    model.eval()
    scene_name = run_dynamic_scene(args)

    # 📒2️⃣ 创建回环检测引擎
    config = load_config(args.config_path)
    cache_path = Path(args.cache_path)
    cache_path_lc = cache_path.parent / f'{cache_path.name}_lc'
    lc_engine = LoopClosureEngine(
        config,
        args.data_path,
        cache_path_lc,
        pi3_model,
        args.window_size,
        args.overlap,
        args.sample_interval
    )

    # 3️⃣ 读取原始窗口缓存
    cache_files = sorted(glob.glob(str(model.temp_cache_dir / 'window_cache_*.pt')),
                         key=lambda p: int(p.split('_')[-1].split('.')[0]))
    raw_predictions = [StreamingWindowEngineLC.parse_cache_file(cache_fname) for cache_fname in cache_files]

    # 📒4️⃣ 执行回环修正（⚠️回环修正具体完成的部分）
    sim3_list_lc = lc_engine.run(raw_predictions)
    sim3_list_lc.insert(0, raw_predictions[0]['sim3'])

    os.makedirs(str(cache_path_lc), exist_ok=True)
    # 5️⃣ 保存修正后的缓存
    for idx, (pred, sim3_lc) in enumerate(zip(raw_predictions, sim3_list_lc)):
        # s, R, t = sim3_lc
        # pred['sim3'] = s, torch.from_numpy(R.astype(np.float32)), torch.from_numpy(t.astype(np.float32))
        pred['sim3'] = sim3_lc
        torch.save(pred, str(cache_path_lc / f'window_cache_{idx}.pt'))

    cache_files_lc = sorted(glob.glob(str(cache_path_lc / 'window_cache_*.pt')),
                            key=lambda p: int(p.split('_')[-1].split('.')[0]))
    # 6️⃣ 重新读取并合并窗口（与demo不同，对齐后直接输出，这里是对于每个窗口点云改进后，会回环检测再输出）
    parsed_caches = [StreamingWindowEngineLC.parse_cache_file(cache_files_lc[0])]
    for cache_fname in cache_files_lc[1:]:
        parsed_caches.append(StreamingWindowEngineLC.parse_cache_file(cache_fname, overlap=args.overlap))

    ret_dict = StreamingWindowEngineLC._post_process_pred(StreamingWindowEngineLC.aggregate_caches(parsed_caches))
    # 7️⃣ 清理并保存结果
    shutil.rmtree(cache_path_lc)
    for key in ret_dict.keys():
        if isinstance(ret_dict[key], torch.Tensor):
            ret_dict[key] = ret_dict[key].cpu().numpy().squeeze(0)

    save_for_viser(ret_dict, scene_name, args.output_path, inverse_extrinsic=False)
