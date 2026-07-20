"""
解析命令行参数
    ↓
load_model()
加载 Pi3 模型并创建流式推理引擎
    ↓
model.eval()
切换到推理模式
    ↓
run_dynamic_scene()
读取图像路径
    ↓
run_model()
滑动窗口推理并保存结果
"""

# 导入模块与全局配置
# 1️⃣ 导入模型和推理工具
import torch

from pi3.models.pi3 import Pi3
from inference_engine import StreamingWindowEngine
from utils.load_fn import load_and_preprocess_images
from eval.save_func import save_for_viser

# 2️⃣ 导入通用工具
import os
import argparse
import re
from tqdm import tqdm

# 3️⃣ 设置推理设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
# 4️⃣ 设置推理精度
if device == "cuda":
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
else:
    dtype = torch.float32


# 定义命令行参数
def get_args_parser():
    # 1️⃣ 创建解析器
    parser = argparse.ArgumentParser('Streaming Pi3 Demo', add_help=False)
    parser.add_argument('--model_ckpt', default='weights/model.safetensors', type=str,
                        help='local checkpoint to load model')                                     # 模型相关参数
    parser.add_argument('--data_path', type=str, help='sequence data path')                         # 输入数据参数
    parser.add_argument('--scene_name', default=None, type=str, help='scene_name')
    parser.add_argument('--cache_path', default='./inference_cache', type=str,                      # 缓存与输出路径
                        help='output inference cache')
    parser.add_argument('--output_path', default='./viser_results', type=str,
                        help='output visualization results')
    parser.add_argument('--sample_interval', default=1, type=int, help='sequence sample interval')  # 图像采样参数
    parser.add_argument('--window_size', default=10, type=int, help='sliding window size')          # 滑动窗口参数
    parser.add_argument('--overlap', default=5, type=int, help='sliding window overlap size')
    parser.add_argument('--depth_refine', action='store_true', help='enable depth refine')          # 深度优化开关
    parser.add_argument('--anchor_propagation', default=None,
                        choices=('none', 'legacy_iou', 'hart'),
                        help='anchor propagation strategy; omitted preserves --depth_refine behavior')
    parser.add_argument('--anchor_min_pixels', default=64, type=int,
                        help='minimum mutual high-confidence pixels for a HART anchor')
    parser.add_argument('--scale_consistency_thresh', default=0.05, type=float,
                        help='complete-link log-scale consistency threshold for HART')
    parser.add_argument('--segment_mode', default='depth',
                        choices=('depth', 'geometry', 'layer_atomic', 'layer_atomic_split'),
                        help='segmentation method used by depth refinement')
    parser.add_argument('--normal_method', default='cross', choices=('cross', 'sobel'),
                        help='surface-normal estimator for geometry segmentation')
    parser.add_argument('--geometry_seg_profile', default='baseline_params',
                        choices=('baseline_params', 'legacy'),
                        help='Felzenszwalb profile for geometry segmentation')
    parser.add_argument('--split_score_thresh', default=0.10, type=float,
                        help='acceptance threshold for post-merge normal splitting')
    parser.add_argument('--split_aux_confirmation', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='confirm normal splits with RGB or normalized 3D gaps')

    return parser       # 返回解析器


def natural_sort_key(path):
    """按文件名中的数字大小生成自然排序键。"""
    parts = re.split(r'(\d+)', os.path.basename(path).lower())
    return [(1, int(part)) if part.isdigit() else (0, part) for part in parts]


def discover_images(data_path, sample_interval):
    """筛选并自然排序图片，然后按指定间隔采样。"""
    image_names = [
        os.path.join(data_path, name)
        for name in os.listdir(data_path)
        if name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return sorted(image_names, key=natural_sort_key)[::sample_interval]


# 加载模型并创建推理引擎
def load_model(args):
    # model
    # 1️⃣ 检查本地权重是否存在
    if not os.path.isfile(args.model_ckpt):
        raise FileNotFoundError(f'Model checkpoint not found: {args.model_ckpt}')

    # 2️⃣ 创建空模型并移动到设备
    model = Pi3().to(device)
    # 3️⃣ 输出权重路径
    print('Loading checkpoint: ', args.model_ckpt)
    # 4️⃣ 加载 Safetensors 权重
    if args.model_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        ckpt = load_file(args.model_ckpt)
    else:
        # 5️⃣ 加载普通 PyTorch 权重
        ckpt = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    # 6️⃣ 把权重载入模型
    print(model.load_state_dict(ckpt, strict=True))
    # 7️⃣ 释放权重字典
    del ckpt

    # 8️⃣ 创建流式推理引擎
    return StreamingWindowEngine(
        model,
        inference_device=device,
        dtype=dtype,
        window_size=args.window_size,
        overlap=args.overlap,
        cache_root=args.cache_path,
        depth_refine=args.depth_refine,
        top_conf_percentile=0.3,
        segment_mode=args.segment_mode,
        normal_method=args.normal_method,
        geometry_seg_profile=args.geometry_seg_profile,
        split_score_thresh=args.split_score_thresh,
        split_aux_confirmation=args.split_aux_confirmation,
        anchor_propagation=args.anchor_propagation,
        anchor_min_pixels=args.anchor_min_pixels,
        scale_consistency_thresh=args.scale_consistency_thresh,
    )

# 执行滑动窗口推理
def run_model(image_names, scene_name, output_path):
    # 1️⃣ 将图像切分成滑动窗口
    image_name_windows = model.img_sliding_window(image_names)

    # 2️⃣ 创建 CUDA 计时事件
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    # 3️⃣ 初始化推理过程
    model.begin()

    # 4️⃣ 记录开始时间
    start_ev.record()
    # 5️⃣ 遍历滑动窗口
    for sample in tqdm(image_name_windows, 'Window inference'):
        # 6️⃣ 加载并预处理图片
        imgs = load_and_preprocess_images(sample).to(device)
        # 7️⃣ 执行当前窗口推理
        model(imgs)
    # 8️⃣ 完成整个序列推理
    model.end()
    # 9️⃣ 记录结束时间
    end_ev.record()
    duration = start_ev.elapsed_time(end_ev)

    # 1️⃣0️⃣ 读取推理结果
    save_dict = model.parse_inference_cache_summary()
    # 1️⃣1️⃣ Tensor 转换为 NumPy
    for key in save_dict.keys():
        if isinstance(save_dict[key], torch.Tensor):
            save_dict[key] = save_dict[key].cpu().numpy().squeeze(0)

    # 1️⃣2️⃣ 保存 Viser 可视化结果
    save_for_viser(save_dict, scene_name, output_path, inverse_extrinsic=False)

    # 1️⃣3️⃣ 等待 GPU 完成任务
    torch.cuda.synchronize()  # make sure the event timestamps are set
    # 1️⃣4️⃣ 获取峰值显存
    gpu_mem_usage = torch.cuda.max_memory_allocated()

    # 1️⃣5️⃣ 打印统计信息
    summary_text = f"""
    Summary:
        Inference sec: {duration / 1000}
        Peak GPU memory usage (GB): {gpu_mem_usage / (1024 ** 3)}
    """
    print(summary_text)

    # save_cache_to_viser(model.cache_dir, scene_name, output_path, overlap)

# 读取场景图像
def run_dynamic_scene(args):
    # 1️⃣ 获取输入目录
    data_path = args.data_path
    # 2️⃣ 确定场景名
    scene_name = data_path.split('/')[-1] if args.scene_name is None else args.scene_name

    # 3️⃣ 筛选图片并按编号自然排序，再按间隔采样
    img_names = discover_images(data_path, args.sample_interval)
    # 5️⃣ 打印图片数量
    print(f'Found {len(img_names)} images.')
    # 6️⃣调用模型推理
    run_model(img_names, scene_name, args.output_path)

# 主程序入口
if __name__ == "__main__":
    # 1️⃣ 创建参数解析器
    args = get_args_parser()
    # 2️⃣ 解析命令行参数
    args = args.parse_args()
    # 3️⃣ 加载模型
    model = load_model(args)

    # 4️⃣ 切换到评估模式
    model.eval()
    # 5️⃣ 执行场景推理
    run_dynamic_scene(args)
