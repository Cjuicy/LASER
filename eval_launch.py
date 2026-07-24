from eval.pose_eval import eval_pose_estimation
from eval.depth_eval import eval_mono_depth_estimation
from pi3.models.pi3 import Pi3
from inference_engine import VanillaEngine
from loop_closure.methods import detect_loop_candidates
from pipeline.config import (
    LoopMethod,
    load_pipeline_config,
)
from pipeline.manifest import ImageManifest
from pipeline.runner import (
    build_default_window_engine,
    complete_reconstruction_payload,
    run_windows,
)
from functools import partial
from dataclasses import replace
import eval.misc as misc  # noqa
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import argparse
import json
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation launch', add_help=False)

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # switch mode for train / eval pose / eval depth
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')

    # for pose eval
    parser.add_argument('--pose_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--pose_eval_stride', default=1, type=int, help='stride for pose evaluation')
    parser.add_argument('--scene_graph_type', default='swinstride-5-noncyclic', type=str,
                        help='scene graph window size')
    parser.add_argument('--save_best_pose', action='store_true', default=False, help='save best pose')
    parser.add_argument('--n_iter', default=300, type=int, help='number of iterations for pose optimization')
    parser.add_argument('--save_pose_qualitative', action='store_true', default=False,
                        help='save qualitative pose results')
    parser.add_argument('--temporal_smoothing_weight', default=0.01, type=float,
                        help='temporal smoothing weight for pose optimization')
    parser.add_argument('--not_shared_focal', action='store_true', default=False,
                        help='use shared focal length for pose optimization')
    parser.add_argument('--use_gt_focal', action='store_true', default=False,
                        help='use ground truth focal length for pose optimization')
    parser.add_argument('--pose_schedule', default='linear', type=str, help='pose optimization schedule')

    parser.add_argument('--flow_loss_weight', default=0.01, type=float, help='flow loss weight for pose optimization')
    parser.add_argument('--flow_loss_fn', default='smooth_l1', type=str, help='flow loss type for pose optimization')
    parser.add_argument('--use_gt_mask', action='store_true', default=False,
                        help='use gt mask for pose optimization, for sintel/davis')
    parser.add_argument('--motion_mask_thre', default=0.35, type=float,
                        help='motion mask threshold for pose optimization')
    parser.add_argument('--sam2_mask_refine', action='store_true', default=False,
                        help='use sam2 mask refine for the motion for pose optimization')
    parser.add_argument('--flow_loss_start_epoch', default=0.1, type=float, help='start epoch for flow loss')
    parser.add_argument('--flow_loss_thre', default=20, type=float, help='threshold for flow loss')
    parser.add_argument('--pxl_thresh', default=50.0, type=float, help='threshold for flow loss')
    parser.add_argument('--depth_regularize_weight', default=0.0, type=float,
                        help='depth regularization weight for pose optimization')
    parser.add_argument('--translation_weight', default=1, type=float, help='translation weight for pose optimization')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode for pose evaluation')
    parser.add_argument('--full_seq', action='store_true', default=False, help='use full sequence for pose evaluation')
    parser.add_argument('--seq_list', nargs='+', default=None, help='list of sequences for pose evaluation')

    parser.add_argument('--eval_dataset', type=str, default='sintel',
                        choices=['davis', 'kitti', 'bonn', 'scannet', 'tum', 'nyu', 'sintel', 'kitti_odometry'],
                        help='choose dataset for pose evaluation')
    # model variant
    parser.add_argument('--model', type=str, required=True,
                        choices=['pi3', 'streaming_pi3', 'streaming_pi3_lc'],
                        help='choose model for pose evaluation')
    # checkpoint loading
    parser.add_argument('--ckpt_path', default=None, type=str, help='trained checkpoint for evaluation')
    parser.add_argument(
        '--pipeline_config',
        default='configs/pipeline/default.yaml',
        type=str,
        help='typed modular pipeline configuration',
    )

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='do not crop the image for monocular depth evaluation')

    # output dir
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path where to save the output")
    return parser


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = (
    torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


def build_streaming_eval_engine(delegate, config):
    return build_default_window_engine(config, delegate)


def _run_modular_evaluation(model, imgs, manifest, detect_loops):
    config = model.pipeline_config
    caches = run_windows(model, manifest, imgs, config)
    candidates = (
        detect_loop_candidates(
            config.loop.detection,
            manifest,
            Path(config.output.cache_dir) / "loop_candidates.txt",
        )
        if detect_loops and config.loop.enabled
        else ()
    )
    constraints = model.loop_strategy.build_constraints(
        caches,
        candidates,
    )
    solution = model.loop_strategy.optimize(caches, constraints)
    result = model.loop_strategy.aggregate(caches, solution)
    result = complete_reconstruction_payload(result, imgs)
    return {
        key: value.detach().cpu()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in result.payload.items()
    }


def inference_streaming_model(model, imgs, *args, **kwargs):
    manifest = ImageManifest(
        paths=tuple(Path(f"frame_{index:08d}") for index in range(len(imgs)))
    )
    return _run_modular_evaluation(
        model,
        imgs,
        manifest,
        detect_loops=False,
    )


def inference_streaming_model_lc(
        model,
        imgs,
        img_dir,
        image_paths=None,
        *args,
        **kwargs,
):
    if image_paths is None:
        raise ValueError(
            "streaming_pi3_lc evaluation requires the exact image_paths "
            "used to build imgs"
        )
    if len(image_paths) != len(imgs):
        raise ValueError(
            "image manifest length does not match evaluation tensor: "
            f"{len(image_paths)} != {len(imgs)}"
        )

    manifest = ImageManifest(
        paths=tuple(Path(path).resolve() for path in image_paths)
    )
    return _run_modular_evaluation(
        model,
        imgs,
        manifest,
        detect_loops=True,
    )


def pi3_main(args):
    print('Launching Pi3 eval')
    misc.init_distributed_mode(args)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = args.cudnn_benchmark

    delegate = Pi3.from_pretrained("yyfz233/Pi3").to(device)
    if args.model == 'pi3':
        model = VanillaEngine(delegate)
    else:
        config = load_pipeline_config(args.pipeline_config).config
        runtime_model = replace(
            config.model,
            inference_device=device,
            process_device="cpu",
            dtype=(
                "bfloat16"
                if dtype is torch.bfloat16
                else "float16"
            ),
        )
        runtime_loop = config.loop
        if args.model == 'streaming_pi3':
            runtime_loop = replace(
                runtime_loop,
                enabled=False,
                method=LoopMethod.TRADITIONAL,
            )
        runtime_config = replace(
            config,
            model=runtime_model,
            loop=runtime_loop,
            output=replace(
                config.output,
                cache_dir=str(Path(args.output_dir) / "inference_cache"),
            ),
        )
        model = build_streaming_eval_engine(delegate, runtime_config)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == 'streaming_pi3':
        infer_func = partial(inference_streaming_model, model)
    else:
        infer_func = partial(inference_streaming_model_lc, model)
    if args.mode == 'eval_pose':
        ate_mean, rpe_trans_mean, rpe_rot_mean, seq_attr, outfile_list, bug = eval_pose_estimation(
            args,
            infer_func,
            device,
            dtype,
            save_dir=args.output_dir,
            inverse_extrinsic=False
        )
        print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')
        result_dict = {
            'Seq Attributes': seq_attr,
            'ATE mean': ate_mean,
            'RPE trans mean': rpe_trans_mean,
            'RPE rot mean': rpe_rot_mean
        }
        with open(f'{args.output_dir}/{args.eval_dataset}_{args.mode}.json', 'w') as f:
            json.dump(result_dict, f, indent=2)
    if args.mode == 'eval_depth':
        eval_mono_depth_estimation(args, model, device, dtype)

    return 0


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    raise SystemExit(pi3_main(args))
