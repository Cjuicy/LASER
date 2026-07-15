from eval_launch import get_args_parser


def test_eval_launch_exposes_unified_segmentation_arguments():
    parser = get_args_parser()
    args = parser.parse_args(["--model", "streaming_pi3"])
    assert args.segment_mode == "depth"
    assert args.normal_method == "cross"
    assert args.geometry_seg_profile == "baseline_params"
    assert args.model_ckpt is None
