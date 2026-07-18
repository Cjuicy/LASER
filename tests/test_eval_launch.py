from pathlib import Path

from eval_launch import get_args_parser


def test_eval_launch_exposes_unified_segmentation_arguments():
    parser = get_args_parser()
    args = parser.parse_args(["--model", "streaming_pi3"])
    assert args.segment_mode == "depth"
    assert args.normal_method == "cross"
    assert args.geometry_seg_profile == "baseline_params"
    assert args.model_ckpt is None
    assert args.diagnostic_max_temp_gib == 50
    assert args.diagnostic_warn_temp_gib == 40
    assert args.diagnostic_selected_interval_limit == 48


def test_eval_launch_comment_names_the_strict_three_method_workflow():
    source = Path(__file__).resolve().parents[1].joinpath("eval_launch.py").read_text()

    assert "strict three-method diagnostic workflow" in source
    assert "four-profile diagnostic workflow" not in source
