import argparse
import math
import torch
from pathlib import Path

from .loop_model import LoopDetector
from .utils.sim3loop import Sim3LoopOptimizer
from .utils.sim3utils import *

from utils.load_fn import load_and_preprocess_images
from pi3.models.pi3 import Pi3
from inference_engine import (
    StreamingWindowEngine,
    StreamingWindowEngineLC,
)
from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.utils.registration_confidence import (
    intersect_confidence_masks,
    select_top_confidence_mask,
    validate_confidence_keep_ratio,
)
from inference_engine.utils.geometry import (
    accumulate_sim3,
    closed_form_inverse_sim3,
)
from utils.image_paths import discover_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def get_args_parser():
    parser = argparse.ArgumentParser('Post process loop closure', add_help=False)
    parser.add_argument('--config_path', default=None, type=str, help='loop closure config')
    parser.add_argument('--data_path', type=str, help='sequence data path')
    parser.add_argument('--cache_path', default='./cache', type=str,
                        help='inference cache path')
    parser.add_argument('--output_path', default='./cache_lc', type=str,
                        help='loop closure cache path')
    parser.add_argument('--window_size', default=20, type=int, help='sliding window size')
    parser.add_argument('--overlap', default=5, type=int, help='sliding window overlap size')

    return parser


def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    return result


def identity_sim3_like(reference_sim3):
    _, rotation, translation = reference_sim3
    return (
        1.0,
        torch.eye(
            rotation.shape[-1],
            dtype=rotation.dtype,
            device=rotation.device,
        ),
        torch.zeros_like(translation),
    )


def accumulate_edges_from_identity(
        sequential_edges,
        reference_sim3,
):
    absolute_transforms = [identity_sim3_like(reference_sim3)]
    for edge in sequential_edges:
        absolute_transforms.append(
            accumulate_sim3(absolute_transforms[-1], edge)
        )
    return absolute_transforms


def build_local_loop_constraint(
        sim3_abs_a,
        sim3_abs_b,
        global_alignment_a,
        global_alignment_b,
):
    for context, transform in (
            ("sim3_abs_a", sim3_abs_a),
            ("sim3_abs_b", sim3_abs_b),
            ("global_alignment_a", global_alignment_a),
            ("global_alignment_b", global_alignment_b),
    ):
        validate_sim3(transform, context)

    global_correction = accumulate_sim3(
        global_alignment_b,
        closed_form_inverse_sim3(*global_alignment_a),
    )
    constraint = accumulate_sim3(
        closed_form_inverse_sim3(*sim3_abs_b),
        accumulate_sim3(
            global_correction,
            sim3_abs_a,
        ),
    )
    validate_sim3(constraint, "loop_constraint_ab")
    return constraint


def validate_sim3(transform, context):
    if not isinstance(transform, (tuple, list)) or len(transform) != 3:
        raise ValueError(
            f"{context} must contain scale, rotation, translation"
        )

    scale, rotation, translation = transform
    scale_tensor = torch.as_tensor(scale)
    rotation_tensor = torch.as_tensor(rotation)
    translation_tensor = torch.as_tensor(translation)

    if scale_tensor.numel() != 1:
        raise ValueError(f"{context} scale must be scalar")
    if tuple(rotation_tensor.shape) != (3, 3):
        raise ValueError(
            f"{context} rotation must have shape (3, 3)"
        )
    if translation_tensor.numel() != 3:
        raise ValueError(
            f"{context} translation must contain 3 values"
        )
    if not torch.isfinite(scale_tensor).all():
        raise ValueError(f"{context} scale is not finite")
    if not torch.isfinite(rotation_tensor).all():
        raise ValueError(f"{context} rotation is not finite")
    if not torch.isfinite(translation_tensor).all():
        raise ValueError(f"{context} translation is not finite")
    if float(scale_tensor.detach().cpu().item()) <= 0.0:
        raise ValueError(f"{context} scale must be positive")


def sim3_metrics(transform):
    scale, rotation, translation = transform
    scale_value = float(
        torch.as_tensor(scale).detach().cpu().item()
    )
    rotation_tensor = torch.as_tensor(
        rotation,
        dtype=torch.float64,
    )
    cosine = torch.clamp(
        (torch.trace(rotation_tensor) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    rotation_degrees = float(
        torch.rad2deg(torch.acos(cosine)).item()
    )
    translation_norm = float(
        torch.linalg.vector_norm(
            torch.as_tensor(
                translation,
                dtype=torch.float64,
            ).reshape(-1)
        ).item()
    )
    return {
        "scale": scale_value,
        "scale_log_abs": abs(math.log(scale_value)),
        "rotation_deg": rotation_degrees,
        "translation_norm": translation_norm,
    }


class LoopClosureEngine:
    def __init__(
            self,
            config,
            image_dir,
            output_dir,
            pi3_model,
            window_size,
            overlap,
            sample_interval=1,
            registration_top_confidence_ratio=0.3,
            image_paths=None,
    ):
        self.config = config

        self.pi3_model = pi3_model
        self.window_size = window_size
        self.overlap = overlap
        self.registration_top_confidence_ratio = (
            validate_confidence_keep_ratio(
                registration_top_confidence_ratio
            )
        )

        self.img_dir = image_dir
        self.img_list = image_paths
        self.sample_interval = sample_interval

        self.loop_detector = LoopDetector(
            image_dir=image_dir,
            sample_interval=sample_interval,
            output=output_dir,
            config=config,
            image_paths=image_paths,
        )

        self.chunk_indices = None
        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.loop_list = []  # e.g. [(1584, 139), ...]
        self.loop_candidates = []
        self.loop_optimizer = Sim3LoopOptimizer(config)
        self.loop_results = []
        self.loop_constraints = []
        self.loop_predict_list = []

    def _ensure_image_manifest(self):
        if self.img_list is None:
            self.img_list = discover_images(
                self.img_dir,
                sample_interval=self.sample_interval,
            )
            self.loop_detector.image_paths = self.img_list

        if len(self.img_list) == 0:
            raise ValueError(
                f"[DIR EMPTY] No images found in {self.img_dir}!"
            )

    def _build_chunk_indices(self):
        step = self.window_size - self.overlap
        if step <= 0:
            raise ValueError("window_size must be greater than overlap")

        chunk_indices = []
        for start_idx in range(0, len(self.img_list), step):
            end_idx = min(
                start_idx + self.window_size,
                len(self.img_list),
            )
            if (
                    start_idx > 0
                    and end_idx - start_idx <= self.overlap
            ):
                break
            chunk_indices.append((start_idx, end_idx))
        return chunk_indices

    def _validate_cache_count(self, raw_predictions):
        self.chunk_indices = self._build_chunk_indices()
        if len(raw_predictions) != len(self.chunk_indices):
            raise ValueError(
                "cache count does not match image manifest: "
                f"expected {len(self.chunk_indices)}, "
                f"got {len(raw_predictions)}"
            )

        for index, prediction in enumerate(raw_predictions):
            if "sim3_abs" not in prediction:
                raise ValueError(
                    f"cache {index} is missing sim3_abs"
                )
            if index > 0 and "sim3_edge" not in prediction:
                raise ValueError(
                    f"cache {index} is missing sim3_edge"
                )

    def get_loop_pairs(self):
        self.loop_detector.run()
        self.loop_detector.save_results()
        self.loop_candidates = list(
            self.loop_detector.loop_closures or []
        )
        self.loop_list = self.loop_detector.get_loop_list()
        del self.loop_detector
        torch.cuda.empty_cache()

    def process_single_chunk(self, range_1, range_2=None):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        images = load_and_preprocess_images(chunk_image_paths).to(device)

        # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.pi3_model(images)
        torch.cuda.empty_cache()

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().squeeze(0)

        # conf_thre = torch.quantile(predictions['conf'], self.top_conf_percentile, interpolation='nearest')
        # predictions['mask'] = predictions['conf'] >= conf_thre

        return predictions

    def process_loops(self, raw_predictions):
        if self.chunk_indices is None:
            self.chunk_indices = self._build_chunk_indices()

        print('Loop SIM(3) estimating...')
        candidates = (
            self.loop_candidates
            if self.loop_candidates
            else [
                (frame_a, frame_b, float("nan"))
                for frame_a, frame_b in self.loop_list
            ]
        )
        self.loop_results = []
        for frame_a, frame_b, similarity in candidates:
            mapped = process_loop_list(
                self.chunk_indices,
                [(frame_a, frame_b)],
                half_window=(
                    self.config['Model']['loop_chunk_size'] // 2
                ),
            )
            if not mapped:
                continue
            self.loop_results.append(
                (
                    *mapped[0],
                    frame_a,
                    frame_b,
                    float(similarity),
                )
            )
        self.loop_results = remove_duplicates(self.loop_results)
        self.loop_predict_list = []
        self.loop_constraints = []

        for item in self.loop_results:
            single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3])
            self.loop_predict_list.append((item, single_chunk_predictions))

        for item in self.loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]
            frame_a = item[0][4]
            frame_b = item[0][5]
            similarity = item[0][6]

            point_map_loop = item[1]['local_points'][:chunk_a_range[1] - chunk_a_range[0]]
            cam_pose_loop = item[1]['camera_poses'][:chunk_a_range[1] - chunk_a_range[0]]
            # conf_mask_loop = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]
            conf_map_loop = item[1]['conf'][:chunk_a_range[1] - chunk_a_range[0]]
            conf_mask_loop = select_top_confidence_mask(
                conf_map_loop,
                self.registration_top_confidence_ratio,
            )

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
            chunk_data_a = raw_predictions[chunk_idx_a]
            point_map_a = chunk_data_a['local_points'][chunk_a_rela_begin:chunk_a_rela_end]
            cam_pose_a = chunk_data_a['camera_poses'][chunk_a_rela_begin:chunk_a_rela_end]
            # conf_mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
            conf_map_a = chunk_data_a['conf'][chunk_a_rela_begin:chunk_a_rela_end]
            conf_mask_a = select_top_confidence_mask(
                conf_map_a,
                self.registration_top_confidence_ratio,
            )
            try:
                mutual_mask_a = intersect_confidence_masks(
                    conf_mask_loop,
                    conf_mask_a,
                    context=(
                        f"loop constraint {chunk_idx_a}->{chunk_idx_b} "
                        "side A"
                    ),
                )
            except ValueError as error:
                print(f"Skipping loop candidate: {error}")
                continue

            s_a, R_a, t_a = register_adjacent_windows(
                point_map_a,
                point_map_loop,
                cam_pose_a,
                cam_pose_loop,
                mutual_mask_a,
            )

            point_map_loop = item[1]['local_points'][-chunk_b_range[1] + chunk_b_range[0]:]
            cam_pose_loop = item[1]['camera_poses'][-chunk_b_range[1] + chunk_b_range[0]:]
            # conf_mask_loop = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]
            conf_map_loop = item[1]['conf'][-chunk_b_range[1] + chunk_b_range[0]:]
            conf_mask_loop = select_top_confidence_mask(
                conf_map_loop,
                self.registration_top_confidence_ratio,
            )

            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
            chunk_data_b = raw_predictions[chunk_idx_b]
            point_map_b = chunk_data_b['local_points'][chunk_b_rela_begin:chunk_b_rela_end]
            cam_pose_b = chunk_data_b['camera_poses'][chunk_b_rela_begin:chunk_b_rela_end]
            # conf_mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
            conf_map_b = chunk_data_b['conf'][chunk_b_rela_begin:chunk_b_rela_end]
            conf_mask_b = select_top_confidence_mask(
                conf_map_b,
                self.registration_top_confidence_ratio,
            )
            try:
                mutual_mask_b = intersect_confidence_masks(
                    conf_mask_loop,
                    conf_mask_b,
                    context=(
                        f"loop constraint {chunk_idx_a}->{chunk_idx_b} "
                        "side B"
                    ),
                )
            except ValueError as error:
                print(f"Skipping loop candidate: {error}")
                continue

            s_b, R_b, t_b = register_adjacent_windows(
                point_map_b,
                point_map_loop,
                cam_pose_b,
                cam_pose_loop,
                mutual_mask_b,
            )

            sim3_abs_a = raw_predictions[chunk_idx_a]["sim3_abs"]
            sim3_abs_b = raw_predictions[chunk_idx_b]["sim3_abs"]
            try:
                loop_constraint = build_local_loop_constraint(
                    sim3_abs_a,
                    sim3_abs_b,
                    (s_a, R_a, t_a),
                    (s_b, R_b, t_b),
                )
                initial_residual = accumulate_sim3(
                    loop_constraint,
                    accumulate_sim3(
                        closed_form_inverse_sim3(*sim3_abs_a),
                        sim3_abs_b,
                    ),
                )
                validate_sim3(
                    initial_residual,
                    "initial_loop_residual",
                )
            except ValueError as error:
                print(
                    "Skipping loop candidate "
                    f"{chunk_idx_a}->{chunk_idx_b}: {error}"
                )
                continue

            measurement_metrics = sim3_metrics(loop_constraint)
            residual_metrics = sim3_metrics(initial_residual)
            print(
                "Loop constraint: "
                f"frames={frame_a}->{frame_b}, "
                f"windows={chunk_idx_a}->{chunk_idx_b}, "
                f"similarity={similarity:.6f}, "
                "measurement_scale="
                f"{measurement_metrics['scale']:.6f}, "
                "measurement_rotation_deg="
                f"{measurement_metrics['rotation_deg']:.6f}, "
                "measurement_translation_norm="
                f"{measurement_metrics['translation_norm']:.6f}, "
                "initial_residual_scale_log_abs="
                f"{residual_metrics['scale_log_abs']:.6f}, "
                "initial_residual_rotation_deg="
                f"{residual_metrics['rotation_deg']:.6f}, "
                "initial_residual_translation_norm="
                f"{residual_metrics['translation_norm']:.6f}"
            )
            s_ab, R_ab, t_ab = loop_constraint
            self.loop_constraints.append(
                (
                    chunk_idx_a,
                    chunk_idx_b,
                    (s_ab, R_ab, t_ab),
                )
            )

        return self.loop_constraints

    def run(self, raw_predictions):
        print(f"Loading images from {self.img_dir}...")
        self._ensure_image_manifest()
        print(f"Found {len(self.img_list)} images")
        self._validate_cache_count(raw_predictions)

        initial_absolute = [
            prediction["sim3_abs"]
            for prediction in raw_predictions
        ]

        self.get_loop_pairs()
        if not self.loop_list:
            return initial_absolute

        loop_constraints = self.process_loops(raw_predictions)
        if not loop_constraints:
            return initial_absolute

        sequential_edges = [
            prediction["sim3_edge"]
            for prediction in raw_predictions[1:]
        ]
        print(
            "Loop graph: "
            f"sequential_constraints={len(sequential_edges)}, "
            f"loop_constraints={len(loop_constraints)}"
        )
        optimized_edges = self.loop_optimizer.optimize(
            sequential_edges,
            loop_constraints,
        )
        if len(optimized_edges) != len(sequential_edges):
            raise ValueError(
                "optimized edge count does not match sequential edge count"
            )

        return accumulate_edges_from_identity(
            optimized_edges,
            initial_absolute[0],
        )


if __name__ == '__main__':
    from .utils.config_utils import load_config

    args = get_args_parser()
    args = args.parse_args()
    pi3_model = Pi3.from_pretrained("yyfz233/Pi3").to(device)

    config = load_config(args.config_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_closure = LoopClosureEngine(
        config,
        args.data_path,
        output_dir / "loop_closures.txt",
        pi3_model,
        args.window_size,
        args.overlap,
    )

    cache_files = sorted(glob.glob(str(Path(args.cache_path) / 'window_cache_*.pt')),
                         key=lambda p: int(p.split('_')[-1].split('.')[0]))
    raw_predictions = [StreamingWindowEngine.parse_cache_file(cache_fname) for cache_fname in cache_files]
    optimized_absolute = loop_closure.run(raw_predictions)
    adjusted_predictions = (
        StreamingWindowEngineLC.apply_optimization_deltas(
            raw_predictions,
            optimized_absolute,
        )
    )

    for idx, output_prediction in enumerate(adjusted_predictions):
        torch.save(
            output_prediction,
            output_dir / f'window_cache_{idx}.pt',
        )
