import torch
import torch.nn as nn

import time

from .streaming_window_engine import StreamingWindowEngine, STOP_SIGNAL
from .inference_utils import (
    register_adjacent_windows,
    estimate_pseudo_depth_and_intrinsics,
    unproject_depth_to_local_points,
    refine_depth_segments
)
from .utils.geometry import (
    apply_sim3_to_pose,
    accumulate_sim3,
    closed_form_inverse_sim3,
)
from .utils.registration_confidence import (
    intersect_confidence_masks,
    select_top_confidence_mask,
)


class StreamingWindowEngineLC(StreamingWindowEngine):
    def __init__(
            self,
            delegate: nn.Module,
            inference_device: str,
            dtype: torch.dtype,
            process_device: str = 'cpu',
            top_conf_percentile: float = 0.5,
            window_size: int = 20,
            overlap: int = 5,
            depth_refine=False,
            cache_root: str = './cache',
            segment_mode: str = "depth",
            normal_method: str = "cross",
            geometry_seg_profile: str = "baseline_params",
            registration_top_confidence_ratio: float = 0.3,
    ):
        super().__init__(
            delegate=delegate.to(inference_device),
            inference_device=inference_device,
            dtype=dtype,
            process_device=process_device,
            top_conf_percentile=top_conf_percentile,
            registration_top_confidence_ratio=registration_top_confidence_ratio,
            window_size=window_size,
            overlap=overlap,
            depth_refine=depth_refine,
            cache_root=cache_root,
            segment_mode=segment_mode,
            normal_method=normal_method,
            geometry_seg_profile=geometry_seg_profile,
        )

    def _registration_worker(self):
        ref_intrinsic = None
        tgt_sp_graph = None

        while True:
            item = self.registration_queue.get()
            if item is STOP_SIGNAL:
                return

            working_window, inference_duration = item
            t_start = time.perf_counter()

            for key in working_window.keys():
                if isinstance(working_window[key], torch.Tensor):
                    working_window[key] = working_window[key].squeeze(0)

            # camera pose registration
            tgt_mask = select_top_confidence_mask(
                working_window['conf'][:self.overlap],
                self.registration_top_confidence_ratio,
            )

            if self.prev_window_cache is not None:
                # fixed intrinsic enforce
                working_window['local_points'] = unproject_depth_to_local_points(
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )
                # mutual conf mask
                prev_mask = select_top_confidence_mask(
                    self.prev_window_cache['conf'][-self.overlap:],
                    self.registration_top_confidence_ratio,
                )
                conf_mask = intersect_confidence_masks(
                    prev_mask,
                    tgt_mask,
                    context="loop streaming registration",
                )

                # metric depth align
                prev_local_points = self.prev_window_cache['local_points'][-self.overlap:]
                cur_local_points = working_window['local_points'][:self.overlap]

                s_d, R, t = register_adjacent_windows(
                    prev_local_points,
                    cur_local_points,
                    self.prev_window_cache['camera_poses'][-self.overlap:],
                    working_window['camera_poses'][:self.overlap],
                    conf_mask
                )

                current_sim3_abs = s_d, R, t
                previous_sim3_abs = self.prev_window_cache['sim3_abs']
                working_window['sim3_abs'] = current_sim3_abs
                working_window['sim3_edge'] = accumulate_sim3(
                    closed_form_inverse_sim3(*previous_sim3_abs),
                    current_sim3_abs,
                )
                working_window['local_points'] = (
                    s_d * working_window['local_points']
                )
                working_window['camera_poses'] = apply_sim3_to_pose(
                    working_window['camera_poses'],
                    s_d,
                    R,
                    t,
                )

                if self.depth_refine:
                    tgt_pcd = working_window['local_points'].cpu().numpy()
                    tgt_sp_graph = self._build_segment_graph(
                        working_window['local_points'],
                        working_window['conf'],
                    )
                    working_window['scale_mask'] = refine_depth_segments(
                        self.prev_window_cache['local_points'].cpu().numpy(),
                        tgt_pcd,
                        self.anchor_sp_graph,
                        tgt_sp_graph,
                        self.overlap
                    )
                    working_window['local_points'] = (
                        working_window['scale_mask']
                        * working_window['local_points']
                    )
            else:
                _, intrinsic_ = estimate_pseudo_depth_and_intrinsics(working_window['local_points'])
                ref_intrinsic = intrinsic_[0]
                working_window['local_points'] = unproject_depth_to_local_points(
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )
                working_window['sim3_abs'] = (
                    1.0,
                    torch.eye(3, device=self.process_device),
                    torch.zeros(3, device=self.process_device)
                )

                if self.depth_refine:
                    tgt_sp_graph = self._build_segment_graph(
                        working_window['local_points'],
                        working_window['conf'],
                    )

            self._update_cache(working_window, tgt_sp_graph)
            self._save_cache()

            reg_duration = time.perf_counter() - t_start
            total_process_time = inference_duration + reg_duration
            self.latencies.append(total_process_time)

    @staticmethod
    def apply_optimization_deltas(
            parsed_caches,
            optimized_abs=None,
    ):
        if optimized_abs is None:
            optimized_abs = [
                cache['sim3_abs']
                for cache in parsed_caches
            ]
        if len(parsed_caches) != len(optimized_abs):
            raise ValueError(
                "optimized transform count does not match cache count"
            )

        adjusted_caches = []
        for index, (cache, optimized) in enumerate(
                zip(parsed_caches, optimized_abs)):
            original_inverse = closed_form_inverse_sim3(
                *cache['sim3_abs']
            )
            delta_scale, delta_rotation, delta_translation = (
                accumulate_sim3(optimized, original_inverse)
            )

            adjusted = dict(cache)
            adjusted['local_points'] = (
                delta_scale * cache['local_points']
            )
            adjusted['camera_poses'] = apply_sim3_to_pose(
                cache['camera_poses'],
                delta_scale,
                delta_rotation,
                delta_translation,
            )
            adjusted['sim3_abs'] = optimized
            if index == 0:
                adjusted.pop('sim3_edge', None)
            else:
                adjusted['sim3_edge'] = accumulate_sim3(
                    closed_form_inverse_sim3(
                        *optimized_abs[index - 1]
                    ),
                    optimized,
                )
            adjusted_caches.append(adjusted)

        return adjusted_caches

    @staticmethod
    def aggregate_caches(parsed_caches, optimized_abs=None):
        adjusted_caches = (
            StreamingWindowEngineLC.apply_optimization_deltas(
                parsed_caches,
                optimized_abs,
            )
        )
        return StreamingWindowEngine.aggregate_caches(adjusted_caches)
