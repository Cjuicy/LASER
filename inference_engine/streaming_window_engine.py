"""
1. images 被拆成多个重叠窗口
2. begin() 启动两个线程
3. forward() 把所有窗口送入推理队列
4. GPU 线程执行模型预测
5. 预测结果移到 CPU
6. CPU 线程配准相邻窗口
7. 每个窗口保存到磁盘
8. end() 等待全部窗口完成
9. parse_inference_cache_summary() 删除重复帧并拼接
10. 根据相机位姿生成全局点云

输入图像窗口
    ↓
推理线程：神经网络预测
    ↓
registration_queue
    ↓
配准线程：窗口对齐、尺度统一、深度优化
    ↓
磁盘缓存 window_cache_x.pt
    ↓
汇总所有窗口
    ↓
完整序列的位姿、局部点云、全局点云
"""
# 导入模块
import torch
import torch.nn as nn

import os
import threading
import queue
import pathlib
import gc
import tempfile
import shutil
import glob
from collections import defaultdict
import time
import numpy as np

# 项目内部依赖
from . import VanillaEngine
# 继承的基础推理引擎
from .inference_utils import (
    dict_to_device,
    register_adjacent_windows,
    estimate_pseudo_depth_and_intrinsics,
    unproject_depth_to_local_points,
    apply_sim3_to_pose,
    make_sp_graph,
    refine_depth_segments,
    sliding_window_t,
    sliding_window_l
)
from .utils.geometry import homogenize_points
from .utils.lsa import (
    GEOMETRY_SEGMENTATION_PROFILES,
    NORMAL_METHODS,
    SEGMENT_MODES,
    get_felzenszwalb_params,
)

# 线程停止标志
STOP_SIGNAL = object()

# 类定义
class StreamingWindowEngine(VanillaEngine):
    def __init__(
            self,
            delegate: nn.Module,
            inference_device: str,
            dtype: torch.dtype,
            intermediate_device: str = 'cuda',
            process_device: str = 'cpu',
            top_conf_percentile: float = 0.5,
            window_size: int = 20,
            overlap: int = 5,
            depth_refine=True,
            cache_root: str = './cache',
            benchmark_latency=True,
            segment_mode: str = "depth",
            normal_method: str = "cross",
            geometry_seg_profile: str = "baseline_params",
            diagnostic_sink=None,
            diagnostic_run_id: str | None = None,
            diagnostic_sequence_id: str | None = None,
            diagnostic_pass: int = 0,
            cache_policy: str = "full",
    ):
        if cache_policy not in ("full", "metrics-only"):
            raise ValueError("cache_policy must be 'full' or 'metrics-only'")
        if diagnostic_sink is not None:
            if not diagnostic_run_id:
                raise ValueError("diagnostic_run_id is required when diagnostics are enabled")
            if not diagnostic_sequence_id:
                raise ValueError("diagnostic_sequence_id is required when diagnostics are enabled")
            if diagnostic_pass not in (1, 2):
                raise ValueError("diagnostic_pass must be 1 or 2 when diagnostics are enabled")
        if segment_mode not in SEGMENT_MODES:
            raise ValueError(
                f"Unknown segment_mode: {segment_mode!r}; expected one of {SEGMENT_MODES}."
            )
        if segment_mode != "depth" and not depth_refine:
            raise ValueError(
                f"segment_mode={segment_mode!r} requires depth_refine=True."
            )
        if segment_mode == "geometry":
            if normal_method not in NORMAL_METHODS:
                raise ValueError(
                    f"Unknown normal_method: {normal_method!r}; expected one of {NORMAL_METHODS}."
                )
            if geometry_seg_profile not in GEOMETRY_SEGMENTATION_PROFILES:
                raise ValueError(
                    "Unknown geometry_seg_profile: "
                    f"{geometry_seg_profile!r}; expected one of "
                    f"{tuple(GEOMETRY_SEGMENTATION_PROFILES)}."
                )

        # 1️⃣ 模型初始化
        super().__init__(
            delegate=delegate.to(inference_device)
        )
        # 2️⃣ 滑动窗口参数
        self.window_size = window_size
        self.overlap = overlap
        self.intermediate_device = intermediate_device
        # 3️⃣ 置信度阈值参数
        self.top_conf_percentile = 1 - top_conf_percentile if top_conf_percentile is not None else 0.0

        # 4️⃣ 设备和数据类型
        self.inference_device = inference_device
        self.process_device = process_device
        self.dtype = dtype
        # 5️⃣ 深度细化
        self.depth_refine = depth_refine
        self.segment_mode = segment_mode
        self.normal_method = normal_method
        self.geometry_seg_profile = geometry_seg_profile
        self.felzenszwalb_params = get_felzenszwalb_params(
            segment_mode,
            geometry_seg_profile,
        )
        self.diagnostic_sink = diagnostic_sink
        self.diagnostic_run_id = diagnostic_run_id
        self.diagnostic_sequence_id = diagnostic_sequence_id
        self.diagnostic_pass = diagnostic_pass
        self.cache_policy = cache_policy

        segmentation_details = f"mode={segment_mode}"
        if segment_mode == "geometry":
            segmentation_details += (
                f", profile={geometry_seg_profile}, normal={normal_method}"
            )
        print(
            "[segmentation] "
            f"{segmentation_details}, "
            f"scale={self.felzenszwalb_params['seg_scale']}, "
            f"sigma={self.felzenszwalb_params['seg_sigma']}, "
            f"min_size={self.felzenszwalb_params['seg_min_size']}"
        )

        # 6️⃣ 缓存目录
        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = cache_root
        self.temp_cache_dir = None
        self.cache_id = 0
        # 7️⃣ 两个线程队列
        self.inference_queue = queue.Queue()
        self.registration_queue = queue.Queue()

        # 8️⃣ 相邻窗口状态
        self.prev_window_cache = None
        self.anchor_sp_graph = None

        # 9️⃣ 线程对象和运行状态
        self._inference_thread = None
        self._registration_thread = None

        self.running = False

        # 1️⃣0️⃣ 延迟统计
        self.benchmark_latency = benchmark_latency
        self.latencies = []
        self.warmup_steps = 2

    # 修改缓存根目录
    def set_cache_dir(self, cache_dir):
        if self.running:
            raise RuntimeError('Cannot change cache directory while running')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    # 开启或关闭深度细化
    def set_depth_refine(self, flag):
        if self.running:
            raise RuntimeError('Cannot change depth refinement mode while running')
        self.depth_refine = flag

    def _diagnostic_context(self):
        if self.diagnostic_sink is None:
            return None
        from .diagnostics.schema import DiagnosticContext

        config_id = self.segment_mode
        if self.segment_mode == "geometry":
            config_id = (
                "geometry_baseline"
                if self.geometry_seg_profile == "baseline_params"
                else "geometry_legacy_reference"
            )
        return DiagnosticContext(
            run_id=self.diagnostic_run_id,
            config_id=config_id,
            sequence_id=self.diagnostic_sequence_id,
            pass_id=self.diagnostic_pass,
            window_id=self.cache_id,
            frame_start=self.cache_id * (self.window_size - self.overlap),
        )

    def _build_segment_graph(self, local_points, conf, images=None):
        kwargs = {
            "conf_map": conf.cpu().numpy(),
            "top_conf_percentile": self.top_conf_percentile,
            "segment_mode": self.segment_mode,
            "normal_method": self.normal_method,
            "geometry_seg_profile": self.geometry_seg_profile,
        }
        context = self._diagnostic_context()
        if self.diagnostic_sink is not None:
            kwargs.update(
                diagnostic_sink=self.diagnostic_sink,
                diagnostic_context=context,
            )
        graph = make_sp_graph(
            local_points.cpu().numpy(),
            **kwargs,
        )
        if self.diagnostic_sink is not None and images is not None:
            image_array = images.cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
            self.diagnostic_sink.emit_inputs(
                context,
                image_array,
                local_points.cpu().numpy(),
                conf.cpu().numpy(),
            )
        return graph

    # 把当前已经完成配准的窗口写入磁盘
    def _save_cache(self):
        destination = self.temp_cache_dir / f'window_cache_{self.cache_id}.pt'
        if self.cache_policy == "full":
            torch.save(self.prev_window_cache, destination)
        else:
            shard = {
                "camera_poses": self.prev_window_cache["camera_poses"],
                "window_id": self.cache_id,
                "frame_start": self.cache_id * (self.window_size - self.overlap),
            }
            partial = destination.with_name(destination.name + ".partial")
            try:
                torch.save(shard, partial)
                os.replace(partial, destination)
            finally:
                partial.unlink(missing_ok=True)
        self.cache_id += 1

    # 更新相邻窗口参考状态
    def _update_cache(self, new_window_cache, sp_graph=None):
        self.prev_window_cache = new_window_cache
        self.anchor_sp_graph = sp_graph
        gc.collect()

    # 将引擎恢复到未运行状态
    def _reset_state(self):
        self.cache_id = 0
        self.inference_queue = queue.Queue()
        self.registration_queue = queue.Queue()

        self.prev_window_cache = None
        self.anchor_sp_graph = None

        self._inference_thread = None
        self._registration_thread = None

        self.latencies = []

        gc.collect()

    # 模型推理线程
    @torch.no_grad()
    def _model_inference_worker(self):
        # 1️⃣ 持续读取输入
        while True:
            # 2️⃣ 检查停止信号
            sample_window = self.inference_queue.get()
            if sample_window is STOP_SIGNAL:
                return

            # 3️⃣ 统计模型推理时间
            t_start = time.perf_counter()

            # 4️⃣ 自动混合精度推理
            with torch.autocast(self.inference_device, dtype=self.dtype):
                prediction_window = self.delegate(sample_window)

            # 5️⃣ 记录推理耗时
            inference_duration = time.perf_counter() - t_start

            # 6️⃣ 把结果移到后处理设备
            processed_window = dict_to_device(prediction_window, self.process_device)
            # 7️⃣ 发送给配准线程
            self.registration_queue.put((processed_window, inference_duration))
            # 8️⃣ 清理 CUDA 缓存
            if self.inference_device == 'cuda':
                torch.cuda.empty_cache()

    # 配准线程
    def _registration_worker(self):
        ref_intrinsic = None        # 第一窗口估计出的参考相机内参
        tgt_sp_graph = None         # 当前窗口的深度分割图

        # 1️⃣ 从配准队列取结果
        while True:
            item = self.registration_queue.get()
            if item is STOP_SIGNAL:
                return

            working_window, inference_duration = item
            t_start = time.perf_counter()

            # 2️⃣ 去掉 batch 维度
            for key in working_window.keys():
                if isinstance(working_window[key], torch.Tensor):
                    working_window[key] = working_window[key].squeeze(0)

            # 3️⃣ 当前窗口重叠区域的置信度筛选
            # camera pose registration
            conf_thresh = torch.quantile(working_window['conf'][:self.overlap], self.top_conf_percentile,
                                         interpolation='nearest')
            tgt_mask = working_window['conf'][:self.overlap] >= conf_thresh

            # ⚠️ 非首窗口处理
            if self.prev_window_cache is not None:
                # 1️⃣ 强制使用固定内参
                # fixed intrinsic enforce
                working_window['local_points'] = unproject_depth_to_local_points(
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )
                # 2️⃣ 构造双向高置信度掩码
                # mutual conf mask
                prev_conf_thresh = torch.quantile(self.prev_window_cache['conf'][-self.overlap:],
                                                  self.top_conf_percentile, interpolation='nearest')
                conf_mask = (self.prev_window_cache['conf'][-self.overlap:] >= prev_conf_thresh) & tgt_mask

                # 3️⃣ 取相邻窗口的重叠点云
                # metric depth align
                prev_local_points = self.prev_window_cache['local_points'][-self.overlap:]
                cur_local_points = working_window['local_points'][:self.overlap]

                # 4️⃣ 求相似变换
                s_d, R, t = register_adjacent_windows(
                    prev_local_points,
                    cur_local_points,
                    self.prev_window_cache['camera_poses'][-self.overlap:],
                    working_window['camera_poses'][:self.overlap],
                    conf_mask
                )

                # 5️⃣ 统一当前窗口的深度尺度
                working_window['local_points'] = s_d * working_window.pop('local_points')
                # 6️⃣ 更新相机位姿
                working_window['camera_poses'] = apply_sim3_to_pose(working_window.pop('camera_poses'), s_d, R, t)

                # 7️⃣ 可选深度细化
                if self.depth_refine:
                    tgt_pcd = working_window['local_points'].cpu().numpy()
                    tgt_sp_graph = self._build_segment_graph(
                        working_window['local_points'],
                        working_window['conf'],
                        working_window.get('images'),
                    )
                    refine_kwargs = {}
                    if self.diagnostic_sink is not None:
                        refine_kwargs.update(
                            diagnostic_sink=self.diagnostic_sink,
                            diagnostic_context=self._diagnostic_context(),
                        )
                    working_window['local_points'] = working_window['local_points'] * refine_depth_segments(
                        self.prev_window_cache['local_points'].cpu().numpy(),
                        tgt_pcd,
                        self.anchor_sp_graph,
                        tgt_sp_graph,
                        self.overlap,
                        **refine_kwargs,
                    )
            # ⚠️ 首窗口处理
            else:
                # 1️⃣ 估计参考内参
                _, intrinsic_ = estimate_pseudo_depth_and_intrinsics(working_window['local_points'])
                ref_intrinsic = intrinsic_[0]
                # 2️⃣ 使用参考内参重新生成点云
                working_window['local_points'] = unproject_depth_to_local_points(
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )

                # 3️⃣ 创建首窗口分割图
                if self.depth_refine:
                    tgt_sp_graph = self._build_segment_graph(
                        working_window['local_points'],
                        working_window['conf'],
                        working_window.get('images'),
                    )

            # ⚠️ 更新并保存窗口
            self._update_cache(working_window, tgt_sp_graph)
            self._save_cache()

            # 延迟记录
            reg_duration = time.perf_counter() - t_start
            total_process_time = inference_duration + reg_duration
            self.latencies.append(total_process_time)

    # 启动引擎
    def begin(self):
        # 1️⃣ 防止重复启动
        if self.running:
            raise RuntimeError('Cannot start a running inference engine')

        # 2️⃣ 创建本次运行的临时目录
        self.temp_cache_dir = pathlib.Path(tempfile.mkdtemp(dir=self.cache_dir))
        # 3️⃣ 创建两个后台线程
        self._inference_thread = threading.Thread(target=self._model_inference_worker, daemon=True)
        self._registration_thread = threading.Thread(target=self._registration_worker, daemon=True)
        # 4️⃣ 启动线程
        self._inference_thread.start()
        self._registration_thread.start()

        self.running = True

    # 提交窗口
    def forward(self, sample, **kwargs):
        self.inference_queue.put(sample)

    # 结束引擎
    def end(self):
        # 1️⃣状态检查
        if not self.running:
            raise RuntimeError('Cannot terminate a stopped inference engine')

        # 2️⃣ 等待推理线程处理完全部输入
        self.inference_queue.put(STOP_SIGNAL)
        self._inference_thread.join()
        # 3️⃣ 等待配准线程处理完全部预测结果
        self.registration_queue.put(STOP_SIGNAL)
        self._registration_thread.join()

        # 4️⃣ 打印性能统计
        if self.benchmark_latency:
            if self.latencies:
                print("\n" + "=" * 50)
                print("        INFERENCE PERFORMANCE SUMMARY        ")
                print("=" * 50)

                # Print list of all times
                latencies_ms = [t * 1000 for t in self.latencies]
                print(f"Raw Latencies (ms): {latencies_ms}")

                if len(self.latencies) > self.warmup_steps + 1:
                    steady_times = self.latencies[self.warmup_steps:-1]
                    avg_steady = sum(steady_times) / len(steady_times)
                    print("-" * 50)
                    print(f"Total Windows:     {len(self.latencies)}")
                    print(f"Warmup Windows:    {self.warmup_steps}")
                    print(f"Steady State Avg:  {avg_steady * 1000:.2f} ms")
                else:
                    avg_all = sum(self.latencies) / len(self.latencies)
                    print(f"Average (All):     {avg_all * 1000:.2f} ms")
                print("=" * 50 + "\n")

        # 5️⃣ 重置内存状态
        self._reset_state()
        self.running = False

    # 图像滑动窗口
    def img_sliding_window(self, imgs):
        if isinstance(imgs, torch.Tensor):
            if len(imgs.shape) == 5:
                return sliding_window_t(imgs, self.window_size, self.overlap, dim=1)
            return sliding_window_t(imgs, self.window_size, self.overlap, dim=0)
        elif isinstance(imgs, list):
            return sliding_window_l(imgs, self.window_size, self.overlap)

    # 读取单个缓存
    @staticmethod
    def parse_cache_file(cache_file, overlap=0):
        # 1️⃣ 加载缓存
        window_cache = torch.load(cache_file, map_location='cpu', weights_only=False)
        # 2️⃣ 删除窗口开头的重叠帧
        for key in window_cache.keys():
            if isinstance(window_cache[key], torch.Tensor):
                window_cache[key] = window_cache[key][overlap:]

        return window_cache

    # 聚合缓存
    @staticmethod
    def aggregate_caches(parsed_caches):
        # 1️⃣ 按字段收集
        aggregated_cache = defaultdict(list)
        for cache in parsed_caches:
            for k, v in cache.items():
                # 2️⃣ 忽略原有 points
                if k == 'points':
                    continue
                aggregated_cache[k].append(v)
        # 3️⃣ 拼接 Tensor
        for k in list(aggregated_cache.keys()):
            if isinstance(aggregated_cache[k][0], torch.Tensor):
                aggregated_cache[k] = torch.concat(aggregated_cache.pop(k), dim=0)[None]

        # 4️⃣ 计算全局点云
        aggregated_cache['points'] = torch.einsum(
            'bnij, bnhwj -> bnhwi',
            aggregated_cache['camera_poses'],
            homogenize_points(aggregated_cache['local_points'])
        )[..., :3]
        return aggregated_cache

    # 汇总完整推理结果
    def parse_inference_cache_summary(self, remove_cache=True):
        # 1️⃣ 检查缓存目录
        assert self.temp_cache_dir is not None
        # 2️⃣ 查找缓存文件
        cache_files = sorted(glob.glob(str(self.temp_cache_dir / 'window_cache_*.pt')),
                             key=lambda p: int(p.split('_')[-1].split('.')[0]))

        # 3️⃣ 解析第一个窗口
        parsed_caches = [self.parse_cache_file(cache_files[0])]
        # 4️⃣ 解析后续窗口
        for cache_fname in cache_files[1:]:
            parsed_caches.append(self.parse_cache_file(cache_fname, overlap=self.overlap))

        # 5️⃣ 聚合并执行父类后处理
        ret_dict = StreamingWindowEngine._post_process_pred(self.aggregate_caches(parsed_caches))

        # 6️⃣ 删除临时缓存
        if remove_cache:
            shutil.rmtree(self.temp_cache_dir)
        # 7️⃣ 返回最终结果
        return ret_dict

    def parse_pose_cache_summary(self, remove_cache=True):
        """Aggregate the bounded metrics-only pose shards."""
        if self.temp_cache_dir is None:
            raise RuntimeError("No inference cache is available")
        cache_files = sorted(
            glob.glob(str(self.temp_cache_dir / 'window_cache_*.pt')),
            key=lambda p: int(p.split('_')[-1].split('.')[0]),
        )
        if not cache_files:
            raise RuntimeError("No pose cache shards were produced")
        poses = []
        for index, cache_file in enumerate(cache_files):
            shard = torch.load(cache_file, map_location='cpu', weights_only=False)
            camera_poses = shard['camera_poses']
            if index:
                camera_poses = camera_poses[self.overlap:]
            poses.append(camera_poses)
        result = {"extrinsic": torch.cat(poses, dim=0)[None]}
        if remove_cache:
            shutil.rmtree(self.temp_cache_dir)
        return result
