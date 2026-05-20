#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import csv
import math
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from random import randint

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from edgs_init import apply_edgs_initialization, build_edgs_init_config
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

try:
    import pynvml  # type: ignore
    PYNVML_FOUND = True
except ImportError:
    PYNVML_FOUND = False


EDGS_TRAIN_RECIPE_OVERRIDES = {
    'opacity_reset_interval': 30000,
    'densify_from_iter': 500,
    'densify_grad_threshold': 0.0002,
    'exposure_lr_final': 0.0001,
}


def _apply_edgs_train_recipe_overrides(opt):
    overrides = {}
    for field_name, target_value in EDGS_TRAIN_RECIPE_OVERRIDES.items():
        current_value = getattr(opt, field_name)
        if current_value != target_value:
            overrides[field_name] = (current_value, target_value)
            setattr(opt, field_name, target_value)
    return overrides


def _apply_edgs_opacity_decay(gaussians):
    gaussians._opacity.data.add_(math.log(0.99))


def _apply_edgs_no_densify_prune(gaussians, radii, iteration, densify_until_iter, min_opacity=0.005):
    if iteration >= densify_until_iter:
        return

    gaussians.tmp_radii = radii
    prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
    gaussians.prune_points(prune_mask)
    gaussians.tmp_radii = None
    torch.cuda.empty_cache()

def _build_viewpoint_stacks(scene, resolution_scales):
    viewpoint_dict = {
        scale: scene.getTrainCameras(scale=scale)
        for scale in resolution_scales
    }
    viewpoint_indices = list(range(len(viewpoint_dict[resolution_scales[0]])))
    return viewpoint_dict, viewpoint_indices


def _refill_viewpoint_indices(num_viewpoints, start_idx=0):
    return [start_idx + idx for idx in torch.randperm(num_viewpoints).tolist()]


def _prepare_resolution_scales(resolution_scales):
    unique_scales = list(dict.fromkeys(resolution_scales))
    if not unique_scales:
        raise ValueError('resolution_scales must contain at least one scale.')

    allowed_scales = {1, 2, 4, 8}
    invalid_scales = [scale for scale in unique_scales if scale not in allowed_scales]
    if invalid_scales:
        raise ValueError(
            'resolution_scales must only contain values from [1, 2, 4, 8]. '
            f'Received invalid scales: {invalid_scales}'
        )

    return unique_scales, list(reversed(unique_scales))


def _log_loaded_resolution_summary(scene, resolution_scales, eval_scale):
    train_summary = []
    for scale in resolution_scales:
        cameras = scene.getTrainCameras(scale=scale)
        if cameras:
            train_summary.append(
                f'{scale}: {cameras[0].image_width}x{cameras[0].image_height}'
            )

    test_cameras = scene.getTestCameras(scale=eval_scale)
    eval_summary = 'no test cameras'
    if test_cameras:
        eval_summary = f'{eval_scale}: {test_cameras[0].image_width}x{test_cameras[0].image_height}'

    print('Loaded train camera resolutions by scale:', ', '.join(train_summary))
    print('Loaded eval camera resolution:', eval_summary)
    if eval_scale > 1:
        print(
            '[WARN] Evaluation and logged eval renders are downsampled by '
            f'resolution scale {eval_scale}. Use `--resolution_scales 1` '
            'for full-resolution visual inspection.'
        )


def _resolve_naive_lod_scale(
    iteration,
    lod_scales,
    naive_lod_stage_iterations,
    phase_start_iteration=1,
):
    if naive_lod_stage_iterations <= 0:
        raise ValueError('--naive_lod_stage_iterations must be at least 1.')

    zero_based_step = max(iteration - phase_start_iteration, 0)
    stage_idx = min(zero_based_step // naive_lod_stage_iterations, len(lod_scales) - 1)
    return stage_idx, lod_scales[stage_idx]


def _build_viewpoint_block(start_idx, end_idx, lod_scales, phase_start_iteration, scale_idx=0):
    return {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'lod_scales': lod_scales,
        'current_scale_idx': scale_idx,
        'phase_start_iteration': phase_start_iteration,
    }


def _find_viewpoint_block(blocks, viewpoint_idx):
    for block in blocks:
        if block['start_idx'] <= viewpoint_idx <= block['end_idx']:
            return block
    raise IndexError(f'No viewpoint block found for viewpoint index {viewpoint_idx}.')


def _resolve_effective_densify_until_iter(
    densify,
    densify_until_iter,
    *,
    def_flag,
    splitter_itr,
    extension_count,
):
    if not densify:
        return 0

    effective_densify_until_iter = densify_until_iter
    if not def_flag and splitter_itr > 0 and extension_count > 0:
        # Keep densification alive through the iteration that appends the last block.
        last_append_iteration = splitter_itr * extension_count
        effective_densify_until_iter = max(effective_densify_until_iter, last_append_iteration + 1)

    return effective_densify_until_iter


def _maybe_update_naive_lod_scale(iteration, lod_state, naive_lod_stage_iterations, log_prefix='Promoting naive LoD training'):
    next_scale_idx, next_scale = _resolve_naive_lod_scale(
        iteration,
        lod_state['lod_scales'],
        naive_lod_stage_iterations,
        phase_start_iteration=lod_state['phase_start_iteration'],
    )
    if next_scale_idx != lod_state['current_scale_idx']:
        previous_scale = lod_state['lod_scales'][lod_state['current_scale_idx']]
        print(
            f"\n[ITER {iteration}] {log_prefix} from resolution scale "
            f"{previous_scale} to {next_scale}"
        )
        lod_state['current_scale_idx'] = next_scale_idx
    return next_scale_idx, next_scale


def _reset_naive_lod_phase(next_iteration, lod_state, log_message='Resetting naive LoD training to resolution scale'):
    reset_scale_idx = 0
    lod_state['phase_start_iteration'] = next_iteration
    lod_state['current_scale_idx'] = reset_scale_idx
    reset_scale = lod_state['lod_scales'][reset_scale_idx]
    print(
        f"\n[ITER {next_iteration}] {log_message} "
        f"{reset_scale} for newly added viewpoints"
    )
    return reset_scale_idx, reset_scale


def _initialize_csv_logger(csv_path, fieldnames):
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(csv_path, fieldnames, row):
    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(row)


def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def _select_fixed_wandb_eval_view(scene, eval_scale):
    test_cameras = scene.getTestCameras(scale=eval_scale)
    if not test_cameras:
        return None
    return test_cameras[len(test_cameras) // 2]


def _get_gpu_memory_mb():
    if not torch.cuda.is_available():
        return 0.0

    device = torch.cuda.current_device()
    bytes_per_mb = 1024.0 * 1024.0
    return torch.cuda.memory_reserved(device) / bytes_per_mb


def _get_gpu_peak_memory_mb():
    if not torch.cuda.is_available():
        return 0.0

    device = torch.cuda.current_device()
    bytes_per_mb = 1024.0 * 1024.0
    return torch.cuda.max_memory_reserved(device) / bytes_per_mb


def _reset_gpu_peak_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())


def _synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _GpuUsageSampler:
    def __init__(self, csv_path, interval_sec=1.0):
        self.csv_path = csv_path
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None
        self._samples = []
        self._csv_file = None
        self._writer = None
        self._device_index = None
        self._nvml_handle = None
        self._start_time = None
        self._provider = None
        self._error = None
        self.fieldnames = [
            'sample_idx',
            'timestamp_unix_sec',
            'elapsed_sec',
            'gpu_index',
            'gpu_utilization_pct',
            'memory_utilization_pct',
            'memory_used_mb',
            'memory_total_mb',
            'power_w',
        ]

    def start(self):
        if not torch.cuda.is_available():
            self._error = 'cuda unavailable'
            return False
        self._device_index = torch.cuda.current_device()
        try:
            if PYNVML_FOUND:
                pynvml.nvmlInit()
                self._provider = 'pynvml'
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            elif shutil.which('nvidia-smi'):
                self._provider = 'nvidia-smi'
            else:
                self._error = 'pynvml unavailable and nvidia-smi not found'
                return False
            self._csv_file = open(self.csv_path, 'w', newline='')
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._start_time = time.perf_counter()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True
        except Exception as exc:
            self._error = str(exc)
            self._close_file()
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return False

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(5.0, self.interval_sec * 4))
        summary = self._build_summary()
        self._close_file()
        if self._provider == 'pynvml':
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return summary

    def _close_file(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def _sample_pynvml(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
        except Exception:
            power_w = None
        return {
            'gpu_utilization_pct': float(util.gpu),
            'memory_utilization_pct': float(util.memory),
            'memory_used_mb': mem.used / (1024.0 * 1024.0),
            'memory_total_mb': mem.total / (1024.0 * 1024.0),
            'power_w': power_w,
        }

    def _sample_nvidia_smi(self):
        command = [
            'nvidia-smi',
            f'--id={self._device_index}',
            '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits',
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        values = [value.strip() for value in result.stdout.strip().split(',')]
        if len(values) != 5:
            raise RuntimeError(f'unexpected nvidia-smi output: {result.stdout!r}')

        def _parse_numeric(raw_value):
            if raw_value in {'N/A', '[N/A]'}:
                return None
            return float(raw_value)

        return {
            'gpu_utilization_pct': _parse_numeric(values[0]),
            'memory_utilization_pct': _parse_numeric(values[1]),
            'memory_used_mb': _parse_numeric(values[2]),
            'memory_total_mb': _parse_numeric(values[3]),
            'power_w': _parse_numeric(values[4]),
        }

    def _read_gpu_sample(self):
        if self._provider == 'pynvml':
            return self._sample_pynvml()
        if self._provider == 'nvidia-smi':
            return self._sample_nvidia_smi()
        raise RuntimeError(f'unsupported GPU sampler provider: {self._provider}')

    def _sample_once(self):
        gpu_sample = self._read_gpu_sample()

        timestamp_unix_sec = time.time()
        elapsed_sec = time.perf_counter() - self._start_time
        sample = {
            'sample_idx': len(self._samples),
            'timestamp_unix_sec': timestamp_unix_sec,
            'elapsed_sec': elapsed_sec,
            'gpu_index': self._device_index,
            'gpu_utilization_pct': gpu_sample['gpu_utilization_pct'],
            'memory_utilization_pct': gpu_sample['memory_utilization_pct'],
            'memory_used_mb': gpu_sample['memory_used_mb'],
            'memory_total_mb': gpu_sample['memory_total_mb'],
            'power_w': gpu_sample['power_w'],
        }
        self._samples.append(sample)
        self._writer.writerow(sample)
        self._csv_file.flush()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception as exc:
                self._error = str(exc)
                break
            self._stop_event.wait(self.interval_sec)
        if self._provider is not None and self._writer is not None:
            try:
                self._sample_once()
            except Exception:
                pass

    def _integrate(self, key, transform=lambda value: value):
        if len(self._samples) < 2:
            return 0.0
        total = 0.0
        for prev, current in zip(self._samples, self._samples[1:]):
            prev_value = prev.get(key)
            if prev_value is None:
                continue
            delta_t = max(0.0, current['elapsed_sec'] - prev['elapsed_sec'])
            total += transform(prev_value) * delta_t
        return total

    def _build_summary(self):
        if not self._samples:
            return {
                'available': False,
                'provider': self._provider,
                'error': self._error,
            }

        gpu_utils = [sample['gpu_utilization_pct'] for sample in self._samples if sample['gpu_utilization_pct'] is not None]
        mem_utils = [sample['memory_utilization_pct'] for sample in self._samples if sample['memory_utilization_pct'] is not None]
        mem_used = [sample['memory_used_mb'] for sample in self._samples if sample['memory_used_mb'] is not None]
        power_values = [sample['power_w'] for sample in self._samples if sample['power_w'] is not None]

        gpu_utilization_seconds = self._integrate('gpu_utilization_pct', lambda value: value / 100.0)
        gpu_memory_gb_hours = self._integrate('memory_used_mb', lambda value: (value / 1024.0) / 3600.0)
        gpu_energy_wh = self._integrate('power_w', lambda value: value / 3600.0) if power_values else None

        return {
            'available': True,
            'provider': self._provider,
            'error': self._error,
            'sample_count': len(self._samples),
            'sampling_interval_sec': self.interval_sec,
            'gpu_utilization_avg_pct': sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
            'gpu_utilization_peak_pct': max(gpu_utils) if gpu_utils else None,
            'gpu_utilization_seconds': gpu_utilization_seconds,
            'gpu_memory_utilization_avg_pct': sum(mem_utils) / len(mem_utils) if mem_utils else None,
            'gpu_memory_utilization_peak_pct': max(mem_utils) if mem_utils else None,
            'gpu_memory_used_avg_mb': sum(mem_used) / len(mem_used) if mem_used else None,
            'gpu_memory_used_peak_mb': max(mem_used) if mem_used else None,
            'gpu_memory_gb_hours': gpu_memory_gb_hours,
            'gpu_power_avg_w': sum(power_values) / len(power_values) if power_values else None,
            'gpu_power_peak_w': max(power_values) if power_values else None,
            'gpu_energy_wh': gpu_energy_wh,
            'total_observed_sec': self._samples[-1]['elapsed_sec'],
        }


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    pkl_name,
    def_flag,
    resolution_scales,
    naive_lod_stage_iterations,
    splitter_itr,
    fixed_wandb_eval_view,
    edgs_init_cfg,
    densify,
    edgs_train_recipe,
):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == 'sparse_adam':
        sys.exit(
            'Trying to use sparse adam but it is not installed, please install the '
            'correct rasterizer using pip install [3dgs_accel].'
        )

    resolution_scales, lod_scales = _prepare_resolution_scales(resolution_scales)
    finest_scale = resolution_scales[0]
    recipe_overrides = {}
    if edgs_train_recipe:
        recipe_overrides = _apply_edgs_train_recipe_overrides(opt)
        print('Using EDGS train recipe compatibility mode.')
        if recipe_overrides:
            print('Applied EDGS train recipe overrides:', recipe_overrides)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    train_csv_fields = [
        'iteration',
        'photometric_loss',
        'total_loss',
        'depth_loss',
        'lod_scale',
        'lod_stage_idx',
        'num_gaussians',
        'iter_time_ms',
    ]
    eval_csv_fields = [
        'iteration',
        'split',
        'eval_scale',
        'num_cameras',
        'l1',
        'psnr',
    ]
    runtime_csv_fields = [
        'event',
        'iteration',
        'gpu_memory_mb',
        'init_time_sec',
        'scene_load_time_sec',
        'total_training_time_sec',
    ]
    gpu_summary_csv_fields = [
        'available',
        'provider',
        'sample_count',
        'sampling_interval_sec',
        'end_to_end_time_sec',
        'gpu_utilization_avg_pct',
        'gpu_utilization_peak_pct',
        'gpu_utilization_seconds',
        'gpu_memory_utilization_avg_pct',
        'gpu_memory_utilization_peak_pct',
        'gpu_memory_used_avg_mb',
        'gpu_memory_used_peak_mb',
        'gpu_memory_gb_hours',
        'gpu_power_avg_w',
        'gpu_power_peak_w',
        'gpu_energy_wh',
        'total_observed_sec',
        'error',
    ]
    train_metrics_csv = os.path.join(dataset.model_path, 'train_metrics.csv')
    eval_metrics_csv = os.path.join(dataset.model_path, 'eval_metrics.csv')
    runtime_metrics_csv = os.path.join(dataset.model_path, 'runtime_metrics.csv')
    gpu_metrics_csv = os.path.join(dataset.model_path, 'gpu_metrics.csv')
    gpu_summary_csv = os.path.join(dataset.model_path, 'gpu_summary.csv')
    _remove_if_exists(gpu_metrics_csv)
    _initialize_csv_logger(train_metrics_csv, train_csv_fields)
    _initialize_csv_logger(eval_metrics_csv, eval_csv_fields)
    _initialize_csv_logger(runtime_metrics_csv, runtime_csv_fields)
    _initialize_csv_logger(gpu_summary_csv, gpu_summary_csv_fields)

    gpu_sampler = _GpuUsageSampler(gpu_metrics_csv)
    gpu_sampler_started = gpu_sampler.start()
    if not gpu_sampler_started:
        print('GPU utilization sampler unavailable; proceeding without whole-run GPU utilization logging.')

    scene_load_start_time = time.perf_counter()
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    initialization_start_time = time.perf_counter()
    _reset_gpu_peak_memory_stats()
    scene_edgs_init_cfg = edgs_init_cfg
    # EDGS initializes the base block after Scene(...) and training_setup(...),
    # not during scene construction.
    if edgs_train_recipe and def_flag and edgs_init_cfg is not None and edgs_init_cfg.use:
        scene_edgs_init_cfg = None
    scene = Scene(
        dataset,
        gaussians,
        shuffle=not edgs_train_recipe,
        resolution_scales=resolution_scales,
        edgs_init_cfg=scene_edgs_init_cfg,
        training_args=opt,
        device="cuda",
    )
    initialization_time_sec = time.perf_counter() - initialization_start_time
    gaussians.training_setup(opt)
    _synchronize_cuda()
    post_scene_init_gpu_memory_mb = _get_gpu_memory_mb()
    post_scene_init_peak_gpu_memory_mb = _get_gpu_peak_memory_mb()
    if edgs_train_recipe and def_flag and edgs_init_cfg is not None and edgs_init_cfg.use:
        _reset_gpu_peak_memory_stats()
        _synchronize_cuda()
        edgs_base_init_start_time = time.perf_counter()
        apply_edgs_initialization(
            gaussians,
            scene.getTrainCameras(scale=finest_scale),
            edgs_init_cfg,
            device="cuda",
        )
        _synchronize_cuda()
        scene.runtime_stats['edgs_base_init_time_sec'] += time.perf_counter() - edgs_base_init_start_time
        scene.runtime_stats['edgs_base_init_gpu_memory_mb'] = _get_gpu_memory_mb()
        scene.runtime_stats['edgs_base_init_peak_gpu_memory_mb'] = _get_gpu_peak_memory_mb()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif 0 in saving_iterations:
        print('\n[ITER 0] Saving initialized Gaussians')
        scene.save(0)
    _synchronize_cuda()
    post_init_gpu_memory_mb = _get_gpu_memory_mb()
    post_init_peak_gpu_memory_mb = max(post_scene_init_peak_gpu_memory_mb, _get_gpu_peak_memory_mb())

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == 'sparse_adam' and SPARSE_ADAM_AVAILABLE
    extension_count = len(getattr(scene, 'extension_set', []))
    effective_densify_until_iter = _resolve_effective_densify_until_iter(
        densify,
        opt.densify_until_iter,
        def_flag=def_flag,
        splitter_itr=splitter_itr,
        extension_count=extension_count,
    )
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=opt.iterations,
    )

    viewpoint_dict, viewpoint_indices = _build_viewpoint_stacks(scene, resolution_scales)
    _log_loaded_resolution_summary(scene, resolution_scales, finest_scale)
    total_viewpoint_count = len(viewpoint_dict[finest_scale])
    effective_naive_lod_stage_iterations = naive_lod_stage_iterations
    if not def_flag and splitter_itr > 0 and len(lod_scales) > 0:
        derived_stage_iterations = max(1, splitter_itr // len(lod_scales))
        if derived_stage_iterations != naive_lod_stage_iterations:
            print(
                'Overriding naive LoD stage length for split training: '
                f'{naive_lod_stage_iterations} -> {derived_stage_iterations} '
                f'(splitter_itr={splitter_itr}, levels={len(lod_scales)}).'
            )
        effective_naive_lod_stage_iterations = derived_stage_iterations
    initial_scale_idx, initial_scale = _resolve_naive_lod_scale(
        max(first_iter, 1),
        lod_scales,
        effective_naive_lod_stage_iterations,
        phase_start_iteration=1,
    )
    lod_state = {
        'lod_scales': lod_scales,
        'current_scale_idx': initial_scale_idx,
        'phase_start_iteration': 1,
    }
    viewpoint_blocks = [
        _build_viewpoint_block(
            0,
            max(total_viewpoint_count - 1, 0),
            lod_scales,
            phase_start_iteration=1,
            scale_idx=initial_scale_idx,
        )
    ]
    print(
        'Using naive LoD schedule with stage length '
        f'{effective_naive_lod_stage_iterations} over scales {lod_scales}. '
        f'Starting at scale {initial_scale}.'
    )
    print(
        'Densification is '
        f'{"enabled" if densify else "disabled"}; '
        f'effective densify_until_iter={effective_densify_until_iter}.'
    )
    if effective_densify_until_iter != (opt.densify_until_iter if densify else 0):
        print(
            'Extended densification cutoff for split training so it remains active '
            f'through the final append iteration (base cutoff={opt.densify_until_iter}, '
            f'extensions={extension_count}, splitter_itr={splitter_itr}).'
        )
    print(f'Initialization time: {initialization_time_sec:.2f}s.')

    if fixed_wandb_eval_view is None:
        fixed_wandb_eval_view = _select_fixed_wandb_eval_view(scene, finest_scale)

    scene_load_time_sec = time.perf_counter() - scene_load_start_time
    scene_load_gpu_memory_mb = _get_gpu_memory_mb()
    edgs_base_init_time_sec = scene.runtime_stats.get('edgs_base_init_time_sec', 0.0)
    edgs_base_init_gpu_memory_mb = scene.runtime_stats.get('edgs_base_init_gpu_memory_mb', 0.0)
    edgs_base_init_peak_gpu_memory_mb = scene.runtime_stats.get('edgs_base_init_peak_gpu_memory_mb', edgs_base_init_gpu_memory_mb)
    edgs_extensions_init_time_sec = scene.runtime_stats.get('edgs_extensions_init_time_sec', 0.0)
    edgs_extensions_init_gpu_memory_mb = scene.runtime_stats.get('edgs_extensions_init_gpu_memory_mb', 0.0)
    edgs_extensions_init_peak_gpu_memory_mb = scene.runtime_stats.get('edgs_extensions_init_peak_gpu_memory_mb', edgs_extensions_init_gpu_memory_mb)
    edgs_extensions_init_count = scene.runtime_stats.get('edgs_extensions_init_count', 0)
    edgs_total_init_time_sec = edgs_base_init_time_sec + edgs_extensions_init_time_sec
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'initialization',
            'iteration': 0,
            'gpu_memory_mb': '',
            'init_time_sec': initialization_time_sec,
            'scene_load_time_sec': '',
            'total_training_time_sec': '',
        },
    )
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'post_scene_init',
            'iteration': 0,
            'gpu_memory_mb': post_scene_init_gpu_memory_mb,
            'init_time_sec': '',
            'scene_load_time_sec': '',
            'total_training_time_sec': '',
        },
    )
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'post_scene_init_peak',
            'iteration': 0,
            'gpu_memory_mb': post_scene_init_peak_gpu_memory_mb,
            'init_time_sec': '',
            'scene_load_time_sec': '',
            'total_training_time_sec': '',
        },
    )
    if edgs_base_init_time_sec > 0.0:
        _append_csv_row(
            runtime_metrics_csv,
            runtime_csv_fields,
            {
                'event': 'edgs_base_init',
                'iteration': 0,
                'gpu_memory_mb': edgs_base_init_gpu_memory_mb,
                'init_time_sec': edgs_base_init_time_sec,
                'scene_load_time_sec': '',
                'total_training_time_sec': '',
            },
        )
        _append_csv_row(
            runtime_metrics_csv,
            runtime_csv_fields,
            {
                'event': 'edgs_base_init_peak',
                'iteration': 0,
                'gpu_memory_mb': edgs_base_init_peak_gpu_memory_mb,
                'init_time_sec': '',
                'scene_load_time_sec': '',
                'total_training_time_sec': '',
            },
        )
    if edgs_extensions_init_count > 0:
        _append_csv_row(
            runtime_metrics_csv,
            runtime_csv_fields,
            {
                'event': 'edgs_extensions_init',
                'iteration': 0,
                'gpu_memory_mb': edgs_extensions_init_gpu_memory_mb,
                'init_time_sec': edgs_extensions_init_time_sec,
                'scene_load_time_sec': '',
                'total_training_time_sec': '',
            },
        )
        _append_csv_row(
            runtime_metrics_csv,
            runtime_csv_fields,
            {
                'event': 'edgs_extensions_init_peak',
                'iteration': 0,
                'gpu_memory_mb': edgs_extensions_init_peak_gpu_memory_mb,
                'init_time_sec': '',
                'scene_load_time_sec': '',
                'total_training_time_sec': '',
            },
        )
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'post_initialization',
            'iteration': 0,
            'gpu_memory_mb': post_init_gpu_memory_mb,
            'init_time_sec': '',
            'scene_load_time_sec': '',
            'total_training_time_sec': '',
        },
    )
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'post_initialization_peak',
            'iteration': 0,
            'gpu_memory_mb': post_init_peak_gpu_memory_mb,
            'init_time_sec': '',
            'scene_load_time_sec': '',
            'total_training_time_sec': '',
        },
    )
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'scene_load',
            'iteration': 0,
            'gpu_memory_mb': scene_load_gpu_memory_mb,
            'init_time_sec': '',
            'scene_load_time_sec': scene_load_time_sec,
            'total_training_time_sec': '',
        },
    )
    if tb_writer:
        tb_writer.add_scalar('runtime/init_time_sec', initialization_time_sec, 0)
        tb_writer.add_scalar('runtime/post_initialization_gpu_memory_mb', post_init_gpu_memory_mb, 0)
        tb_writer.add_scalar('runtime/post_initialization_peak_gpu_memory_mb', post_init_peak_gpu_memory_mb, 0)
        tb_writer.add_scalar('runtime/gpu_memory_scene_load_mb', scene_load_gpu_memory_mb, 0)
        tb_writer.add_scalar('runtime/scene_load_time_sec', scene_load_time_sec, 0)
        if edgs_base_init_time_sec > 0.0:
            tb_writer.add_scalar('runtime/edgs_base_init_time_sec', edgs_base_init_time_sec, 0)
            tb_writer.add_scalar('runtime/edgs_base_init_gpu_memory_mb', edgs_base_init_gpu_memory_mb, 0)
            tb_writer.add_scalar('runtime/edgs_base_init_peak_gpu_memory_mb', edgs_base_init_peak_gpu_memory_mb, 0)
        if edgs_extensions_init_count > 0:
            tb_writer.add_scalar('runtime/edgs_extensions_init_time_sec', edgs_extensions_init_time_sec, 0)
            tb_writer.add_scalar('runtime/edgs_extensions_init_gpu_memory_mb', edgs_extensions_init_gpu_memory_mb, 0)
            tb_writer.add_scalar('runtime/edgs_extensions_init_count', edgs_extensions_init_count, 0)
        if edgs_total_init_time_sec > 0.0:
            tb_writer.add_scalar('runtime/edgs_total_init_time_sec', edgs_total_init_time_sec, 0)
    if WANDB_FOUND and wandb.run is not None:
        runtime_log = {
            'runtime/init_time_sec': initialization_time_sec,
            'runtime/post_initialization_gpu_memory_mb': post_init_gpu_memory_mb,
            'runtime/post_initialization_peak_gpu_memory_mb': post_init_peak_gpu_memory_mb,
            'runtime/gpu_memory_scene_load_mb': scene_load_gpu_memory_mb,
            'runtime/scene_load_time_sec': scene_load_time_sec,
        }
        if edgs_base_init_time_sec > 0.0:
            runtime_log['runtime/edgs_base_init_time_sec'] = edgs_base_init_time_sec
            runtime_log['runtime/edgs_base_init_gpu_memory_mb'] = edgs_base_init_gpu_memory_mb
            runtime_log['runtime/edgs_base_init_peak_gpu_memory_mb'] = edgs_base_init_peak_gpu_memory_mb
        if edgs_extensions_init_count > 0:
            runtime_log['runtime/edgs_extensions_init_time_sec'] = edgs_extensions_init_time_sec
            runtime_log['runtime/edgs_extensions_init_gpu_memory_mb'] = edgs_extensions_init_gpu_memory_mb
            runtime_log['runtime/edgs_extensions_init_count'] = edgs_extensions_init_count
        if edgs_total_init_time_sec > 0.0:
            runtime_log['runtime/edgs_total_init_time_sec'] = edgs_total_init_time_sec
        wandb.log(runtime_log, step=0)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc='Training progress')
    training_start_time = time.perf_counter()
    first_iter += 1

    losses = []
    times = []
    num_gaussians = []
    l1s = []
    psnrs = []

    try:
        for iteration in range(first_iter, opt.iterations + 1):
            times.append(time.time_ns())
            num_gaussians.append(gaussians._xyz.shape[0])

            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    (
                        custom_cam,
                        do_training,
                        pipe.convert_SHs_python,
                        pipe.compute_cov3D_python,
                        keep_alive,
                        scaling_modifer,
                    ) = network_gui.receive()
                    if custom_cam is not None:
                        net_image = render(
                            custom_cam,
                            gaussians,
                            pipe,
                            background,
                            scaling_modifier=scaling_modifer,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                        )['render']
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception:
                    network_gui.conn = None

            iter_start.record()
            lr_iteration = max(iteration, 8_000) if edgs_train_recipe else iteration
            gaussians.update_learning_rate(lr_iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            if not viewpoint_indices:
                viewpoint_indices = _refill_viewpoint_indices(total_viewpoint_count)

            viewpoint_idx = viewpoint_indices.pop(randint(0, len(viewpoint_indices) - 1))
            active_block = viewpoint_blocks[-1]
            _maybe_update_naive_lod_scale(
                iteration,
                active_block,
                effective_naive_lod_stage_iterations,
                log_prefix='Promoting active viewpoint block from resolution scale',
            )
            viewpoint_block = _find_viewpoint_block(viewpoint_blocks, viewpoint_idx)
            render_scale = viewpoint_block['lod_scales'][viewpoint_block['current_scale_idx']]
            viewpoint_cam = viewpoint_dict[render_scale][viewpoint_idx]

            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device='cuda') if opt.random_background else background
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            image = render_pkg['render']
            viewspace_point_tensor = render_pkg['viewspace_points']
            visibility_filter = render_pkg['visibility_filter']
            radii = render_pkg['radii']

            if viewpoint_cam.alpha_mask is not None:
                image *= viewpoint_cam.alpha_mask.cuda()

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            losses.append(loss.item())

            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg['depth']
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0.0

            loss.backward()
            iter_end.record()

            if iteration < opt.iterations and edgs_train_recipe:
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            'Loss': f'{ema_loss_for_log:.7f}',
                            'Depth': f'{ema_Ll1depth_for_log:.7f}',
                            'LoD': render_scale,
                            'NG': f'{gaussians.get_xyz.shape[0]}',
                        }
                    )
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                iter_time_ms = iter_start.elapsed_time(iter_end)
                _append_csv_row(
                    train_metrics_csv,
                    train_csv_fields,
                    {
                        'iteration': iteration,
                        'photometric_loss': Ll1.item(),
                        'total_loss': loss.item(),
                        'depth_loss': Ll1depth,
                        'lod_scale': render_scale,
                        'lod_stage_idx': viewpoint_block['current_scale_idx'],
                        'num_gaussians': gaussians.get_xyz.shape[0],
                        'iter_time_ms': iter_time_ms,
                    },
                )

                training_loop_gpu_memory_mb = _get_gpu_memory_mb()
                _append_csv_row(
                    runtime_metrics_csv,
                    runtime_csv_fields,
                    {
                        'event': 'iteration',
                        'iteration': iteration,
                        'gpu_memory_mb': training_loop_gpu_memory_mb,
                        'init_time_sec': '',
                        'scene_load_time_sec': '',
                        'total_training_time_sec': '',
                    },
                )
                if iteration == first_iter:
                    _append_csv_row(
                        runtime_metrics_csv,
                        runtime_csv_fields,
                        {
                            'event': 'training_loop',
                            'iteration': iteration,
                            'gpu_memory_mb': training_loop_gpu_memory_mb,
                            'init_time_sec': '',
                            'scene_load_time_sec': '',
                            'total_training_time_sec': '',
                        },
                    )
                    if tb_writer:
                        tb_writer.add_scalar(
                            'runtime/gpu_memory_training_loop_mb',
                            training_loop_gpu_memory_mb,
                            iteration,
                        )
                    if WANDB_FOUND and wandb.run is not None:
                        wandb.log(
                            {'runtime/gpu_memory_training_loop_mb': training_loop_gpu_memory_mb},
                            step=iteration,
                        )

                if WANDB_FOUND and wandb.run is not None:
                    wandb.log(
                        {
                            'train/photometric_loss': Ll1.item(),
                            'train/total_loss': loss.item(),
                            'train/depth_loss': Ll1depth,
                            'train/lod_scale': render_scale,
                            'train/lod_stage_idx': viewpoint_block['current_scale_idx'],
                            'train/num_gaussians': gaussians.get_xyz.shape[0],
                        },
                        step=iteration,
                    )

                l1, psnr_value = training_report(
                    tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                    dataset.train_test_exp,
                    finest_scale,
                    render_scale,
                    eval_metrics_csv,
                    eval_csv_fields,
                    fixed_wandb_eval_view,
                )
                if l1 is not None:
                    l1s.append(l1)
                    psnrs.append(psnr_value)
                if iteration in saving_iterations:
                    print(f'\n[ITER {iteration}] Saving Gaussians')
                    scene.save(iteration)

                if iteration < effective_densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            radii,
                        )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()
                elif edgs_train_recipe:
                    _apply_edgs_no_densify_prune(
                        gaussians,
                        radii,
                        iteration,
                        effective_densify_until_iter,
                    )

                if (
                    edgs_train_recipe
                    and iteration < effective_densify_until_iter
                    and iteration % 10 == 0
                ):
                    _apply_edgs_opacity_decay(gaussians)

                if iteration % splitter_itr == 0 and not def_flag:
                    print('Adding new gaussians')
                    previous_num_viewpoints = len(viewpoint_dict[finest_scale])
                    scene.extend()
                    viewpoint_dict, _ = _build_viewpoint_stacks(scene, resolution_scales)
                    new_total_viewpoints = len(viewpoint_dict[finest_scale])
                    new_viewpoint_start_idx = previous_num_viewpoints
                    new_viewpoint_count = max(
                        0,
                        new_total_viewpoints - previous_num_viewpoints,
                    )
                    if new_viewpoint_count > 0:
                        total_viewpoint_count = new_total_viewpoints
                        viewpoint_indices = _refill_viewpoint_indices(total_viewpoint_count)
                        reset_scale_idx, reset_scale = _reset_naive_lod_phase(
                            iteration + 1,
                            lod_state,
                            log_message='Starting appended viewpoint block at resolution scale',
                        )
                        viewpoint_blocks.append(
                            _build_viewpoint_block(
                                new_viewpoint_start_idx,
                                new_total_viewpoints - 1,
                                lod_scales,
                                phase_start_iteration=iteration + 1,
                                scale_idx=reset_scale_idx,
                            )
                        )
                        print(
                            'Sampling uniformly across all viewpoint indices with new active block:',
                            new_viewpoint_start_idx,
                            'to',
                            new_total_viewpoints - 1,
                        )
                        print(
                            'Older viewpoint blocks keep their highest promoted scale; '
                            f'new block starts at scale {reset_scale}.'
                        )
                        print('New Viewpoint Count:', new_viewpoint_count)
                    else:
                        total_viewpoint_count = previous_num_viewpoints
                        print('No new viewpoints were added by this extension step.')

                if iteration < opt.iterations and not edgs_train_recipe:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration in checkpoint_iterations:
                    print(f'\n[ITER {iteration}] Saving Checkpoint')
                    torch.save(
                        (gaussians.capture(), iteration),
                        scene.model_path + '/chkpnt' + str(iteration) + '.pth',
                    )
    except Exception:
        print(traceback.format_exc())

    total_training_time_sec = time.perf_counter() - training_start_time
    gpu_summary = gpu_sampler.stop()
    end_to_end_time_sec = scene_load_time_sec + total_training_time_sec
    print(f'Total training time: {total_training_time_sec:.2f}s')
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'training_complete',
            'iteration': opt.iterations,
            'gpu_memory_mb': '',
            'init_time_sec': '',
            'scene_load_time_sec': '',
            'total_training_time_sec': total_training_time_sec,
        },
    )
    _append_csv_row(
        gpu_summary_csv,
        gpu_summary_csv_fields,
        {
            'available': gpu_summary.get('available', False),
            'provider': gpu_summary.get('provider'),
            'sample_count': gpu_summary.get('sample_count'),
            'sampling_interval_sec': gpu_summary.get('sampling_interval_sec'),
            'end_to_end_time_sec': end_to_end_time_sec,
            'gpu_utilization_avg_pct': gpu_summary.get('gpu_utilization_avg_pct'),
            'gpu_utilization_peak_pct': gpu_summary.get('gpu_utilization_peak_pct'),
            'gpu_utilization_seconds': gpu_summary.get('gpu_utilization_seconds'),
            'gpu_memory_utilization_avg_pct': gpu_summary.get('gpu_memory_utilization_avg_pct'),
            'gpu_memory_utilization_peak_pct': gpu_summary.get('gpu_memory_utilization_peak_pct'),
            'gpu_memory_used_avg_mb': gpu_summary.get('gpu_memory_used_avg_mb'),
            'gpu_memory_used_peak_mb': gpu_summary.get('gpu_memory_used_peak_mb'),
            'gpu_memory_gb_hours': gpu_summary.get('gpu_memory_gb_hours'),
            'gpu_power_avg_w': gpu_summary.get('gpu_power_avg_w'),
            'gpu_power_peak_w': gpu_summary.get('gpu_power_peak_w'),
            'gpu_energy_wh': gpu_summary.get('gpu_energy_wh'),
            'total_observed_sec': gpu_summary.get('total_observed_sec'),
            'error': gpu_summary.get('error'),
        },
    )
    if tb_writer:
        tb_writer.add_scalar('runtime/total_training_time_sec', total_training_time_sec, opt.iterations)
        tb_writer.add_scalar('runtime/end_to_end_time_sec', end_to_end_time_sec, opt.iterations)
        if gpu_summary.get('available'):
            if gpu_summary.get('gpu_utilization_avg_pct') is not None:
                tb_writer.add_scalar('runtime/gpu_utilization_avg_pct', gpu_summary['gpu_utilization_avg_pct'], opt.iterations)
            if gpu_summary.get('gpu_utilization_peak_pct') is not None:
                tb_writer.add_scalar('runtime/gpu_utilization_peak_pct', gpu_summary['gpu_utilization_peak_pct'], opt.iterations)
            if gpu_summary.get('gpu_utilization_seconds') is not None:
                tb_writer.add_scalar('runtime/gpu_utilization_seconds', gpu_summary['gpu_utilization_seconds'], opt.iterations)
            if gpu_summary.get('gpu_memory_used_avg_mb') is not None:
                tb_writer.add_scalar('runtime/gpu_memory_used_avg_mb', gpu_summary['gpu_memory_used_avg_mb'], opt.iterations)
            if gpu_summary.get('gpu_memory_used_peak_mb') is not None:
                tb_writer.add_scalar('runtime/gpu_memory_used_peak_mb', gpu_summary['gpu_memory_used_peak_mb'], opt.iterations)
            if gpu_summary.get('gpu_memory_gb_hours') is not None:
                tb_writer.add_scalar('runtime/gpu_memory_gb_hours', gpu_summary['gpu_memory_gb_hours'], opt.iterations)
            if gpu_summary.get('gpu_power_avg_w') is not None:
                tb_writer.add_scalar('runtime/gpu_power_avg_w', gpu_summary['gpu_power_avg_w'], opt.iterations)
            if gpu_summary.get('gpu_power_peak_w') is not None:
                tb_writer.add_scalar('runtime/gpu_power_peak_w', gpu_summary['gpu_power_peak_w'], opt.iterations)
            if gpu_summary.get('gpu_energy_wh') is not None:
                tb_writer.add_scalar('runtime/gpu_energy_wh', gpu_summary['gpu_energy_wh'], opt.iterations)
    if WANDB_FOUND and wandb.run is not None:
        wandb_runtime_log = {
            'runtime/total_training_time_sec': total_training_time_sec,
            'runtime/end_to_end_time_sec': end_to_end_time_sec,
            'runtime/gpu_sampler_available': gpu_summary.get('available', False),
        }
        if gpu_summary.get('available'):
            if gpu_summary.get('gpu_utilization_avg_pct') is not None:
                wandb_runtime_log['runtime/gpu_utilization_avg_pct'] = gpu_summary['gpu_utilization_avg_pct']
            if gpu_summary.get('gpu_utilization_peak_pct') is not None:
                wandb_runtime_log['runtime/gpu_utilization_peak_pct'] = gpu_summary['gpu_utilization_peak_pct']
            if gpu_summary.get('gpu_utilization_seconds') is not None:
                wandb_runtime_log['runtime/gpu_utilization_seconds'] = gpu_summary['gpu_utilization_seconds']
            if gpu_summary.get('gpu_memory_utilization_avg_pct') is not None:
                wandb_runtime_log['runtime/gpu_memory_utilization_avg_pct'] = gpu_summary['gpu_memory_utilization_avg_pct']
            if gpu_summary.get('gpu_memory_utilization_peak_pct') is not None:
                wandb_runtime_log['runtime/gpu_memory_utilization_peak_pct'] = gpu_summary['gpu_memory_utilization_peak_pct']
            if gpu_summary.get('gpu_memory_used_avg_mb') is not None:
                wandb_runtime_log['runtime/gpu_memory_used_avg_mb'] = gpu_summary['gpu_memory_used_avg_mb']
            if gpu_summary.get('gpu_memory_used_peak_mb') is not None:
                wandb_runtime_log['runtime/gpu_memory_used_peak_mb'] = gpu_summary['gpu_memory_used_peak_mb']
            if gpu_summary.get('gpu_memory_gb_hours') is not None:
                wandb_runtime_log['runtime/gpu_memory_gb_hours'] = gpu_summary['gpu_memory_gb_hours']
            if gpu_summary.get('gpu_power_avg_w') is not None:
                wandb_runtime_log['runtime/gpu_power_avg_w'] = gpu_summary['gpu_power_avg_w']
            if gpu_summary.get('gpu_power_peak_w') is not None:
                wandb_runtime_log['runtime/gpu_power_peak_w'] = gpu_summary['gpu_power_peak_w']
            if gpu_summary.get('gpu_energy_wh') is not None:
                wandb_runtime_log['runtime/gpu_energy_wh'] = gpu_summary['gpu_energy_wh']
            wandb_runtime_log['runtime/gpu_sampler_provider'] = gpu_summary.get('provider')
            wandb_runtime_log['runtime/gpu_sample_count'] = gpu_summary.get('sample_count')
        wandb.log(wandb_runtime_log, step=opt.iterations)
        wandb.summary['runtime/total_training_time_sec'] = total_training_time_sec
        wandb.summary['runtime/end_to_end_time_sec'] = end_to_end_time_sec
        wandb.summary['runtime/gpu_sampler_available'] = gpu_summary.get('available', False)
        if gpu_summary.get('available'):
            for key, value in wandb_runtime_log.items():
                if key not in {'runtime/total_training_time_sec', 'runtime/end_to_end_time_sec'}:
                    wandb.summary[key] = value
        elif gpu_summary.get('error'):
            wandb.summary['runtime/gpu_sampler_error'] = gpu_summary.get('error')

    if pkl_name:
        with open(pkl_name, 'wb') as f:
            data_dict = {
                'losses': losses,
                'times': times,
                'num_gaussians': num_gaussians,
                'l1s': l1s,
                'psnrs': psnrs,
                'total_training_time_sec': total_training_time_sec,
            }
            pickle.dump(data_dict, f)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join('./output/', unique_str[0:10])

    print(f'Output folder: {args.model_path}')
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print('Tensorboard not available: not logging progress')
    return tb_writer


def _resolve_wandb_name(args):
    if args.wandb_name:
        return args.wandb_name
    return f"run-{str(uuid.uuid4())[:10]}"


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
    eval_scale,
    training_scale,
    eval_metrics_csv,
    eval_csv_fields,
    fixed_wandb_eval_view=None,
):
    l1_test = None
    psnr_test = None
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train/lod_scale', training_scale, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras(scale=eval_scale)},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)['render'],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to('cuda'), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and idx < 5:
                        tb_writer.add_images(
                            config['name'] + '_view_{}/render'.format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + '_view_{}/ground_truth'.format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']}: "
                    f"L1 {l1_test} PSNR {psnr_test}"
                )
                _append_csv_row(
                    eval_metrics_csv,
                    eval_csv_fields,
                    {
                        'iteration': iteration,
                        'split': config['name'],
                        'eval_scale': eval_scale,
                        'num_cameras': len(config['cameras']),
                        'l1': l1_test.item(),
                        'psnr': psnr_test.item(),
                    },
                )
                if WANDB_FOUND and wandb.run is not None and config['name'] == 'test':
                    eval_log = {
                        'eval/L1': l1_test.item(),
                        'eval/PSNR': psnr_test.item(),
                        'eval/lod_scale': training_scale,
                    }
                    if fixed_wandb_eval_view is not None:
                        sample_image = torch.clamp(
                            renderFunc(fixed_wandb_eval_view, scene.gaussians, *renderArgs)['render'],
                            0.0,
                            1.0,
                        )
                        sample_gt = torch.clamp(
                            fixed_wandb_eval_view.original_image.to('cuda'),
                            0.0,
                            1.0,
                        )
                        if train_test_exp:
                            sample_image = sample_image[..., sample_image.shape[-1] // 2:]
                            sample_gt = sample_gt[..., sample_gt.shape[-1] // 2:]
                        eval_log['eval/images/render'] = wandb.Image(
                            sample_image.permute(1, 2, 0).detach().cpu().numpy(),
                            caption=f'{fixed_wandb_eval_view.image_name} render',
                        )
                        eval_log['eval/images/ground_truth'] = wandb.Image(
                            sample_gt.permute(1, 2, 0).detach().cpu().numpy(),
                            caption=f'{fixed_wandb_eval_view.image_name} gt',
                        )
                    wandb.log(eval_log, step=iteration)
                if tb_writer:
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - l1_loss',
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - psnr',
                        psnr_test,
                        iteration,
                    )

        torch.cuda.empty_cache()
        return l1_test, psnr_test

    return None, None


if __name__ == '__main__':
    parser = ArgumentParser(description='Training script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=None)
    parser.add_argument('--save_iterations', nargs='+', type=int, default=np.arange(900, 35000, 5000, dtype=int).tolist())
    parser.add_argument('--splitter_itr', type=int, default=10000)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--pkl_name', type=str, default='')
    parser.add_argument('--default', action='store_true')
    parser.add_argument('--resolution_scales', nargs='+', type=int, default=[1])
    parser.add_argument('--naive_lod_stage_iterations', type=int, default=5000)
    parser.add_argument(
        '--densify',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable standard Gaussian densification during training.',
    )
    parser.add_argument(
        '--edgs_init',
        action='store_true',
        default=False,
        help='Enable EDGS-style RoMa initialization.',
    )
    parser.add_argument('--edgs_matches_per_ref', type=int, default=15000)
    parser.add_argument('--edgs_num_refs', type=int, default=180)
    parser.add_argument('--edgs_nns_per_ref', type=int, default=3)
    parser.add_argument('--edgs_scaling_factor', type=float, default=0.001)
    parser.add_argument('--edgs_proj_err_tolerance', type=float, default=0.01)
    parser.add_argument('--edgs_roma_model', type=str, default='outdoors')
    parser.add_argument('--edgs_add_sfm_init', action='store_true', default=False)
    parser.add_argument(
        '--edgs_packet_window_size',
        type=int,
        default=0,
        help='If > 0, restrict EDGS initialization to a contiguous packet window of this many packet-backed cameras.',
    )
    parser.add_argument(
        '--edgs_packet_window_anchor',
        type=str,
        default='middle',
        choices=['start', 'middle', 'end'],
        help='Where to place the EDGS packet window when packet-backed cameras are available.',
    )
    parser.add_argument(
        '--edgs_skip_frames',
        type=int,
        default=0,
        help='For EDGS packet-backed input, keep every (skip_frames + 1)th frame after packet-window selection.',
    )
    parser.add_argument(
        '--edgs_max_frames',
        type=int,
        default=0,
        help='For EDGS packet-backed input, cap the selected EDGS frame set to this many frames after skipping; 0 disables the cap.',
    )
    parser.add_argument(
        '--edgs_init_extensions',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply EDGS initialization to split extension blocks as well.',
    )
    parser.add_argument(
        '--edgs_train_recipe',
        action='store_true',
        default=False,
        help='Use EDGS-like optimizer defaults and training schedule after initialization.',
    )
    parser.add_argument('--disable_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='gaussian-splatting')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument(
        '--wandb_name',
        type=str,
        default=None,
        help='Optional W&B run name. If omitted, a random name is generated for each launch.',
    )
    args = parser.parse_args(sys.argv[1:])

    if args.test_iterations is None:
        args.test_iterations = np.arange(1000, args.iterations + 1, 1000, dtype=int).tolist()
    else:
        args.test_iterations = list(args.test_iterations)
    args.save_iterations = list(args.save_iterations)
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    if 0 not in args.save_iterations:
        args.save_iterations.insert(0, 0)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    print('Optimizing ' + args.model_path)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not WANDB_FOUND and not args.disable_wandb:
        print('wandb not available: proceeding without Weights & Biases logging')

    fixed_wandb_eval_view = None
    wandb_context = nullcontext()
    if WANDB_FOUND and not args.disable_wandb:
        resolved_wandb_name = _resolve_wandb_name(args)
        args.wandb_name = resolved_wandb_name
        print(f'W&B run name: {resolved_wandb_name}')
        wandb_context = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=resolved_wandb_name,
            resume='never',
            config=vars(args),
        )

    with wandb_context:
        edgs_init_cfg = build_edgs_init_config(args)
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            args.pkl_name,
            args.default,
            args.resolution_scales,
            args.naive_lod_stage_iterations,
            args.splitter_itr,
            fixed_wandb_eval_view,
            edgs_init_cfg,
            args.densify,
            args.edgs_train_recipe,
        )

    print('\nTraining complete.')
