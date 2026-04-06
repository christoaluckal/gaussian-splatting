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

import csv
import os
import pickle
import sys
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

    allowed_scales = {2, 4, 8}
    invalid_scales = [scale for scale in unique_scales if scale not in allowed_scales]
    if invalid_scales:
        raise ValueError(
            'resolution_scales must only contain values from [2, 4, 8]. '
            f'Received invalid scales: {invalid_scales}'
        )

    return unique_scales, list(reversed(unique_scales))


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


def _maybe_update_naive_lod_scale(iteration, lod_state, naive_lod_stage_iterations):
    next_scale_idx, next_scale = _resolve_naive_lod_scale(
        iteration,
        lod_state['lod_scales'],
        naive_lod_stage_iterations,
        phase_start_iteration=lod_state['phase_start_iteration'],
    )
    if next_scale_idx != lod_state['current_scale_idx']:
        previous_scale = lod_state['lod_scales'][lod_state['current_scale_idx']]
        print(
            f"\n[ITER {iteration}] Promoting naive LoD training from resolution scale "
            f"{previous_scale} to {next_scale}"
        )
        lod_state['current_scale_idx'] = next_scale_idx
    return next_scale


def _reset_naive_lod_phase(next_iteration, lod_state):
    lod_state['phase_start_iteration'] = next_iteration
    lod_state['current_scale_idx'] = 0
    reset_scale = lod_state['lod_scales'][0]
    print(
        f"\n[ITER {next_iteration}] Resetting naive LoD training to resolution scale "
        f"{reset_scale} for newly added viewpoints"
    )
    return reset_scale


def _initialize_csv_logger(csv_path, fieldnames):
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(csv_path, fieldnames, row):
    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(row)


def _select_fixed_wandb_eval_view(scene, eval_scale):
    test_cameras = scene.getTestCameras(scale=eval_scale)
    if not test_cameras:
        return None
    return test_cameras[len(test_cameras) // 2]


def _get_gpu_memory_stats_mb():
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'max_allocated_mb': 0.0,
            'max_reserved_mb': 0.0,
        }

    device = torch.cuda.current_device()
    bytes_per_mb = 1024.0 * 1024.0
    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / bytes_per_mb,
        'reserved_mb': torch.cuda.memory_reserved(device) / bytes_per_mb,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / bytes_per_mb,
        'max_reserved_mb': torch.cuda.max_memory_reserved(device) / bytes_per_mb,
    }


def _tensor_nbytes(tensor):
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_tensors(nested)


def _gaussian_model_memory_mb(gaussians):
    total_bytes = 0
    seen_ptrs = set()

    for value in gaussians.__dict__.values():
        for tensor in _iter_tensors(value):
            ptr = tensor.data_ptr()
            if ptr != 0 and ptr not in seen_ptrs:
                total_bytes += _tensor_nbytes(tensor)
                seen_ptrs.add(ptr)

    return total_bytes / (1024.0 * 1024.0)


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
):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == 'sparse_adam':
        sys.exit(
            'Trying to use sparse adam but it is not installed, please install the '
            'correct rasterizer using pip install [3dgs_accel].'
        )

    resolution_scales, lod_scales = _prepare_resolution_scales(resolution_scales)
    finest_scale = resolution_scales[0]

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
        'gpu_allocated_mb',
        'gpu_reserved_mb',
        'gpu_peak_allocated_mb',
        'gpu_peak_reserved_mb',
        'gaussian_model_mb',
        'scene_load_time_sec',
        'total_training_time_sec',
    ]
    train_metrics_csv = os.path.join(dataset.model_path, 'train_metrics.csv')
    eval_metrics_csv = os.path.join(dataset.model_path, 'eval_metrics.csv')
    runtime_metrics_csv = os.path.join(dataset.model_path, 'runtime_metrics.csv')
    _initialize_csv_logger(train_metrics_csv, train_csv_fields)
    _initialize_csv_logger(eval_metrics_csv, eval_csv_fields)
    _initialize_csv_logger(runtime_metrics_csv, runtime_csv_fields)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    scene_load_start_time = time.perf_counter()
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == 'sparse_adam' and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=opt.iterations,
    )

    viewpoint_dict, viewpoint_indices = _build_viewpoint_stacks(scene, resolution_scales)
    active_viewpoint_start_idx = 0
    total_viewpoint_count = len(viewpoint_dict[finest_scale])
    initial_scale_idx, initial_scale = _resolve_naive_lod_scale(
        max(first_iter, 1),
        lod_scales,
        naive_lod_stage_iterations,
        phase_start_iteration=1,
    )
    lod_state = {
        'lod_scales': lod_scales,
        'current_scale_idx': initial_scale_idx,
        'phase_start_iteration': 1,
    }
    print(
        'Using naive LoD schedule with stage length '
        f'{naive_lod_stage_iterations} over scales {lod_scales}. '
        f'Starting at scale {initial_scale}.'
    )

    if fixed_wandb_eval_view is None:
        fixed_wandb_eval_view = _select_fixed_wandb_eval_view(scene, finest_scale)

    scene_load_time_sec = time.perf_counter() - scene_load_start_time
    scene_mem_stats = _get_gpu_memory_stats_mb()
    scene_gaussian_model_mb = _gaussian_model_memory_mb(gaussians)
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'scene_load',
            'iteration': 0,
            'gpu_allocated_mb': scene_mem_stats['allocated_mb'],
            'gpu_reserved_mb': scene_mem_stats['reserved_mb'],
            'gpu_peak_allocated_mb': scene_mem_stats['max_allocated_mb'],
            'gpu_peak_reserved_mb': scene_mem_stats['max_reserved_mb'],
            'gaussian_model_mb': scene_gaussian_model_mb,
            'scene_load_time_sec': scene_load_time_sec,
            'total_training_time_sec': '',
        },
    )
    if tb_writer:
        tb_writer.add_scalar('runtime/gpu_allocated_scene_load_mb', scene_mem_stats['allocated_mb'], 0)
        tb_writer.add_scalar('runtime/gpu_peak_allocated_scene_load_mb', scene_mem_stats['max_allocated_mb'], 0)
        tb_writer.add_scalar('runtime/gaussian_model_scene_load_mb', scene_gaussian_model_mb, 0)
        tb_writer.add_scalar('runtime/scene_load_time_sec', scene_load_time_sec, 0)
    if WANDB_FOUND and wandb.run is not None:
        wandb.log(
            {
                'runtime/gpu_allocated_scene_load_mb': scene_mem_stats['allocated_mb'],
                'runtime/gpu_reserved_scene_load_mb': scene_mem_stats['reserved_mb'],
                'runtime/gpu_peak_allocated_scene_load_mb': scene_mem_stats['max_allocated_mb'],
                'runtime/gpu_peak_reserved_scene_load_mb': scene_mem_stats['max_reserved_mb'],
                'runtime/gaussian_model_scene_load_mb': scene_gaussian_model_mb,
                'runtime/scene_load_time_sec': scene_load_time_sec,
            },
            step=0,
        )

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
            gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            if not viewpoint_indices:
                viewpoint_indices = _refill_viewpoint_indices(total_viewpoint_count)

            viewpoint_idx = viewpoint_indices.pop(randint(0, len(viewpoint_indices) - 1))
            current_scale = _maybe_update_naive_lod_scale(
                iteration,
                lod_state,
                naive_lod_stage_iterations,
            )
            render_scale = finest_scale if viewpoint_idx < active_viewpoint_start_idx else current_scale
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
                        'lod_stage_idx': lod_state['current_scale_idx'],
                        'num_gaussians': gaussians.get_xyz.shape[0],
                        'iter_time_ms': iter_time_ms,
                    },
                )

                runtime_mem_stats = _get_gpu_memory_stats_mb()
                gaussian_model_mb = _gaussian_model_memory_mb(gaussians)
                _append_csv_row(
                    runtime_metrics_csv,
                    runtime_csv_fields,
                    {
                        'event': 'iteration',
                        'iteration': iteration,
                        'gpu_allocated_mb': runtime_mem_stats['allocated_mb'],
                        'gpu_reserved_mb': runtime_mem_stats['reserved_mb'],
                        'gpu_peak_allocated_mb': runtime_mem_stats['max_allocated_mb'],
                        'gpu_peak_reserved_mb': runtime_mem_stats['max_reserved_mb'],
                        'gaussian_model_mb': gaussian_model_mb,
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
                            'gpu_allocated_mb': runtime_mem_stats['allocated_mb'],
                            'gpu_reserved_mb': runtime_mem_stats['reserved_mb'],
                            'gpu_peak_allocated_mb': runtime_mem_stats['max_allocated_mb'],
                            'gpu_peak_reserved_mb': runtime_mem_stats['max_reserved_mb'],
                            'gaussian_model_mb': gaussian_model_mb,
                            'scene_load_time_sec': '',
                            'total_training_time_sec': '',
                        },
                    )
                    if tb_writer:
                        tb_writer.add_scalar(
                            'runtime/gpu_peak_allocated_training_loop_mb',
                            runtime_mem_stats['max_allocated_mb'],
                            iteration,
                        )
                        tb_writer.add_scalar(
                            'runtime/gaussian_model_training_loop_mb',
                            gaussian_model_mb,
                            iteration,
                        )
                    if WANDB_FOUND and wandb.run is not None:
                        wandb.log(
                            {
                                'runtime/gpu_peak_allocated_training_loop_mb': runtime_mem_stats['max_allocated_mb'],
                                'runtime/gpu_peak_reserved_training_loop_mb': runtime_mem_stats['max_reserved_mb'],
                                'runtime/gaussian_model_training_loop_mb': gaussian_model_mb,
                            },
                            step=iteration,
                        )

                if WANDB_FOUND and wandb.run is not None:
                    wandb.log(
                        {
                            'train/photometric_loss': Ll1.item(),
                            'train/total_loss': loss.item(),
                            'train/depth_loss': Ll1depth,
                            'train/lod_scale': render_scale,
                            'train/lod_stage_idx': lod_state['current_scale_idx'],
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

                if iteration < opt.densify_until_iter:
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

                if iteration % splitter_itr == 0 and not def_flag:
                    print('Adding new gaussians')
                    previous_num_viewpoints = len(viewpoint_dict[finest_scale])
                    previous_active_viewpoint_start_idx = active_viewpoint_start_idx
                    scene.extend()
                    viewpoint_dict, _ = _build_viewpoint_stacks(scene, resolution_scales)
                    new_total_viewpoints = len(viewpoint_dict[finest_scale])
                    new_viewpoint_start_idx = previous_num_viewpoints
                    new_viewpoint_count = max(
                        0,
                        new_total_viewpoints - previous_num_viewpoints,
                    )
                    if new_viewpoint_count > 0:
                        active_viewpoint_start_idx = new_viewpoint_start_idx
                        total_viewpoint_count = new_total_viewpoints
                        viewpoint_indices = _refill_viewpoint_indices(total_viewpoint_count)
                        _reset_naive_lod_phase(iteration + 1, lod_state)
                        print(
                            'Sampling uniformly across all viewpoint indices with new active block:',
                            active_viewpoint_start_idx,
                            'to',
                            new_total_viewpoints - 1,
                        )
                        print('New Viewpoint Count:', new_viewpoint_count)
                    else:
                        active_viewpoint_start_idx = previous_active_viewpoint_start_idx
                        total_viewpoint_count = previous_num_viewpoints
                        print('No new viewpoints were added by this extension step.')

                if iteration < opt.iterations:
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
    print(f'Total training time: {total_training_time_sec:.2f}s')
    final_mem_stats = _get_gpu_memory_stats_mb()
    final_gaussian_model_mb = _gaussian_model_memory_mb(gaussians)
    _append_csv_row(
        runtime_metrics_csv,
        runtime_csv_fields,
        {
            'event': 'training_complete',
            'iteration': opt.iterations,
            'gpu_allocated_mb': final_mem_stats['allocated_mb'],
            'gpu_reserved_mb': final_mem_stats['reserved_mb'],
            'gpu_peak_allocated_mb': final_mem_stats['max_allocated_mb'],
            'gpu_peak_reserved_mb': final_mem_stats['max_reserved_mb'],
            'gaussian_model_mb': final_gaussian_model_mb,
            'scene_load_time_sec': '',
            'total_training_time_sec': total_training_time_sec,
        },
    )
    if tb_writer:
        tb_writer.add_scalar('runtime/total_training_time_sec', total_training_time_sec, opt.iterations)
    if WANDB_FOUND and wandb.run is not None:
        wandb.log(
            {
                'runtime/total_training_time_sec': total_training_time_sec,
                'runtime/gpu_peak_allocated_final_mb': final_mem_stats['max_allocated_mb'],
                'runtime/gpu_peak_reserved_final_mb': final_mem_stats['max_reserved_mb'],
                'runtime/gaussian_model_final_mb': final_gaussian_model_mb,
            },
            step=opt.iterations,
        )
        wandb.summary['runtime/total_training_time_sec'] = total_training_time_sec
        wandb.summary['runtime/gpu_peak_allocated_final_mb'] = final_mem_stats['max_allocated_mb']
        wandb.summary['runtime/gpu_peak_reserved_final_mb'] = final_mem_stats['max_reserved_mb']
        wandb.summary['runtime/gaussian_model_final_mb'] = final_gaussian_model_mb

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
    parser.add_argument('--resolution_scales', nargs='+', type=int, default=[2])
    parser.add_argument('--naive_lod_stage_iterations', type=int, default=5000)
    parser.add_argument('--disable_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='gaussian-splatting')
    parser.add_argument('--wandb_name', type=str, default='naive-lod')
    args = parser.parse_args(sys.argv[1:])

    if args.test_iterations is None:
        args.test_iterations = np.arange(1000, args.iterations + 1, 1000, dtype=int).tolist()
    else:
        args.test_iterations = list(args.test_iterations)
    args.save_iterations = list(args.save_iterations)
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
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
        wandb_context = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    with wandb_context:
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
        )

    print('\nTraining complete.')
