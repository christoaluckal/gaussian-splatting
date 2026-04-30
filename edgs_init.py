import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


_CORR_INIT_MODULE = None


def _load_corr_init_module():
    global _CORR_INIT_MODULE
    if _CORR_INIT_MODULE is not None:
        return _CORR_INIT_MODULE

    repo_root = Path(__file__).resolve().parent.parent
    edgs_root = repo_root / "EDGS"
    roma_root = edgs_root / "submodules" / "RoMa"

    for path in (str(edgs_root), str(roma_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    _CORR_INIT_MODULE = importlib.import_module("source.corr_init")
    return _CORR_INIT_MODULE


class _TrainCameraScene:
    def __init__(self, train_cameras):
        self._train_cameras = list(train_cameras)

    def getTrainCameras(self):
        return list(self._train_cameras)


def _camera_packet_sort_key(camera):
    packet_metadata = getattr(camera, "packet_metadata", None) or {}
    return (
        packet_metadata.get("packet_index", float("inf")),
        packet_metadata.get("camera_id", float("inf")),
        packet_metadata.get("timestamp_sec", float("inf")),
        camera.image_name,
    )


def _select_edgs_train_cameras(train_cameras, edgs_cfg):
    train_cameras = list(train_cameras)
    packet_set_size = getattr(edgs_cfg, "packet_window_size", 0)
    packet_skip_frames = max(int(getattr(edgs_cfg, "skip_frames", 0) or 0), 0)
    packet_max_frames = int(getattr(edgs_cfg, "max_frames", 0) or 0)
    packet_cameras = [camera for camera in train_cameras if getattr(camera, "packet_metadata", None)]
    if packet_set_size is None or packet_set_size <= 0:
        selected_cameras = (
            sorted(packet_cameras, key=_camera_packet_sort_key)
            if packet_cameras
            else train_cameras
        )
    else:
        if len(packet_cameras) < 2:
            selected_cameras = train_cameras
        else:
            packet_cameras = sorted(packet_cameras, key=_camera_packet_sort_key)
            target_count = max(int(packet_set_size), int(getattr(edgs_cfg, "nns_per_ref", 1)))
            window_size = min(target_count, len(packet_cameras))
            if window_size < 2:
                selected_cameras = train_cameras
            else:
                window_anchor = getattr(edgs_cfg, "packet_window_anchor", "middle")
                if window_anchor == "start":
                    start_idx = 0
                elif window_anchor == "end":
                    start_idx = len(packet_cameras) - window_size
                else:
                    start_idx = max((len(packet_cameras) - window_size) // 2, 0)
                end_idx = start_idx + window_size
                selected_cameras = packet_cameras[start_idx:end_idx]

    if packet_skip_frames > 0:
        selected_cameras = selected_cameras[:: packet_skip_frames + 1]

    if packet_max_frames > 0:
        selected_cameras = selected_cameras[:packet_max_frames]

    if len(selected_cameras) < 2:
        return train_cameras
    return selected_cameras


def _describe_edgs_camera_selection(selected_train_cameras, full_train_count):
    packet_metadata = [
        getattr(camera, "packet_metadata", None) or {}
        for camera in selected_train_cameras
        if getattr(camera, "packet_metadata", None)
    ]
    if not packet_metadata:
        return f'{len(selected_train_cameras)}/{full_train_count} train cameras'

    packet_indices = [metadata.get("packet_index") for metadata in packet_metadata]
    timestamps = [metadata.get("timestamp_sec") for metadata in packet_metadata]
    packet_indices = [index for index in packet_indices if index is not None]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if not packet_indices or not timestamps:
        return f'{len(selected_train_cameras)}/{full_train_count} packet-backed train cameras'

    duration_sec = max(timestamps) - min(timestamps)
    return (
        f'{len(selected_train_cameras)}/{full_train_count} train cameras, '
        f'packet range {min(packet_indices)}-{max(packet_indices)}, '
        f'time span {duration_sec:.3f}s'
    )


def build_edgs_init_config(args):
    return SimpleNamespace(
        use=args.edgs_init,
        matches_per_ref=args.edgs_matches_per_ref,
        num_refs=args.edgs_num_refs,
        nns_per_ref=args.edgs_nns_per_ref,
        scaling_factor=args.edgs_scaling_factor,
        proj_err_tolerance=args.edgs_proj_err_tolerance,
        roma_model=args.edgs_roma_model,
        add_SfM_init=args.edgs_add_sfm_init,
        init_extensions=args.edgs_init_extensions,
        packet_window_size=args.edgs_packet_window_size,
        packet_window_anchor=args.edgs_packet_window_anchor,
        skip_frames=args.edgs_skip_frames,
        max_frames=args.edgs_max_frames,
    )


def apply_edgs_initialization(
    gaussians,
    train_cameras,
    edgs_cfg,
    device="cuda",
    verbose=False,
):
    if edgs_cfg is None or not edgs_cfg.use:
        return False

    if gaussians.optimizer is None:
        raise ValueError("EDGS initialization requires GaussianModel.training_setup(...) first.")

    corr_init = _load_corr_init_module()
    selected_train_cameras = _select_edgs_train_cameras(train_cameras, edgs_cfg)
    if len(selected_train_cameras) < 2:
        return False
    print(
        '[EDGS init] Selected '
        + _describe_edgs_camera_selection(selected_train_cameras, len(train_cameras))
    )

    scene_wrapper = _TrainCameraScene(selected_train_cameras)
    n_splats_at_init = len(gaussians._xyz)

    init_fn = (
        corr_init.init_gaussians_with_corr_fast
        if edgs_cfg.nns_per_ref == 1
        else corr_init.init_gaussians_with_corr
    )
    init_fn(
        gaussians,
        scene_wrapper,
        edgs_cfg,
        device,
        verbose=verbose,
        roma_model=None,
    )

    if not edgs_cfg.add_SfM_init:
        with torch.no_grad():
            n_splats_after_init = len(gaussians._xyz)
            gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0], device=device)
            prune_mask = torch.cat(
                (
                    torch.ones(n_splats_at_init, dtype=torch.bool, device=device),
                    torch.zeros(
                        n_splats_after_init - n_splats_at_init,
                        dtype=torch.bool,
                        device=device,
                    ),
                ),
                dim=0,
            )
            gaussians.prune_points(prune_mask)

    with torch.no_grad():
        gaussians._scaling = gaussians.scaling_inverse_activation(
            gaussians.scaling_activation(gaussians._scaling) * 0.5
        )

    return True
