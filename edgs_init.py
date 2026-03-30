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
    scene_wrapper = _TrainCameraScene(train_cameras)
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
