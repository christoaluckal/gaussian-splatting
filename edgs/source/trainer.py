import torch
from random import randint
from tqdm.rich import trange
from tqdm import tqdm as tqdm
from source.networks import Warper3DGS
import wandb
import sys

sys.path.append('./submodules/gaussian-splatting/')
import lpips
from source.losses import ssim, l1_loss, psnr
from rich.console import Console
from rich.theme import Theme
import numpy as np

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

from source.corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from source.utils_aux import log_samples
from gaussian_renderer import forward_k_times
import torchvision
from source.timer import Timer
import os

class EDGSTrainer:
    def __init__(self,
                 GS: Warper3DGS,
                 training_config,
                 dataset_white_background=False,
                 device=torch.device('cuda'),
                 log_wandb=True,
                 ):
        self.GS = GS
        self.scene = GS.scene
        self.viewpoint_stack = GS.viewpoint_stack
        self.gaussians = GS.gaussians

        self.training_config = training_config
        # self.GS_optimizer = GS.gaussians.optimizer
        self.dataset_white_background = dataset_white_background

        self.training_step = 1
        self.gs_step = 0
        self.CONSOLE = Console(width=120, theme=custom_theme)
        self.saving_iterations = training_config.save_iterations
        self.evaluate_iterations = None
        self.batch_size = training_config.batch_size
        self.ema_loss_for_log = 0.0

        # Logs in the format {step:{"loss1":loss1_value, "loss2":loss2_value}}
        self.logs_losses = {}
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.device = device
        self.timer = Timer()
        self.log_wandb = log_wandb



    def load_checkpoints(self, load_cfg):
        # Load 3DGS checkpoint
        if load_cfg.gs:
            self.gs.gaussians.restore(
                torch.load(f"{load_cfg.gs}/chkpnt{load_cfg.gs_step}.pth")[0],
                self.training_config)
            self.GS_optimizer = self.GS.gaussians.optimizer
            self.CONSOLE.print(f"3DGS loaded from checkpoint for iteration {load_cfg.gs_step}",
                               style="info")
            self.training_step += load_cfg.gs_step
            self.gs_step += load_cfg.gs_step

    def train(self, train_cfg):
        # 3DGS training
        self.GS_optimizer = self.GS.gaussians.optimizer
        self.CONSOLE.print("Train 3DGS for {} iterations".format(train_cfg.gs_epochs), style="info")    
        with trange(self.training_step, self.training_step + train_cfg.gs_epochs, desc="[green]Train gaussians") as progress_bar:
            for self.training_step in progress_bar:
                radii = self.train_step_gs(max_lr=train_cfg.max_lr, no_densify=train_cfg.no_densify)
                with torch.no_grad():
                    if train_cfg.no_densify:
                        self.prune(radii)
                    else:
                        if self.gs_step % train_cfg.densification_interval == 0:
                            self.densify_and_prune(radii)
                    if train_cfg.reduce_opacity:
                        # Slightly reduce opacity every few steps:
                        if self.gs_step < self.training_config.densify_until_iter and self.gs_step % 10 == 0:
                            opacities_new = torch.log(torch.exp(self.GS.gaussians._opacity.data) * 0.99)
                            self.GS.gaussians._opacity.data = opacities_new
                    self.timer.pause()
                    # Progress bar
                    if self.training_step % 10 == 0:
                        progress_bar.set_postfix({"[red]Loss": f"{self.ema_loss_for_log:.{7}f}"}, refresh=True)
                    # Log and save
                    if self.training_step in self.saving_iterations:
                        self.save_model()
                    if self.evaluate_iterations is not None:
                        if self.training_step in self.evaluate_iterations:
                            self.evaluate()
                    else:
                        if (self.training_step <= 3000 and self.training_step % 500 == 0) or \
                            (self.training_step > 3000 and self.training_step % 1000 == 228) :
                            self.evaluate()

                    if (self.training_step % 1000 == 0 or self.training_step == train_cfg.gs_epochs - 1) and self.GS.gaussians.probabilistic:
                        # torch.cuda.empty_cache()
                        self.render_set(self.scene, self.GS.pipe, self.training_step)
                    self.timer.start()


    def evaluate(self):
        torch.cuda.empty_cache()
        log_gen_images, log_real_images = [], []
        validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras(), 'cam_idx': self.training_config.TEST_CAM_IDX_TO_LOG},
                              {'name': 'train',
                               'cameras': [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in
                                           range(0, 150, 5)], 'cam_idx': 10})
        if self.log_wandb:
            wandb.log({f"Number of Gaussians": len(self.GS.gaussians._xyz)}, step=self.training_step)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_splat_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(self.GS(viewpoint)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(self.device), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_splat_test += self.lpips(image, gt_image).detach().double()
                    if idx in [config['cam_idx']]:
                        log_gen_images.append(image)
                        log_real_images.append(gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_splat_test /= len(config['cameras'])
                if self.log_wandb:
                    wandb.log({f"{config['name']}/L1": l1_test.item(), f"{config['name']}/PSNR": psnr_test.item(), \
                            f"{config['name']}/SSIM": ssim_test.item(), f"{config['name']}/LPIPS_splat": lpips_splat_test.item()}, step = self.training_step)
                self.CONSOLE.print("\n[ITER {}], #{} gaussians, Prob:{} Evaluating {}: L1={:.6f},  PSNR={:.6f}, SSIM={:.6f}, LPIPS_splat={:.6f} ".format(
                    self.training_step, len(self.GS.gaussians._xyz), self.GS.gaussians.probabilistic, config['name'], l1_test.item(), psnr_test.item(), ssim_test.item(), lpips_splat_test.item()), style="info")
        if self.log_wandb:
            with torch.no_grad():
                log_samples(torch.stack((log_real_images[0],log_gen_images[0])) , [], self.training_step, caption="Real and Generated Samples")
                wandb.log({"time": self.timer.get_elapsed_time()}, step=self.training_step)
        torch.cuda.empty_cache()

    def train_step_gs(self, max_lr = False, no_densify = False):
        self.gs_step += 1
        if max_lr:
            self.GS.gaussians.update_learning_rate(max(self.gs_step, 8_000))
        else:
            self.GS.gaussians.update_learning_rate(self.gs_step)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.gs_step % 1000 == 0:
            self.GS.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
      
        render_pkg = self.GS(viewpoint_cam=viewpoint_cam)
        image = render_pkg["render"]
        # Loss
        gt_image = viewpoint_cam.original_image.to(self.device)
        L1_loss = l1_loss(image, gt_image)
        

        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - self.training_config.lambda_dssim) * L1_loss + \
               self.training_config.lambda_dssim * ssim_loss
        loss_kl_scal = self.gaussians.compute_kl_uniform_scal()
        loss_kl_xyz = self.gaussians.compute_kl_xyz()
        loss_kl_opacity = self.gaussians.compute_kl_opacity()
        prob_loss = 1.0*(loss_kl_scal + loss_kl_xyz + loss_kl_opacity)
        wandb.log({"train/photo_loss": loss.item(),
                   "train/prob_loss": prob_loss.item()}, step=self.training_step)
        loss += prob_loss
        self.timer.pause() 
        self.logs_losses[self.training_step] = {"loss": loss.item(),
                                                "L1_loss": L1_loss.item(),
                                                "ssim_loss": ssim_loss.item()}
        
        if self.log_wandb:
            for k, v in self.logs_losses[self.training_step].items():
                wandb.log({f"train/{k}": v}, step=self.training_step)
        self.ema_loss_for_log = 0.4 * self.logs_losses[self.training_step]["loss"] + 0.6 * self.ema_loss_for_log
        self.timer.start()
        self.GS_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            if self.gs_step < self.training_config.densify_until_iter and not no_densify:
                self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]])
                self.GS.gaussians.add_densification_stats(render_pkg["viewspace_points"],
                                                                     render_pkg["visibility_filter"])

        # Optimizer step
        g = self.GS.gaussians
        # print("xyz_offset grad mean:", g.offsets["_xyz_offset"].grad.abs().mean().item())
        # print("scaling_offset grad mean:", g.offsets["_scaling_offset"].grad.abs().mean().item())
        # print("opacity_offset grad mean:", g.offsets["_opacity_offset"].grad.abs().mean().item())
        self.GS_optimizer.step()
        self.GS_optimizer.zero_grad(set_to_none=True)
        return render_pkg["radii"]

    def densify_and_prune(self, radii = None):
        # Densification or pruning
        if self.gs_step < self.training_config.densify_until_iter:
            if self.GS.gaussians.probabilistic:
                if self.gs_step % self.GS.gaussians.spawn_interval == 0:
                    self.GS.gaussians.spawn(self.GS.scene.cameras_extent)
            if (self.gs_step > self.training_config.densify_from_iter) and \
                    (self.gs_step % self.training_config.densification_interval == 0):
                size_threshold = 20 if self.gs_step > self.training_config.opacity_reset_interval else None
                self.GS.gaussians.densify_and_prune(self.training_config.densify_grad_threshold,
                                                               0.005,
                                                               self.GS.scene.cameras_extent,
                                                               size_threshold, radii)
            if self.gs_step % self.training_config.opacity_reset_interval == 0 or (
                    self.dataset_white_background and self.gs_step == self.training_config.densify_from_iter):
                self.GS.gaussians.reset_opacity()             

          

    def save_model(self):
        print("\n[ITER {}] Saving Gaussians".format(self.gs_step))
        self.scene.save(self.gs_step)
        print("\n[ITER {}] Saving Checkpoint".format(self.gs_step))
        torch.save((self.GS.gaussians.capture(), self.gs_step),
                self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth")


    def init_with_corr(self, cfg, opt, verbose=False, roma_model=None): 
        """
        Initializes image with matchings. Also removes SfM init points.
        Args:
            cfg: configuration part named init_wC. Check train.yaml
            verbose: whether you want to print intermediate results. Useful for debug.
            roma_model: optionally you can pass here preinit RoMA model to avoid reinit 
                it every time.  
        """
        if not cfg.use:
            return None
        N_splats_at_init = len(self.GS.gaussians._xyz)
        print("N_splats_at_init:", N_splats_at_init)
        if cfg.nns_per_ref == 1:
            init_fn = init_gaussians_with_corr_fast
        else:
            init_fn = init_gaussians_with_corr
        camera_set, selected_indices, visualization_dict = init_fn(
            self.GS.gaussians, 
            self.scene, 
            cfg, 
            opt,
            self.device,                                                                                    
            verbose=verbose,
            roma_model=roma_model)
        # self.gaussians.training_setup(self.training_config)
        self.GS_optimizer = self.gaussians.optimizer

        # Remove SfM points and leave only matchings inits
        if not cfg.add_SfM_init:
            with torch.no_grad():
                N_splats_after_init = len(self.GS.gaussians._xyz)
                print("N_splats_after_init:", N_splats_after_init)
                self.gaussians.tmp_radii = torch.zeros(self.gaussians._xyz.shape[0]).to(self.device)
                mask = torch.concat([torch.ones(N_splats_at_init, dtype=torch.bool),
                                    torch.zeros(N_splats_after_init-N_splats_at_init, dtype=torch.bool)],
                                axis=0)
                # self.GS.gaussians.prune_points(mask)
        with torch.no_grad():
            gaussians =  self.gaussians
            gaussians._scaling =  gaussians.scaling_inverse_activation(gaussians.scaling_activation(gaussians._scaling)*0.5)
        return visualization_dict
    

    def prune(self, radii, min_opacity=0.005):
        self.GS.gaussians.tmp_radii = radii
        if self.gs_step < self.training_config.densify_until_iter:
            prune_mask = (self.GS.gaussians.get_opacity < min_opacity).squeeze()
            if hasattr(self.GS.gaussians, "mr_list") and self.GS.gaussians.mr_list is not None:
                self.GS.gaussians.mr_list = self.GS.gaussians.mr_list[~prune_mask]
            self.GS.gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
        self.GS.gaussians.tmp_radii = None


    def render_set(self, scene, pipeline, training_step):
        gaussians, views = scene.gaussians, self.scene.getTestCameras()

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = f"{scene.model_path}/{training_step}/renders"
        gts_path = f"{scene.model_path}/{training_step}/gt"
        unc_path = f"{scene.model_path}/{training_step}/unc"
        raw_path = f"{scene.model_path}/{training_step}/raw"

        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(unc_path, exist_ok=True)

        means = []
        gts = []
        stds = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            gt = view.original_image[0:3, :, :]
            out = forward_k_times(view, gaussians, pipeline, background)
            mean = out['comp_rgb'].detach()
            rgbs = out['comp_rgbs'].detach()
            std = out['comp_std'].detach()
            depths = out['depths'].detach()

            # mae = ((mean - gt)).abs()

            # ause_mae, ause_err_mae, ause_err_by_var_mae = ause_br(std.reshape(-1), mae.reshape(-1), err_type='mae')
            # mean_nll = nll_kernel_density(rgbs.permute(1,2,3,0), std, gt)

            # psnr_all += psnr(mean, gt).mean().item()
            # ssim_all += ssim(mean, gt).mean().item()
            # lpips_all += lpips(mean, gt, net_type="vgg").mean().item()

            # ause_mae_all += ause_mae.item()
            # mean_nll_all += mean_nll.item()

            # if eval_depth: 
            #     depths = depths * scene.depth_scale

            #     depth = depths.mean(dim=0)
            #     depth_std = depths.std(dim=0)
            #     depth_gt = view.depth

            #     depth_mae = ((depth - depth_gt)).abs()
            #     depth_ause_mae, depth_ause_err_mae, depth_ause_err_by_var_mae = ause_br(depth_std.reshape(-1), depth_mae.reshape(-1), err_type='mae')
            #     depth_ause_mae_all += depth_ause_mae


            unc_vis_multiply = 10
            if idx == 0:
                print(f"@@@@@@@@@@@@@ std min {std.min().item()} max {std.max().item()} mean {std.mean().item()} @@@@@@@@@@@@@")
                torchvision.utils.save_image(mean, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(unc_vis_multiply*std, os.path.join(unc_path, '{0:05d}'.format(idx) + ".png"))
                # norm_std = (std - std.min()) / (std.max() - std.min() + 1e-8)
                norm_std = std

                wandb.log({f"rendered_images/mean_{idx}": wandb.Image(mean.cpu(), caption=f"Rendered Mean {idx}")}, step=training_step)
                wandb.log({f"rendered_images/gt_{idx}": wandb.Image(gt.cpu(), caption=f"Ground Truth {idx}")}, step=training_step)
                wandb.log({f"rendered_images/std_{idx}": wandb.Image(norm_std.cpu(), caption=f"Rendered Std {idx}")}, step=training_step)

                

            # np.save(os.path.join(raw_path, '{0:05d}_mean.npy'.format(idx)), mean.cpu().numpy())
            # np.save(os.path.join(raw_path, '{0:05d}_std.npy'.format(idx)), std.cpu().numpy())
            # np.save(os.path.join(raw_path, '{0:05d}_rgbs.npy'.format(idx)), rgbs.cpu().numpy())
            # np.save(os.path.join(raw_path, '{0:05d}_depths.npy'.format(idx)), depths.cpu().numpy())

    

        # psnr_all /= len(views)
        # ause_mae_all /= len(views)
        # mean_nll_all /= len(views)
        # ssim_all /= len(views)
        # lpips_all /= len(views)

        # depth_ause_mae_all /= len(views)

        # csv_file = f"output/eval_results_{dataset.dataset_name}.csv"
        # with open(csv_file, mode='a', newline='') as file:
        #     # writer = csv.writer(file)

        #     if eval_depth: 
        #         results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all} Depth AUSE {depth_ause_mae_all}"
        #         print(results)
        #         writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all, depth_ause_mae_all])
        #     else: 
        #         results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all}"
        #         print(results)
        #         writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all])

