import json
import math
import os
import time
from typing import Tuple
import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser

from gaussian_splatting.scene.gaussian_model import GaussianModel2D
from gaussian_splatting.utils.colmap import Dataset, Parser
from gaussian_splatting.utils.traj import generate_interpolated_path
from gaussian_splatting.utils.general_utils import set_random_seed
from gaussian_splatting.utils.camera_utils import CameraOptModule
from gaussian_splatting.utils.graphics_utils import colormap
from gaussian_splatting.gaussian_renderer import render_2d
from configs.base_config import Config
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)
        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        self.gaussians = GaussianModel2D(sh_degree=self.cfg.sh_degree, config=self.cfg)
        self.gaussians.initialize_from_pcd(parser=self.parser)
        self.gaussians.init_lr(spatial_lr_scale=5)
        self.gaussians.training_setup()
        print("Model initialized. Number of GS:", self.gaussians.means.shape[0])
        self.model_type = cfg.model_type

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            self.gaussians.xyz_scheduler
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                if data.get("points"):
                    points = data["points"].to(device)  # [1, M, 2]
                else: 
                    points = None 
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            self.gaussians.scheduleSHdegree(step=step)

            # forward
            render_pkg = render_2d(
                cfg=cfg, 
                pc=self.gaussians,
                world2cam=torch.linalg.inv(camtoworlds), 
                Ks=Ks, 
                width=width, height=height, 
                sh_degree=self.gaussians.active_sh_degree
            )
            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = (
                render_pkg["colors"], 
                render_pkg["alphas"], 
                render_pkg["normals"], 
                render_pkg["normals_from_depth"], 
                render_pkg["distort"], 
                render_pkg["median_depth"], 
                render_pkg["info"]
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.gaussians.strategy.step_pre_backward(
                params=self.gaussians.splats,
                optimizers=self.gaussians.optimizers,
                state=self.gaussians.strategy_state,
                step=step,
                info=info,
            )
            masks = data["mask"].to(device) if "mask" in data else None
            if masks is not None:
                pixels = pixels * masks[..., None]
                colors = colors * masks[..., None]

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                if points is None:
                    # resize it to the render-depth shape
                    (H, W) = depths.shape[1:-1]
                    if (H, W) != depths_gt.shape[-2:]:
                        depths_gt = F.interpolate(
                            depths_gt.unsqueeze(0),
                            size=(H, W),      # (H_r, W_r)
                            mode="bilinear",                      # or "bilinear" if you prefer
                            align_corners=False
                        ).squeeze()
                    # calculate loss in disparity space
                    disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths)).squeeze()
                    # disp_gt = 1.0 / depths_gt  # [1, M]
                    disp_gt = torch.where(depths_gt > 0.0, 1 / depths_gt, torch.zeros_like(depths_gt))
                    # print("disp: ", disp.shape)
                    # print("disp_gt: ", disp_gt.shape)
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss += depthloss * cfg.depth_lambda                    
                else:
                    # query depths from depth map
                    points = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points.unsqueeze(2)  # [1, M, 1, 2]
                    depths = F.grid_sample(
                        depths.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    depths = depths.squeeze(3).squeeze(1)  # [1, M]
                    # calculate loss in disparity space
                    disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                    disp_gt = 1.0 / depths_gt  # [1, M]
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={self.gaussians.active_sh_degree}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.normal_loss:
                desc += f"normal loss={normalloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.gaussians.means), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            self.gaussians.strategy.step_post_backward(
                params=self.gaussians.splats,
                optimizers=self.gaussians.optimizers,
                state=self.gaussians.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.gaussians.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.gaussians.means),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.gaussians.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            # forward
            render_pkg = render_2d(
                cfg=cfg, 
                pc=self.gaussians,
                world2cam=torch.linalg.inv(camtoworlds), 
                Ks=Ks, 
                width=width, height=height, 
                sh_degree=self.gaussians.active_sh_degree
            )
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
            ) = (
                render_pkg["colors"], 
                render_pkg["alphas"], 
                render_pkg["normals"], 
                render_pkg["normals_from_depth"], 
                render_pkg["distort"], 
                render_pkg["median_depth"], 
                render_pkg["info"]
            )
            colors = torch.clamp(colors, 0.0, 1.0)
            colors = colors[..., :3]  # Take RGB channels
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            # write median depths
            render_median = (render_median - render_median.min()) / (
                render_median.max() - render_median.min()
            )
            # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            render_median = (
                render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
                (render_median * 255).astype(np.uint8),
            )

            # write normals
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
            )

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                np.max(normals_from_depth) - np.min(normals_from_depth)
            )
            normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            if len(normals_from_depth_output.shape) == 4:
                normals_from_depth_output = normals_from_depth_output.squeeze(0)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                normals_from_depth_output,
            )

            # write distortions

            render_dist = render_distort
            dist_max = torch.max(render_dist)
            dist_min = torch.min(render_dist)
            render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            render_dist = (
                colormap(render_dist.cpu().numpy()[0])
                .permute((1, 2, 0))
                .numpy()
                .astype(np.uint8)
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_distortions_{step}.png", render_dist
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.gaussians.means)}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.gaussians.means),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds#[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 12)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            # forward
            render_pkg = render_2d(
                cfg=cfg, 
                pc=self.gaussians,
                world2cam=torch.linalg.inv(camtoworlds[i : i + 1]), 
                Ks=K[None], 
                width=width, height=height, 
                sh_degree=self.gaussians.active_sh_degree
            )
            renders, surf_normals = render_pkg["colors"], render_pkg["normals_from_depth"]

            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=1 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=10)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        # forward
        render_pkg = render_2d(
            cfg=cfg, 
            pc=self.gaussians,
            world2cam=torch.linalg.inv(c2w)[None], 
            Ks=K[None], 
            width=W, height=H, 
            sh_degree=self.gaussians.active_sh_degree,
            render_mode="RGB"
        )
        render_colors = render_pkg["colors"]
        return render_colors[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.gaussians.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
