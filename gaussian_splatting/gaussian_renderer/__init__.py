from gaussian_splatting.scene.gaussian_model import GaussianModel2D
from torch import Tensor
from typing import Dict, Tuple
from gsplat.rendering import rasterization_2dgs
import torch 

def render_2d(
        cfg,
        pc: GaussianModel2D,
        world2cam: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        near_plane=0.01,
        far_plane=100,
        render_mode="RGB+ED",
        distloss=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = pc.means  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = pc.quats  # [N, 4]
        scales = pc.scales  # [N, 3]
        opacities = pc.opacities  # [N,]
        colors = pc.features  # [N, K, 3]
        # colors = torch.cat([pc.splats["sh0"], pc.splats["shN"]], 1)  # [N, K, 3]

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities.squeeze(),
            colors=colors,
            viewmats=world2cam,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=cfg.packed, 
            absgrad=cfg.absgrad,
            sparse_grad=cfg.sparse_grad,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
            distloss=distloss,
        )

        return {
            "colors": render_colors,
            "alphas": render_alphas,
            "normals": render_normals,
            "normals_from_depth": normals_from_depth,
            "distort": render_distort,
            "median_depth": render_median,
            "info": info,
        }