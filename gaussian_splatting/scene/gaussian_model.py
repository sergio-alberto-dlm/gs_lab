import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from gsplat import DefaultStrategy 

from gaussian_splatting.utils.general_utils import (
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
)
from gaussian_splatting.utils.general_utils import knn
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p
from gaussian_splatting.utils.colmap import Parser


class GaussianModel2D:
    def __init__(self, sh_degree: int, device: str = "cuda", config: dict = None):
        self.device = device
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.sh_degree_interval = config.sh_degree_interval
        self.config = config 

        # Splats store all trainable gaussian properties
        init_empty = lambda: torch.empty((0,), device=self.device)
        self.splats = nn.ParameterDict({
            key: nn.Parameter(init_empty())
            for key in ["means", "scales", "quats", "opacities", "sh0", "shN"]
        })

        # Activations for parameter transforms
        self._activations = {
            "scale": torch.exp,
            "inv_scale": torch.log,
            "opacity": torch.sigmoid,
            "inv_opacity": inverse_sigmoid,
            "rotation": nn.functional.normalize,
        }

        self.optimizers = None
        self.ply_input = None
        self.strategy = None
        self.strategy_state = None 

    @property
    def means(self):
        return self.splats["means"]

    @property
    def scales(self):
        return self._activations["scale"](self.splats["scales"])

    @property
    def quats(self):
        return self._activations["rotation"](self.splats["quats"], dim=-1)

    @property
    def opacities(self):
        return self._activations["opacity"](self.splats["opacities"])

    @property
    def features(self):
        sh0 = self.splats["sh0"]
        shN = self.splats["shN"]
        return torch.cat([sh0, shN], dim=1)

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    # def update_learning_rate(self, iteration):
    #     # ----> verify the means lr scheduling within gsplat context 
    #     """Learning rate scheduling per step"""
    #     for param_group in self.optimizer.param_groups:
    #         if param_group["name"] == "means":
    #             # lr = self.xyz_scheduler_args(iteration)
    #             lr = helper(
    #                 iteration,
    #                 lr_init=self.lr_init,
    #                 lr_final=self.lr_final,
    #                 lr_delay_mult=self.lr_delay_mult,
    #                 max_steps=self.max_steps,
    #             )

    #             param_group["lr"] = lr
    #             return lr

    def scheduleSHdegree(self, step):
        # sh schedule
        self.active_sh_degree = min(step // self.sh_degree_interval, self.max_sh_degree)

    def initialize_from_pcd(self, parser: Parser, init_scale=1.0, init_opacity=0.1):
        pts = torch.from_numpy(parser.points).float().to(self.device)
        cols = torch.from_numpy(parser.points_rgb / 255).float().to(self.device)
        sh_coeffs = RGB2SH(cols)

        N = pts.shape[0]
        features = torch.zeros((N, (self.max_sh_degree+1)**2, 3), device=self.device)
        features[:,0,:] = sh_coeffs

        # scales by average neighbor distance
        d2 = knn(pts, 4)[:,1:]**2
        avg = torch.sqrt(d2.mean(dim=-1, keepdim=True)) * init_scale
        scales = torch.log(avg).repeat(1,3)

        quats = torch.rand((N,4), device=self.device)
        opac = torch.logit(torch.full((N,), init_opacity, device=self.device))

        # assign parameters
        self.splats["means"].data = pts
        self.splats["scales"].data = scales
        self.splats["quats"].data = quats
        self.splats["opacities"].data = opac

        # split SH features
        self.splats["sh0"].data = features[:,:1,:]
        self.splats["shN"].data = features[:,1:,:]

    def construct_list_of_attributes(self):
        attrs = ["x","y","z","nx","ny","nz"]
        f0, fN = self.splats["sh0"], self.splats["shN"]
        for i in range(f0.shape[1]*f0.shape[2]): attrs.append(f"f_dc_{i}")
        for i in range(fN.shape[1]*fN.shape[2]): attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(self.splats["scales"].shape[1]): attrs.append(f"scale_{i}")
        for i in range(self.splats["quats"].shape[1]): attrs.append(f"rot_{i}")
        return attrs

    def save_ply(self, path: str):
        mkdir_p(os.path.dirname(path))
        xyz = self.splats["means"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        sh0 = self.splats["sh0"].detach().transpose(1,2).flatten(start_dim=1).cpu().numpy()
        shN = self.splats["shN"].detach().transpose(1,2).flatten(start_dim=1).cpu().numpy()
        opac = self.splats["opacities"].detach().cpu().numpy()[...,np.newaxis]
        scale = self.splats["scales"].detach().cpu().numpy()
        quat = self.splats["quats"].detach().cpu().numpy()
        dtype = [(a,'f4') for a in self.construct_list_of_attributes()]
        el_arr = np.empty(xyz.shape[0], dtype=dtype)
        data = np.concatenate([xyz,normals,sh0,shN,opac,scale,quat],axis=1)
        el_arr[:] = [tuple(r) for r in data]
        PlyData([PlyElement.describe(el_arr,'vertex')]).write(path)

    def load_ply(self, path: str):
        ply = PlyData.read(path)['vertex']
        pts = np.vstack([ply['x'],ply['y'],ply['z']]).T
        norms = np.vstack([ply['nx'],ply['ny'],ply['nz']]).T
        self.ply_input = BasicPointCloud(points=pts,colors=np.ones_like(pts),normals=norms)
        xyz = np.stack([ply['x'],ply['y'],ply['z']],axis=1)
        opac = np.asarray(ply['opacity'])[...,None]
        dc_count = self.splats['sh0'].shape[1]*self.splats['sh0'].shape[2]
        f0 = np.stack([ply[f'f_dc_{i}'] for i in range(dc_count)],axis=1)
        f0 = f0.reshape(-1,self.splats['sh0'].shape[1],self.splats['sh0'].shape[2])
        rest_count = self.splats['shN'].shape[1]*self.splats['shN'].shape[2]
        fN = np.stack([ply[f'f_rest_{i}'] for i in range(rest_count)],axis=1)
        fN = fN.reshape(-1,self.splats['shN'].shape[1],self.splats['shN'].shape[2])
        sc = np.stack([ply[f'scale_{i}'] for i in range(self.splats['scales'].shape[1])],axis=1)
        qt = np.stack([ply[f'rot_{i}'] for i in range(self.splats['quats'].shape[1])],axis=1)
        self.splats['means']= nn.Parameter(torch.tensor(xyz,device=self.device))
        self.splats['sh0']  = nn.Parameter(torch.tensor(f0,device=self.device))
        self.splats['shN']  = nn.Parameter(torch.tensor(fN,device=self.device))
        self.splats['opacities']= nn.Parameter(torch.tensor(opac,device=self.device))
        self.splats['scales']= nn.Parameter(torch.tensor(sc,device=self.device))
        self.splats['quats']= nn.Parameter(torch.tensor(qt,device=self.device))
        self.unique_kf_ids = torch.zeros((xyz.shape[0],),dtype=torch.int,device=self.device)
        self.active_sh_degree = self.max_sh_degree

    def training_setup(self):
        # set up optimizers for each splat parameter
        names_lr = [
            ("means",     self.config.position_lr_init * self.spatial_lr_scale),
            ("scales",    self.config.scaling_lr       * self.spatial_lr_scale),
            ("quats",     self.config.rotation_lr),
            ("opacities", self.config.opacity_lr),
            ("sh0",       self.config.feature_lr),
            ("shN",       self.config.feature_lr/20.0),
        ]
        self.optimizers = {
            name: torch.optim.Adam([
                {"params": self.splats[name], "lr": lr, "name": name}
            ], eps=1e-15, betas=(0.9,0.999))
            for name, lr in names_lr
        }
        # means has a learning rate schedule, that end at 0.01 of the initial value
        self.xyz_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / self.config.max_steps)
            )
        # Densification strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            revised_opacity=False,
            prune_opa=self.config.prune_opa,
            grow_grad2d=self.config.grow_grad2d,
            grow_scale3d=self.config.grow_scale3d,
            prune_scale3d=self.config.prune_scale3d,
            refine_start_iter=self.config.refine_start_iter,
            refine_stop_iter=self.config.refine_stop_iter,
            reset_every=self.config.reset_every,
            refine_every=self.config.refine_every,
            absgrad=self.config.absgrad,
            key_for_gradient="means2d",
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

    def clean_gaussians(self):
        self.splats["means"] = torch.empty(0, device="cuda")
        self.splats["sh0"] = torch.empty(0, device="cuda")
        self.splats["shN"] = torch.empty(0, device="cuda")
        self.splats["opacities"] = torch.empty(0, device="cuda")
        self.splats["scales"] = torch.empty(0, device="cuda")
        self.splats["quats"] = torch.empty(0, device="cuda")
        self.optimizers = None
        
    def load_checkpoint(self, ckpt_path, map_location=None):
        """
        Load a checkpoint and restore the splats parameters.

        Args:
            ckpt_path (str): Path to the .pt checkpoint file.
            map_location (str or torch.device, optional): 
                Where to remap storages. Defaults to model.device.
        Returns:
            int or None: The training step at which the checkpoint was saved.
        """
        # decide where to load tensors
        if map_location is None:
            map_location = self.device

        # load the checkpoint
        ckpt = torch.load(ckpt_path, map_location=map_location)

        # restore the splats state_dict
        self.splats.load_state_dict(ckpt["splats"])

        # if you scheduled SH degree based on step, update it
        step = ckpt.get("step", None)
        if step is not None:
            self.scheduleSHdegree(step)

        print(f"✅ Loaded splats from '{ckpt_path}' at step {step}")
        return step
    