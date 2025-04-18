import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Literal

@dataclass
class Config:
    disable_viewer: bool = False
    ckpt: Optional[str] = None
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results/garden"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    port: int = 8080
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30000
    eval_steps: List[int] = field(default_factory=lambda: [7000, 30000])
    save_steps: List[int] = field(default_factory=lambda: [7000, 30000])
    init_type: str = "sfm"
    init_num_pts: int = 100000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    near_plane: float = 0.2
    far_plane: float = 200
    prune_opa: float = 0.05
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    reset_every: int = 3000
    refine_every: int = 100
    packed: bool = False
    sparse_grad: bool = False
    absgrad: bool = False
    antialiased: bool = False
    revised_opacity: bool = False
    random_bkgd: bool = False
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6
    depth_loss: bool = False
    depth_lambda: float = 1e-2
    normal_loss: bool = False
    normal_lambda: float = 5e-2
    normal_start_iter: int = 7000
    dist_loss: bool = False
    dist_lambda: float = 1e-2
    dist_start_iter: int = 3000
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"
    tb_every: int = 100
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)

def load_config_from_yaml(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)