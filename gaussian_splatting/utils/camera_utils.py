import struct 

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    # Convert all tensors to numpy arrays in one go
    print(f"Saving ply to {dir}")
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    means = numpy_data["means"]
    scales = numpy_data["scales"]
    quats = numpy_data["quats"]
    opacities = numpy_data["opacities"]

    sh0 = numpy_data["sh0"].transpose(0, 2, 1).reshape(means.shape[0], -1)
    shN = numpy_data["shN"].transpose(0, 2, 1).reshape(means.shape[0], -1)

    # Create a mask to identify rows with NaN or Inf in any of the numpy_data arrays
    invalid_mask = (
        np.isnan(means).any(axis=1)
        | np.isinf(means).any(axis=1)
        | np.isnan(scales).any(axis=1)
        | np.isinf(scales).any(axis=1)
        | np.isnan(quats).any(axis=1)
        | np.isinf(quats).any(axis=1)
        | np.isnan(opacities).any(axis=0)
        | np.isinf(opacities).any(axis=0)
        | np.isnan(sh0).any(axis=1)
        | np.isinf(sh0).any(axis=1)
        | np.isnan(shN).any(axis=1)
        | np.isinf(shN).any(axis=1)
    )

    # Filter out rows with NaNs or Infs from all data arrays
    means = means[~invalid_mask]
    scales = scales[~invalid_mask]
    quats = quats[~invalid_mask]
    opacities = opacities[~invalid_mask]
    sh0 = sh0[~invalid_mask]
    shN = shN[~invalid_mask]

    num_points = means.shape[0]

    with open(dir, "wb") as f:
        # Write PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")

        if colors is not None:
            for j in range(colors.shape[1]):
                f.write(f"property float f_dc_{j}\n".encode())
        else:
            for i, data in enumerate([sh0, shN]):
                prefix = "f_dc" if i == 0 else "f_rest"
                for j in range(data.shape[1]):
                    f.write(f"property float {prefix}_{j}\n".encode())

        f.write(b"property float opacity\n")

        for i in range(scales.shape[1]):
            f.write(f"property float scale_{i}\n".encode())
        for i in range(quats.shape[1]):
            f.write(f"property float rot_{i}\n".encode())

        f.write(b"end_header\n")

        # Write vertex data
        for i in range(num_points):
            f.write(struct.pack("<fff", *means[i]))  # x, y, z
            f.write(struct.pack("<fff", 0, 0, 0))  # nx, ny, nz (zeros)

            if colors is not None:
                color = colors.detach().cpu().numpy()
                for j in range(color.shape[1]):
                    f_dc = (color[i, j] - 0.5) / 0.2820947917738781
                    f.write(struct.pack("<f", f_dc))
            else:
                for data in [sh0, shN]:
                    for j in range(data.shape[1]):
                        f.write(struct.pack("<f", data[i, j]))

            f.write(struct.pack("<f", opacities[i]))  # opacity

            for data in [scales, quats]:
                for j in range(data.shape[1]):
                    f.write(struct.pack("<f", data[i, j]))
