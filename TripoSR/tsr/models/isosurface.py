from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torchmcubes import marching_cubes as _torch_marching_cubes
except Exception:
    _torch_marching_cubes = None

try:
    import mcubes
except Exception:
    mcubes = None

try:
    from skimage import measure as sk_measure
except Exception:
    sk_measure = None


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        if _torch_marching_cubes is not None:
            self.mc_backend = "torchmcubes"
            self.mc_func: Callable = _torch_marching_cubes
        elif mcubes is not None:
            self.mc_backend = "mcubes"
            self.mc_func = mcubes.marching_cubes
        elif sk_measure is not None:
            self.mc_backend = "skimage"
            self.mc_func = sk_measure.marching_cubes
        else:
            raise ImportError(
                "No marching cubes backend found. Install torchmcubes, PyMCubes, or scikit-image."
            )
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        if self.mc_backend == "torchmcubes":
            try:
                v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
            except AttributeError:
                print("torchmcubes has no CUDA support; using CPU backend.")
                v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
        elif self.mc_backend == "mcubes":
            v_np, f_np = self.mc_func(level.detach().cpu().numpy(), 0.0)
            v_pos = torch.from_numpy(v_np.astype(np.float32))
            t_pos_idx = torch.from_numpy(f_np.astype(np.int64))
        else:
            verts, faces, _, _ = self.mc_func(
                level.detach().cpu().numpy(),
                level=0.0,
                allow_degenerate=False,
            )
            v_pos = torch.from_numpy(verts.astype(np.float32))
            t_pos_idx = torch.from_numpy(faces.astype(np.int64))
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
