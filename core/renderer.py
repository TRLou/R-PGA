from __future__ import annotations

import math
from typing import Dict, Any

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .gaussian_model_simple import GaussianModelSimple
from .cameras_simple import CameraSimple


class PipelineParams:
	def __init__(self, compute_cov3D_python: bool = False, debug: bool = False) -> None:
		self.compute_cov3D_python = compute_cov3D_python
		self.debug = debug


@torch.no_grad()
def _zeros_like_with_grad(t: torch.Tensor) -> torch.Tensor:
	# we need grad on screenspace points during forward for visibility mask; but we won't backprop through it here
	return torch.zeros_like(t, dtype=t.dtype, device=t.device, requires_grad=True) + 0


def render_SH(camera: CameraSimple, pc: GaussianModelSimple, pipe: PipelineParams, bg_color: torch.Tensor,
			scaling_modifier: float = 1.0) -> Dict[str, Any]:
	# screenspace points holder for gradient of 2D projection
	screenspace_points = _zeros_like_with_grad(pc.get_xyz)

	tanfovx = math.tan(camera.FoVx * 0.5)
	tanfovy = math.tan(camera.FoVy * 0.5)

	raster_settings = GaussianRasterizationSettings(
		image_height=int(camera.image_height),
		image_width=int(camera.image_width),
		tanfovx=tanfovx,
		tanfovy=tanfovy,
		bg=bg_color,
		scale_modifier=scaling_modifier,
		viewmatrix=camera.world_view_transform,
		projmatrix=camera.full_proj_transform,
		sh_degree=pc.active_sh_degree,
		campos=camera.camera_center,
		prefiltered=False,
		debug=pipe.debug,
	)

	rasterizer = GaussianRasterizer(raster_settings=raster_settings)

	means3D = pc.get_xyz
	means2D = screenspace_points
	opacity = pc.get_opacity

	shs = pc.get_features  # (P, 3, (deg+1)^2)
	scales = None
	rotations = None
	cov3D_precomp = None
	if pipe.compute_cov3D_python:
		cov3D_precomp = pc.get_covariance(scaling_modifier)
	else:
		scales = pc.get_scaling
		rotations = pc.get_rotation

	rendered_image, radii, depth, alpha = rasterizer(
		means3D=means3D,
		means2D=means2D,
		shs=shs,
		colors_precomp=None,
		opacities=opacity,
		scales=scales,
		rotations=rotations,
		cov3D_precomp=cov3D_precomp,
	)

	return {
		"render": rendered_image,  # (3, H, W) in [0,1]
		"depth": depth,
		"alpha": alpha,
		"viewspace_points": screenspace_points,
		"visibility_filter": radii > 0,
		"radii": radii,
	}


