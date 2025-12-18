from __future__ import annotations

import math
from typing import Dict, Any

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .gaussian_model_pbr import GaussianModelPBR
from .cameras_simple import CameraSimple


class PipelineParams:
	def __init__(self, compute_cov3D_python: bool = False, debug: bool = False) -> None:
		self.compute_cov3D_python = compute_cov3D_python
		self.debug = debug


def render_pbr(camera: CameraSimple, pc: GaussianModelPBR, pipe: PipelineParams, bg_color: torch.Tensor,
			scaling_modifier: float = 1.0, iteration: int = 60001, is_train: bool = True,
			first_stage_step: int = 5000, second_stage_step: int = 30000, remove_noise: bool = False, hdr_rotation: bool = False) -> Dict[str, Any]:
	# screenspace grad holder
	screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device) + 0

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

	renderer = GaussianRasterizer(raster_settings=raster_settings)

	means3D = pc.get_xyz
	means2D = screenspace_points
	opacity = pc.get_opacity

	if iteration <= first_stage_step:
		colors_precomp = pc.get_albedo_init
	else:
		# colors from PBR compute_color (colors_precomp)
		result = pc.compute_color(camera.camera_center, iteration, is_train, first_stage_step, second_stage_step, remove_noise, hdr_rotation, 0.0, force_color_sh_only=False)
		colors_precomp = result["color"]

	scales = None
	rotations = None
	cov3D = None
	if pipe.compute_cov3D_python:
		cov3D = pc.get_covariance(scaling_modifier)
	else:
		scales = pc.get_scaling
		rotations = pc.get_rotation

	rendered_image, radii, depth, alpha = renderer(
		means3D=means3D,
		means2D=means2D,
		shs=None,
		colors_precomp=colors_precomp,
		opacities=opacity,
		scales=scales,
		rotations=rotations,
		cov3D_precomp=cov3D,
	)

	return {
		"render": rendered_image,
		"depth": depth,
		"alpha": alpha,
		"viewspace_points": screenspace_points,
		"visibility_filter": radii > 0,
		"radii": radii,
	}


