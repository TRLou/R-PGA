from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch

from .graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


@dataclass
class CameraSimple:
	image_width: int
	image_height: int
	FoVx: float
	FoVy: float
	world_view_transform: torch.Tensor
	full_proj_transform: torch.Tensor
	camera_center: torch.Tensor
	znear: float = 0.01
	zfar: float = 100.0

	@staticmethod
	def from_json_record(rec: Dict[str, Any]) -> "CameraSimple":
		width = int(rec["width"]) 
		height = int(rec["height"]) 
		fx = float(rec["fx"]) 
		fy = float(rec["fy"]) 
		FoVx = focal2fov(fx, width)
		FoVy = focal2fov(fy, height)
		R = np.array(rec["rotation"], dtype=np.float32)
		T = np.array(rec["position"], dtype=np.float32)

		w2v = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device="cuda").transpose(0, 1)
		proj = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).transpose(0, 1).cuda()
		# GIR/3DGS pipeline expects separate view and projection matrices
		# Pass projection as-is; do NOT pre-multiply with view.
		full = proj
		cam_center = torch.inverse(w2v)[3, :3]

		return CameraSimple(
			image_width=width,
			image_height=height,
			FoVx=FoVx,
			FoVy=FoVy,
			world_view_transform=w2v,
			full_proj_transform=full,
			camera_center=cam_center,
		)


