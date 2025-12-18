from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from plyfile import PlyData

from .general_utils import build_scaling_rotation, strip_symmetric


class GaussianModelSimple(nn.Module):
	"""
	精简版 3DGS 模型：仅包含几何（xyz/scale/rotation/opacity）与 SH 系数。
	用于渲染-检测-攻击闭环中的可微更新（这里只更新 DC 项）。
	"""

	def __init__(self, max_sh_degree: int = 3) -> None:
		super().__init__()
		self.max_sh_degree = max_sh_degree
		self.active_sh_degree = max_sh_degree
		self._xyz = nn.Parameter(torch.empty(0, 3, device="cuda"))
		self._features_dc = nn.Parameter(torch.empty(0, 3, 1, device="cuda"))
		self._features_rest = nn.Parameter(torch.empty(0, 3, (max_sh_degree + 1) ** 2 - 1, device="cuda"))
		self._scaling = nn.Parameter(torch.empty(0, 3, device="cuda"))
		self._rotation = nn.Parameter(torch.empty(0, 4, device="cuda"))
		self._opacity = nn.Parameter(torch.empty(0, 1, device="cuda"))

	@property
	def get_xyz(self) -> torch.Tensor:
		return self._xyz

	@property
	def get_scaling(self) -> torch.Tensor:
		return torch.exp(self._scaling)

	@property
	def get_rotation(self) -> torch.Tensor:
		return torch.nn.functional.normalize(self._rotation, dim=-1)

	@property
	def get_opacity(self) -> torch.Tensor:
		return torch.sigmoid(self._opacity)

	@property
	def get_features(self) -> torch.Tensor:
		# (P, 1, 3) + (P, coeff-1, 3) -> (P, coeff, 3)
		return torch.cat([self._features_dc, self._features_rest], dim=1)

	def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
		L = build_scaling_rotation(scaling_modifier * self.get_scaling, self.get_rotation)
		return strip_symmetric(L @ L.transpose(1, 2))

	def freeze_all_but_dc(self) -> None:
		for p in [self._xyz, self._features_rest, self._opacity, self._scaling, self._rotation]:
			p.requires_grad_(False)
		self._features_dc.requires_grad_(True)

	def load_ply(self, ply_path: str) -> None:
		assert os.path.isfile(ply_path), f"PLY not found: {ply_path}"
		plydata = PlyData.read(ply_path)

		xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
					  np.asarray(plydata.elements[0]["y"]),
					  np.asarray(plydata.elements[0]["z"])), axis=1)
		opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

		features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
		features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
		features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
		features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

		extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
		extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
		features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
		for idx, attr_name in enumerate(extra_f_names):
			features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
		features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

		scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
		scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
		scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
		for idx, attr_name in enumerate(scale_names):
			scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

		rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
		rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
		rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
		for idx, attr_name in enumerate(rot_names):
			rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

		# 推断 SH 阶数以对齐 active_sh_degree
		coeff_per_channel = features_extra.shape[2] + 1
		sh_deg = int(np.sqrt(coeff_per_channel) - 1)
		self.max_sh_degree = sh_deg
		self.active_sh_degree = sh_deg

		self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda"), requires_grad=False)
		self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous())
		self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous(), requires_grad=False)
		self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda"), requires_grad=False)
		self._scaling = nn.Parameter(torch.log(torch.tensor(scales, dtype=torch.float32, device="cuda")), requires_grad=False)
		self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda"), requires_grad=False)

	def save_ply(self, path: str) -> None:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		from plyfile import PlyElement, PlyData  # local import to avoid global dep if unused
		xyz = self._xyz.detach().cpu().numpy()
		f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
		f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
		opacities = self._opacity.detach().cpu().numpy()
		scale = self._scaling.detach().cpu().numpy()
		rotation = self._rotation.detach().cpu().numpy()

		# 法线用于可视化，这里使用单位法线占位（不影响 3DGS 渲染）
		normals = np.zeros_like(xyz, dtype=np.float32)

		def attributes_list() -> list[str]:
			l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
			for i in range(f_dc.shape[1]):
				l.append(f'f_dc_{i}')
			for i in range(f_rest.shape[1]):
				l.append(f'f_rest_{i}')
			l.append('opacity')
			for i in range(scale.shape[1]):
				l.append(f'scale_{i}')
			for i in range(rotation.shape[1]):
				l.append(f'rot_{i}')
			return l

		dtype_full = [(attribute, 'f4') for attribute in attributes_list()]
		elements = np.empty(xyz.shape[0], dtype=dtype_full)
		attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
		elements[:] = list(map(tuple, attributes))
		el = PlyElement.describe(elements, 'vertex')
		PlyData([el]).write(path)


