from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch import nn

try:
	import nvdiffrast.torch as dr
except Exception:
	dr = None

from .general_utils import build_scaling_rotation, strip_symmetric
from .ir_utils import linear_to_srgb, sample_diffuse_directions, dot
from .sh_utils import RGB2SH


class GaussianModelPBR(nn.Module):
	"""
	PBR 版 3DGS 模型：包含几何、SH 特征、不透明度、材质（albedo/metallic/roughness），
	带有 GIR 风格的 compute_color()，用于 PBR 渲染。
	仅将 albedo 作为优化变量，其它参数默认冻结。
	"""

	def __init__(self, sh_degree: int = 3, environment_texture: str | None = None, environment_scale: float = 1.0) -> None:
		super().__init__()
		self.max_sh_degree = sh_degree
		self.active_sh_degree = sh_degree
		self._xyz = nn.Parameter(torch.empty(0, 3, device="cuda"), requires_grad=False)
		self._features_dc = nn.Parameter(torch.empty(0, 1, 3, device="cuda"), requires_grad=False)
		self._features_rest = nn.Parameter(torch.empty(0, (sh_degree + 1) ** 2 - 1, 3, device="cuda"), requires_grad=False)
		self._scaling = nn.Parameter(torch.empty(0, 3, device="cuda"), requires_grad=False)
		self._rotation = nn.Parameter(torch.empty(0, 4, device="cuda"), requires_grad=False)
		self._opacity = nn.Parameter(torch.empty(0, 1, device="cuda"), requires_grad=False)

		self._albedo_init = nn.Parameter(torch.empty(0, 3, device="cuda"), requires_grad=True)
		self._metallic_init = nn.Parameter(torch.empty(0, 1, device="cuda"), requires_grad=False)
		self._roughness_init = nn.Parameter(torch.empty(0, 1, device="cuda"), requires_grad=False)

		self.max_radii2D = torch.empty(0, device="cuda")

		# FG LUT
		fg_lut_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'load', 'lights', 'bsdf_256_256.bin'))
		self.FG_LUT = torch.from_numpy(
			np.fromfile(fg_lut_path, dtype=np.float32).reshape(1, 256, 256, 2)
		).cuda()

		# 环境光（若缺失则回退为常量环境）
		try:
			import envlight  # type: ignore
			self.envlight = envlight.EnvLight(environment_texture, scale=environment_scale, min_res=16, max_res=512, min_roughness=0.08, max_roughness=0.5, trainable=True).cuda()
		except Exception:
			class _DummyEnvLight(torch.nn.Module):
				def __init__(self):
					super().__init__()
				def build_base(self):
					return
				def build_mips(self):
					return
				def forward(self, dirs: torch.Tensor, roughness: torch.Tensor | None = None):
					return torch.ones((*dirs.shape[:-1], 3), device=dirs.device) * 0.5
			self.envlight = _DummyEnvLight().cuda()

		# 预采样漫反射方向
		self.diffuse_sample_num = 128
		az, el = self.sample_sphere(self.diffuse_sample_num, 0)
		az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi
		self.diffuse_direction_samples = torch.from_numpy(np.stack([az, el], -1).astype(np.float32)).cuda()
		self.diffuse_occ = torch.ones((0, self.diffuse_sample_num), device="cuda")

	def freeze_all_but_albedo(self) -> None:
		for p in [self._xyz, self._features_dc, self._features_rest, self._opacity, self._scaling, self._rotation, self._metallic_init, self._roughness_init]:
			p.requires_grad_(False)
		self._albedo_init.requires_grad_(True)

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
		# (P,1,3)+(P,S-1,3)->(P,S,3)
		return torch.cat([self._features_dc, self._features_rest], dim=1)

	@property
	def get_albedo_init(self) -> torch.Tensor:
		return self._albedo_init.clamp(0.0, 1.0)

	@property
	def get_metallic_init(self) -> torch.Tensor:
		return self._metallic_init.clamp(0.0, 1.0)

	@property
	def get_roughness_init(self) -> torch.Tensor:
		return self._roughness_init.clamp(0.0, 1.0)

	@property
	def get_rotation_matrix(self) -> torch.Tensor:
		# 四元数 -> 旋转矩阵
		r = self.get_rotation
		norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
		q = r / norm[:, None]
		R = torch.zeros((q.size(0), 3, 3), device=r.device)
		rw, rx, ry, rz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
		R[:, 0, 0] = 1 - 2 * (ry * ry + rz * rz)
		R[:, 0, 1] = 2 * (rx * ry - rw * rz)
		R[:, 0, 2] = 2 * (rx * rz + rw * ry)
		R[:, 1, 0] = 2 * (rx * ry + rw * rz)
		R[:, 1, 1] = 1 - 2 * (rx * rx + rz * rz)
		R[:, 1, 2] = 2 * (ry * rz - rw * rx)
		R[:, 2, 0] = 2 * (rx * rz - rw * ry)
		R[:, 2, 1] = 2 * (ry * rz + rw * rx)
		R[:, 2, 2] = 1 - 2 * (rx * rx + ry * ry)
		return R

	@property
	def get_eigenvector(self) -> torch.Tensor:
		# 以最小缩放方向近似法线方向
		scales = self.get_scaling
		idx = torch.min(scales, dim=1)[1]
		normals = self.get_rotation_matrix[torch.arange(scales.shape[0]), :, idx]
		return torch.nn.functional.normalize(normals, dim=1)

	def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
		L = build_scaling_rotation(scaling_modifier * self.get_scaling, self.get_rotation)
		return strip_symmetric(L @ L.transpose(1, 2))

	def sample_sphere(self, num_samples: int, begin_elevation: int = 0):
		ratio = (begin_elevation + 90) / 180
		num_points = int(num_samples // (1 - ratio))
		phi = (np.sqrt(5) - 1.0) / 2.0
		azimuths, elevations = [], []
		for n in range(num_points - num_samples, num_points):
			z = 2.0 * n / num_points - 1.0
			azimuths.append(2 * np.pi * n * phi % (2 * np.pi))
			elevations.append(np.arcsin(z))
		return np.array(azimuths), np.array(elevations)

	def compute_color(self, camera_center: torch.Tensor, iteration: int = 60001, is_train: bool = True,
					 first_stage_step: int = 5000, second_stage_step: int = 30000,
					 remove_noise: bool = False, hdr_rotation: bool = False, exposure: float = 0.0, force_color_sh_only: bool = False):
		means3D = self.get_xyz
		albedo = self.get_albedo_init
		shading_normal = self.get_eigenvector
		metallic = self.get_metallic_init
		roughness = self.get_roughness_init
		envlight = self.envlight

		v = torch.nn.functional.normalize(camera_center - means3D, dim=1)
		n_dot_v = dot(v, shading_normal)
		reflective = n_dot_v * shading_normal * 2 - v

		occ = None
		indirect_light = None
		if (not force_color_sh_only) and (iteration > second_stage_step):
			shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
			from .sh_utils import eval_sh  # local import
			sh2rgb = eval_sh(self.active_sh_degree, shs_view, reflective)
			indirect_light = torch.clamp_min(sh2rgb + 0.5, 0.0)

		if hdr_rotation:
			reflective_x = reflective[:, 0:1]
			reflective_y = reflective[:, 1:2]
			reflective_z = reflective[:, 2:3]
			reflective = torch.cat([-reflective_y, reflective_z, -reflective_x], dim=-1)

		diffuse_albedo = (1 - metallic) * albedo
		fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)

		if dr is not None:
			fg = dr.texture(
				self.FG_LUT,
				fg_uv.reshape(1, 1, -1, 2).contiguous(),
				filter_mode="linear",
				boundary_mode="clamp",
			).reshape(albedo.shape[0], 2)
		else:
			# grid_sample fallback
			lut = self.FG_LUT.permute(0, 3, 1, 2).contiguous()
			u, v = fg_uv[:, 0:1], fg_uv[:, 1:2]
			grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).unsqueeze(0)
			import torch.nn.functional as F
			out = F.grid_sample(lut, grid, mode='bilinear', padding_mode='border', align_corners=True)
			fg = out[0, :, :, 0].transpose(0, 1)

		F0 = (1 - metallic) * 0.04 + metallic * albedo
		specular_albedo = F0 * fg[:, 0:1] + fg[:, 1:2]

		if is_train:
			envlight.build_base()
		envlight.build_mips()

		if iteration > second_stage_step:
			diffuse_dirs = sample_diffuse_directions(shading_normal, self.diffuse_direction_samples, is_train=False)
			diffuse_dirs = diffuse_dirs.reshape(-1, 3)
			if hdr_rotation:
				dx, dy, dz = diffuse_dirs[:, 0:1], diffuse_dirs[:, 1:2], diffuse_dirs[:, 2:3]
				diffuse_dirs = torch.cat([-dy, dz, -dx], dim=-1)
			diffuse_light = envlight(diffuse_dirs)
			diffuse_light = diffuse_light.reshape(albedo.shape[0], self.diffuse_sample_num, 3).mean(dim=1)
		else:
			if hdr_rotation:
				nx, ny, nz = shading_normal[:, 0:1], shading_normal[:, 1:2], shading_normal[:, 2:3]
				shading_normal = torch.cat([-ny, nz, -nx], dim=-1)
			diffuse_light = envlight(shading_normal)
		
		specular_light = envlight(reflective, roughness)
		
		diffuse_color = diffuse_albedo * diffuse_light
		if indirect_light is not None:
			specular_color = (indirect_light + specular_light) * specular_albedo
		else:
			specular_color = specular_light * specular_albedo

		color = linear_to_srgb((diffuse_color + specular_color) * 2 ** exposure).clamp(0.0, 1.0)
		mask = (n_dot_v > 0).float()
		color = color * mask
		
		result = {"color": color}

		if not is_train:
			result.update({
				"albedo": linear_to_srgb(albedo).clamp(0.0, 1.0),
				"diffuse_albedo": linear_to_srgb(diffuse_albedo).clamp(0.0, 1.0),
				"diffuse_light": linear_to_srgb(diffuse_light).clamp(0.0, 1.0),
				"diffuse_color": linear_to_srgb(diffuse_color*2**exposure).clamp(0.0, 1.0),
				"specular_albedo": linear_to_srgb(specular_albedo).clamp(0.0, 1.0),
				"specular_light": linear_to_srgb(specular_light).clamp(0.0, 1.0),
				"specular_color": linear_to_srgb(specular_color*2**exposure).clamp(0.0, 1.0),
				"specular_indirect_light": None,
				"specular_direct_light": None,
				"specular_indirect_color": None,
				"specular_direct_color": None,
				"occ": None
			})
		return result

	def load_ply(self, path: str) -> None:
		from plyfile import PlyData
		plydata = PlyData.read(path)

		xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])), axis=1)
		opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
		albedo_init = np.stack((np.asarray(plydata.elements[0]["albedo_r"]), np.asarray(plydata.elements[0]["albedo_g"]), np.asarray(plydata.elements[0]["albedo_b"])), axis=1)
		metallic_init = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
		roughness_init = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

		features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
		features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"]) if "f_dc_0" in plydata.elements[0].data.dtype.names else 0.0
		features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"]) if "f_dc_1" in plydata.elements[0].data.dtype.names else 0.0
		features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"]) if "f_dc_2" in plydata.elements[0].data.dtype.names else 0.0

		extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
		extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
		features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
		for idx, attr_name in enumerate(extra_f_names):
			features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
		features_extra = features_extra.reshape((features_extra.shape[0], 3, -1)) if features_extra.shape[1] > 0 else np.zeros((xyz.shape[0], 3, 0), dtype=np.float32)

		scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
		scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
		for idx, attr_name in enumerate(scale_names):
			scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

		rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")], key=lambda x: int(x.split('_')[-1]))
		rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
		for idx, attr_name in enumerate(rot_names):
			rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

		coeff_per_channel = features_extra.shape[2] + 1
		self.max_sh_degree = int(np.sqrt(max(coeff_per_channel, 1)) - 1)
		self.active_sh_degree = self.max_sh_degree

		self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda"), requires_grad=False)
		self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous(), requires_grad=False)
		self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous(), requires_grad=False)
		self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda"), requires_grad=False)
		self._scaling = nn.Parameter(torch.log(torch.tensor(scales, dtype=torch.float32, device="cuda")), requires_grad=False)
		self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda"), requires_grad=False)
		self._albedo_init = nn.Parameter(torch.tensor(albedo_init, dtype=torch.float32, device="cuda"), requires_grad=True)
		self._metallic_init = nn.Parameter(torch.tensor(metallic_init, dtype=torch.float32, device="cuda"), requires_grad=False)
		self._roughness_init = nn.Parameter(torch.tensor(roughness_init, dtype=torch.float32, device="cuda"), requires_grad=False)
		self.diffuse_occ = torch.ones((self._xyz.shape[0], self.diffuse_sample_num), device="cuda")

	def save_ply(self, path: str) -> None:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		from plyfile import PlyElement, PlyData
		xyz = self._xyz.detach().cpu().numpy()
		f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
		f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
		opacities = self._opacity.detach().cpu().numpy()
		scale = self._scaling.detach().cpu().numpy()
		rotation = self._rotation.detach().cpu().numpy()
		albedo_init = self._albedo_init.detach().cpu().numpy()
		metallic_init = self._metallic_init.detach().cpu().numpy()
		roughness_init = self._roughness_init.detach().cpu().numpy()
		normals = self.get_eigenvector.detach().cpu().numpy()

		attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
		for i in range(f_dc.shape[1]):
			attrs.append(f'f_dc_{i}')
		for i in range(f_rest.shape[1]):
			attrs.append(f'f_rest_{i}')
		attrs.append('opacity')
		for i in range(scale.shape[1]):
			attrs.append(f'scale_{i}')
		for i in range(rotation.shape[1]):
			attrs.append(f'rot_{i}')
		attrs += ['albedo_r', 'albedo_g', 'albedo_b', 'metallic', 'roughness']

		dtype_full = [(a, 'f4') for a in attrs]
		elements = np.empty(xyz.shape[0], dtype=dtype_full)
		attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, albedo_init, metallic_init, roughness_init), axis=1)
		elements[:] = list(map(tuple, attributes))
		el = PlyElement.describe(elements, 'vertex')
		PlyData([el]).write(path)
