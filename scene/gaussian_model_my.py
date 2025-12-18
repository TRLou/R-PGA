#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
高斯模型定义

该文件实现用于3D高斯溅射渲染的核心数据结构和训练相关操作，包括：
- 高斯参数（位置、缩放、旋转、不透明度、SH特征、材质参数等）的存取与激活函数
- 学习率调度与优化器参数分组
- 点的密度化、克隆、分裂与剪枝逻辑
- 材质与光照相关的颜色计算（漫反射/镜面、直接/间接）
- 体素网格构建与遮挡估计
- PLY存储与加载
"""

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, eval_sh
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.ir_utils import linear_to_srgb, sample_diffuse_directions, dot
import nvdiffrast.torch as dr
import envlight
from einops import rearrange
# from envlight import EnvmapMaterial

class GaussianModelMy:

    def setup_functions(self):
        """
        配置各参数的激活与反激活函数，以及协方差构造函数。
        - 缩放: exp / log
        - 旋转: 归一化四元数
        - 不透明度: sigmoid / inverse_sigmoid
        - 协方差: 基于缩放与旋转的对称正定矩阵
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.metallic_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid


    def __init__(self, sh_degree : int, environment_texture=None, environment_scale=1.0):
        """
        初始化高斯模型。
        参数:
        - sh_degree: 最大SH阶数
        - environment_texture: 环境光贴图路径或数据
        - environment_scale: 环境光强度缩放
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # Color SH for background rendering (separate from indirect-light SH)
        self._color_features_dc = torch.empty(0)
        self._color_features_rest = torch.empty(0)

        # Foreground scoring and mask (updated in intervals)
        self.fg_score = torch.empty(0)
        self.fg_mask = torch.empty(0, dtype=torch.bool)

        self._albedo_init = torch.empty(0)
        self._metallic_init = torch.empty(0)
        self._roughness_init = torch.empty(0)
        self.diffuse_occ = torch.empty(0)
        self.FG_LUT = torch.from_numpy(
            np.fromfile("load/lights/bsdf_256_256.bin", dtype=np.float32).reshape(
                1, 256, 256, 2
            )).cuda()
        self.grid = torch.empty(0)
        self.min_pts = torch.empty(0)
        self.max_pts = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # 采样设置与可训练环境光
        self.diffuse_sample_num = 128
        self.specular_sample_num = 24
        self.envlight = envlight.EnvLight(environment_texture, scale=environment_scale, min_res=16, max_res=512, min_roughness=0.08, max_roughness=0.5, trainable=True).cuda()

        # Foreground usage flag toggled by training loop after first interval
        self.use_fg_mask = False

        # 预采样漫反射方向（用于积分近似），映射到[0,1]
        az, el = self.sample_sphere(self.diffuse_sample_num, 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi # scale to [0,1]
        self.diffuse_direction_samples = np.stack([az, el], -1)
        self.diffuse_direction_samples = torch.from_numpy(self.diffuse_direction_samples.astype(np.float32)).cuda() # [dn0,2]

        self.visibility_score = torch.empty(0)
        self.aniso_score = torch.empty(0)
        self.score_counter = 0

        # Voxel grid and occlusion parameters (configurable)
        self.grid_resolution = 256  # e.g., 128 or 256
        self.grid_covariance_scale = 1.0  # ellipsoid dilation factor in grid
        # Near-range occlusion sampling along reflection ray
        self.occ_num_samples = 64
        self.occ_near_start = 0.05  # relative start along max distance
        self.occ_near_end = 0.10    # relative end along max distance
        self.occ_hit_threshold = 10  # hits over samples to consider occluded

        # Debug counters for densification sources
        self.debug_split_added = 0
        self.debug_clone_added = 0
        self.debug_extra_split_added = 0
        self.debug_pruned = 0
        self.debug_extra_pruned = 0

        # Per-point auxiliary signal: albedo gradient proxy (to be updated externally)
        self.albedo_gradient = torch.empty(0)

        # Hybrid-driven thresholds (ratios) - DEPRECATED
        # Percentile-based selection for extra densify (top/bottom fractions)
        self.vs_top_percent = 0.10
        self.ag_top_percent = 0.10
        self.vs_low_percent = 0.10

    def capture(self):
        """
        捕获可序列化的模型状态，用于保存检查点。
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._albedo_init,
            self._metallic_init,
            self._roughness_init,
            self.diffuse_occ,
            self.grid,
            self.max_pts,
            self.min_pts,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.envlight.state_dict(),
            self._color_features_dc,
            self._color_features_rest,
            self.fg_score,
            self.fg_mask
        )
    
    def restore(self, model_args, training_args):
        """
        从保存的状态恢复模型与优化器，并重新按训练参数进行设置。
        """
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._albedo_init,
        self._metallic,
        self._roughness,
        self.diffuse_occ,
        self.grid,
        self.max_pts,
        self.min_pts,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        envlight_state,
        self._color_features_dc,
        self._color_features_rest,
        self.fg_score,
        self.fg_mask) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        # restore envlight
        try:
            self.envlight.load_state_dict(envlight_state)
        except Exception:
            pass

    @property
    def get_albedo_init(self):
        """反照率初始化参数，限制到[0,1]并做轻微偏移以稳定训练"""
        return (self._albedo_init + 0.5).clamp(0.0,1.0)
    
    @property
    def get_metallic_init(self):
        """金属度初始化，限制到[0,1]"""
        return self._metallic_init.clamp(0.0, 1.0)

    @property
    def get_roughness_init(self):
        """粗糙度初始化，限制到[0,1]"""
        return self._roughness_init.clamp(0.0, 1.0)

    @property
    def get_rotation_matrix(self):
        """四元数转换到旋转矩阵"""
        return build_rotation(self._rotation)
    
    @property
    def get_FG_LUT(self):
        """Fresnel与几何项查找表(FG LUT)"""
        return self.FG_LUT
    
    @property
    def get_envlight(self):
        """可训练环境光对象"""
        return self.envlight

    @property
    def get_eigenvector(self):
        """
        以最小缩放方向近似法线方向。
        从旋转矩阵中取与最小尺度对应的列向量作为法线并归一化。
        """
        scales = self.get_scaling
        N = scales.shape[0]
        idx = torch.min(scales, dim=1)[1]
        normals = self.get_rotation_matrix[np.arange(N), :, idx]
        normals = torch.nn.functional.normalize(normals, dim=1)
        return normals
    
    def sample_sphere(self, num_samples,begin_elevation = 0):
        """ 
        在球面上均匀采样方位角与仰角。
        参考: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        返回两个数组 (azimuths, elevations)
        """
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(2 * np.pi * n * phi % (2 * np.pi))
            elevations.append(np.arcsin(z))
        return np.array(azimuths), np.array(elevations)
    

    def get_sphere_intersection(self, pts, dirs, R=1):
        """
        计算从点 pts 沿方向 dirs 与球面半径R的交点距离（正向解）。
        用于在球面范围内进行体素采样累积。
        """
        dtx = torch.sum(pts*dirs,dim=-1,keepdim=True) # rn,1
        xtx = torch.sum(pts**2,dim=-1,keepdim=True) # rn,1
        dist = dtx ** 2 - xtx + R**2
        assert torch.sum(dist<0)==0
        dist = -dtx + torch.sqrt(dist+1e-6) # rn,1
        return dist

    def query_pts_value(self, points, grid_num=128):
        """
        查询体素网格中若干三维点的占据值（0/1），用于遮挡估计。
        """
        if self.grid.numel() == 0:
            return torch.zeros(points.shape[:-1] + (1,), dtype=torch.uint8, device=points.device)
        res = int(self.grid.shape[0])
        normalized_pc = (points - self.min_pts) / (self.max_pts - self.min_pts)
        indices = (normalized_pc * (res - 1)).long().clamp(0, res - 1)
        result = self.grid[indices[:, :, 0], indices[:, : , 1], indices[:, : , 2]]
        return result

    def compute_occlusion(self, n_dot_v, reflective, index=None):
        """
        通过体素网格沿反射方向采样，估计镜面遮挡 occ。
        - 支持两种输入形状：
          n_dot_v: [P,1] 且 reflective: [P,3]; 或 n_dot_v: [P,sn,1] 且 reflective: [P,sn,3]
        - 若提供 index（Bool/Long），将同时对子集进行裁剪，以保证掩码与被索引张量形状一致。
        返回：
          当输入为 [P,1] 时返回 [P,1]；当输入为 [P,sn,1] 时返回 [P,sn,1]
        """
        with torch.no_grad():
            # 解析索引并对子集进行裁剪，确保 xyz / n_dot_v / reflective 对齐
            if index is not None:
                if isinstance(index, torch.Tensor) and index.dtype == torch.bool:
                    idx_mask = index
                else:
                    idx_mask = index
                xyz = self.get_xyz[idx_mask, :]
                n_local = n_dot_v[idx_mask]
                r_local = reflective[idx_mask]
            else:
                xyz = self.get_xyz
                n_local = n_dot_v
                r_local = reflective

            # 若禁用遮挡采样，直接返回与子集形状一致的零张量
            if self.occ_num_samples <= 0:
                return torch.zeros_like(n_local)

            device = xyz.device
            P = int(xyz.shape[0])
            if P == 0:
                # 空子集
                if n_local.dim() == 3:
                    return torch.zeros((0, n_local.shape[1], 1), device=device)
                return torch.zeros((0, 1), device=device)

            distances = torch.norm(xyz, dim=1)
            max_radiu = torch.max(distances) + 0.25
            sam_pts_num = int(self.occ_num_samples)
            near_start = float(self.occ_near_start)
            near_end = float(self.occ_near_end)
            near_end = max(near_end, near_start + 1e-3)
            z_base = torch.linspace(0, 1, sam_pts_num, device=device)
            hit_th = float(self.occ_hit_threshold)

            # 分支1：多方向 [P, sn, 1]
            if n_local.dim() == 3:
                Psub, sn, _ = n_local.shape
                valid_2d = (n_local.squeeze(-1) > 0.3)  # [Psub, sn]
                occ_weight = torch.zeros((Psub, sn), device=device)
                if valid_2d.any():
                    flat_valid = valid_2d.reshape(-1)  # [Psub*sn]
                    # 展开 xyz 与方向到逐元素配对后再用扁平掩码筛选
                    xyz_all = xyz.unsqueeze(1).expand(-1, sn, 3).reshape(-1, 3)
                    dirs_all = r_local.reshape(-1, 3)
                    pts_filter = xyz_all[flat_valid]
                    dirs = dirs_all[flat_valid]
                    if pts_filter.numel() > 0:
                        max_dist = self.get_sphere_intersection(pts_filter, dirs, max_radiu)  # [Nv,1]
                        z_vals = (near_start + (near_end - near_start) * z_base).unsqueeze(0) * max_dist  # [Nv,sam]
                        pts_sample = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + pts_filter.unsqueeze(-2)  # [Nv,sam,3]
                        pts_weight = self.query_pts_value(pts_sample)  # [Nv,sam,1]
                        pts_hits = torch.sum(pts_weight, dim=-2).squeeze(-1).float()  # [Nv]
                        occ_weight.view(-1)[flat_valid] = pts_hits
                occ = occ_weight.unsqueeze(-1)
                return (occ > hit_th).float()

            # 分支2：单方向 [P,1]
            valid_1d = (n_local.squeeze(-1) > 0.3)  # [P]
            occ_weight = torch.zeros((P,), device=device)
            if valid_1d.any():
                pts_filter = xyz[valid_1d]
                dirs = r_local[valid_1d]
                if pts_filter.numel() > 0:
                    max_dist = self.get_sphere_intersection(pts_filter, dirs, max_radiu)  # [Nv,1]
                    z_vals = (near_start + (near_end - near_start) * z_base).unsqueeze(0) * max_dist  # [Nv,sam]
                    pts_sample = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + pts_filter.unsqueeze(-2)  # [Nv,sam,3]
                    pts_weight = self.query_pts_value(pts_sample)  # [Nv,sam,1]
                    pts_hits = torch.sum(pts_weight, dim=-1).float().squeeze(-1)  # [Nv]
                    occ_weight[valid_1d] = pts_hits
            occ = occ_weight.unsqueeze(-1)
            return (occ > hit_th).float()

    def get_grid(self, grid_num=128):
        """
        基于高斯的世界坐标、缩放与旋转，构建近似占据的体素网格。
        用于后续的遮挡估计。
        """
        with torch.no_grad():
            if grid_num is None:
                grid_num = int(self.grid_resolution)
            voxel_grid = torch.zeros((grid_num, grid_num, grid_num), dtype=torch.uint8, device="cuda")
            xyz = self.get_xyz
            min_vals, _ = torch.min(xyz, dim=0)
            min_vals = min_vals - 0.1
            max_vals, _ = torch.max(xyz, dim=0)
            max_vals = max_vals + 0.1
            xyz = (xyz - min_vals) / (max_vals - min_vals)
            scales = self.get_scaling
            rotations = self.get_rotation
            L = build_scaling_rotation(scales, rotations)
            covariance = L @ L.transpose(1, 2) * float(self.grid_covariance_scale)
            x111 = ((xyz + covariance[..., 0] + covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x110 = ((xyz + covariance[..., 0] + covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            x101 = ((xyz + covariance[..., 0] - covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x100 = ((xyz + covariance[..., 0] - covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            x011 = ((xyz - covariance[..., 0] + covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x010 = ((xyz - covariance[..., 0] + covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            x001 = ((xyz - covariance[..., 0] - covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x000 = ((xyz - covariance[..., 0] - covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            xyz_idx = (xyz * (grid_num - 1)).long()
            indices = torch.cat([x111, x110, x101, x100, x011, x010, x001, x000, xyz_idx], 0).clamp(0, grid_num - 1)
            voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            self.grid = voxel_grid
            self.min_pts = min_vals
            self.max_pts = max_vals

    def get_grid_fg(self, grid_num=128, mask=None):
        """
        HYBRID_FG_ONLY(v2): 仅使用前景椭球构建体素占据网格。

        参数:
        - grid_num: 体素分辨率
        - mask: 可选 Bool/Long 索引；若不提供则尝试使用 `self.fg_mask`，不可用时回退到 `get_grid()`。

        回退策略:
        - 坐标系(min/max)始终以全量点计算，保证与 `get_grid` 一致，便于回退。
        """
        with torch.no_grad():
            if grid_num is None:
                grid_num = int(self.grid_resolution)
            voxel_grid = torch.zeros((grid_num, grid_num, grid_num), dtype=torch.uint8, device="cuda")

            xyz_all = self.get_xyz
            min_vals, _ = torch.min(xyz_all, dim=0)
            min_vals = min_vals - 0.1
            max_vals, _ = torch.max(xyz_all, dim=0)
            max_vals = max_vals + 0.1

            # 选择掩码
            sel_mask = None
            if mask is not None:
                sel_mask = mask
            elif getattr(self, 'use_fg_mask', False) and isinstance(getattr(self, 'fg_mask', None), torch.Tensor) and self.fg_mask.numel() == xyz_all.shape[0]:
                sel_mask = self.fg_mask

            if sel_mask is None:
                # 回退全量网格
                self.get_grid(grid_num)
                return

            # 过滤后的子集
            if isinstance(sel_mask, torch.Tensor) and sel_mask.dtype == torch.bool:
                xyz = xyz_all[sel_mask]
                scales = self.get_scaling[sel_mask]
                rotations = self.get_rotation[sel_mask]
            else:
                xyz = xyz_all[sel_mask]
                scales = self.get_scaling[sel_mask]
                rotations = self.get_rotation[sel_mask]

            if xyz.shape[0] > 0:
                xyz_norm = (xyz - min_vals) / (max_vals - min_vals)
                L = build_scaling_rotation(scales, rotations)
                covariance = L @ L.transpose(1, 2) * float(self.grid_covariance_scale)
                x111 = ((xyz_norm + covariance[..., 0] + covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
                x110 = ((xyz_norm + covariance[..., 0] + covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
                x101 = ((xyz_norm + covariance[..., 0] - covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
                x100 = ((xyz_norm + covariance[..., 0] - covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
                x011 = ((xyz_norm - covariance[..., 0] + covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
                x010 = ((xyz_norm - covariance[..., 0] + covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
                x001 = ((xyz_norm - covariance[..., 0] - covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
                x000 = ((xyz_norm - covariance[..., 0] - covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
                centers = (xyz_norm * (grid_num - 1)).long()
                indices = torch.cat([x111, x110, x101, x100, x011, x010, x001, x000, centers], 0).clamp(0, grid_num - 1)
                voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

            self.grid = voxel_grid
            self.min_pts = min_vals
            self.max_pts = max_vals

    def get_diffuse_occ(self):
        """
        预计算漫反射方向上的遮挡图 self.diffuse_occ，用于在第二阶段后修正漫反射光照。
        """
        with torch.no_grad():
            # 若启用混合渲染且前景掩码可用，则仅以前景椭球构网格，否则使用全量
            if getattr(self, 'use_fg_mask', False) and isinstance(self.fg_mask, torch.Tensor) and self.fg_mask.numel() == self.get_xyz.shape[0]:
                self.get_grid_fg()
            else:
                self.get_grid()
            normal = self.get_eigenvector
            self.diffuse_occ = torch.zeros((self.get_xyz.shape[0], self.diffuse_sample_num), device=normal.device)
            diffuse_directions = sample_diffuse_directions(normal, self.diffuse_direction_samples, is_train=False)
            xyz = self.get_xyz
            distances = torch.norm(xyz, dim=1)
            max_radiu = torch.max(distances) + 0.25
            sam_pts_num = 128
            for i in range(self.diffuse_sample_num):
                dirs = diffuse_directions[:,i,:]
                max_dist = self.get_sphere_intersection(xyz, dirs, max_radiu) # pn,1
                z_vals = torch.linspace(0, 1, sam_pts_num)[16:].cuda()
                z_vals = max_dist * z_vals.unsqueeze(0)
                pts_sample = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + xyz.unsqueeze(-2)
                pts_weight = self.query_pts_value(pts_sample)
                pts_weight = torch.sum(pts_weight, dim=-1)
                self.diffuse_occ[:,i] = (pts_weight > 1).float()

    def compute_color(self, camera_center, iteration=None, is_train=None, first_stage_step=5000, second_stage_step=30000, remove_noise=False, hdr_rotation=False, exposure=0.0, force_color_sh_only=False):
        """
        计算颜色与相关分解（漫反射/镜面、直接/间接），并返回渲染所需的各通道。
        参数:
        - camera_center: 相机中心（用于计算视向量）
        - iteration/is_train/first_stage_step/second_stage_step: 控制训练阶段逻辑
        - remove_noise: 可选的点位噪声抑制
        - hdr_rotation: 对法线/反射/漫反射方向执行固定旋转（数据增强或特定坐标系需要）
        - exposure: 曝光补偿，对最终颜色进行线性调整
        返回:
        - dict，包含 color / albedo / diffuse_albedo / diffuse_light / diffuse_color / specular_* / occ 等
        """
        means3D = self.get_xyz
        if remove_noise:
            v = camera_center - means3D
            dis = torch.sum(v * v, dim=-1)
            xyz_mask = torch.where(dis > 1.25, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
            self._xyz.data = self._xyz.data * xyz_mask.unsqueeze(-1)
            means3D = self.get_xyz

        albedo = self.get_albedo_init
        shading_normal = self.get_eigenvector
        metallic = self.get_metallic_init
        roughness = self.get_roughness_init
        envlight = self.get_envlight

        v = camera_center - means3D        # 入射向量
        v = torch.nn.functional.normalize(v, dim=1)
        n_dot_v = dot(v, shading_normal)   # cos_theta
        reflective = n_dot_v * shading_normal * 2 - v   # 计算反射向量

        occ = None
        if iteration > second_stage_step:
            shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, reflective)   # 给定sh，反射向量，计算出间接光的sh（rgb）
            indirect_light = torch.clamp_min(sh2rgb + 0.5, 0.0)  # [-0.5, 0.5] 投影回 [0, 1]
            # Compute occlusion only for foreground when fg_mask is used
            if getattr(self, 'use_fg_mask', False) and self.fg_mask.numel() == means3D.shape[0]:
                with torch.no_grad():
                    # Only compute occ for foreground points, others zero
                    occ_fg = self.compute_occlusion(n_dot_v, reflective, index=self.fg_mask)
                    if occ_fg is not None and occ_fg.numel() > 0:
                        if occ_fg.dim() == 3:
                            occ = torch.zeros((means3D.shape[0], occ_fg.shape[1], 1), device=means3D.device)
                            occ[self.fg_mask] = occ_fg
                        else:
                            occ = torch.zeros((means3D.shape[0], 1), device=means3D.device)
                            occ[self.fg_mask] = occ_fg
                    else:
                        occ = torch.zeros((means3D.shape[0], 1), device=means3D.device)
            else:
                occ = self.compute_occlusion(n_dot_v, reflective)

        if hdr_rotation:
            reflective_x = reflective[:,0].unsqueeze(-1)
            reflective_y = reflective[:,1].unsqueeze(-1)
            reflective_z = reflective[:,2].unsqueeze(-1)

            reflective = torch.cat([-reflective_y, reflective_z, -reflective_x], dim=-1) #rot_y(90)@rot_x(-90)

        prefix_shape = albedo.shape[:-1]
        diffuse_albedo = (1 - metallic) * albedo
        fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
        
        # 这里fg1 和 fg2 经degub测试为一致结果
        # fg_bp = dr.texture(
        #     self.get_FG_LUT,
        #     fg_uv.reshape(1, -1, 1, 2).contiguous(),
        #     filter_mode="linear",
        #     boundary_mode="clamp",
        # ).reshape(*prefix_shape, 2)
        
        fg = dr.texture(
            self.get_FG_LUT,
            fg_uv.reshape(1, 1, -1, 2).contiguous(),
            filter_mode="linear",
            boundary_mode="clamp",
        ).reshape(*prefix_shape, 2)
        
        
        F0 = (1 - metallic) * 0.04 + metallic * albedo
        specular_albedo = F0 * fg[:, 0:1] + fg[:, 1:2]
        if is_train:
            envlight.build_base()
        envlight.build_mips()
        if iteration > second_stage_step:
            diffuse_directions = sample_diffuse_directions(shading_normal, self.diffuse_direction_samples, is_train=False)  # [pn,sn0,3]
            diffuse_directions = rearrange(diffuse_directions, "B N C -> (B N) C")
            if hdr_rotation:
                diffuse_directions_x = diffuse_directions[:,0].unsqueeze(-1)
                diffuse_directions_y = diffuse_directions[:,1].unsqueeze(-1)
                diffuse_directions_z = diffuse_directions[:,2].unsqueeze(-1)
                diffuse_directions = torch.cat([-diffuse_directions_y, diffuse_directions_z, -diffuse_directions_x], dim=-1) #rot_y(90)@rot_x(-90)
            diffuse_light = envlight(diffuse_directions)
            diffuse_occ = rearrange(self.diffuse_occ, "B N -> (B N)").unsqueeze(1)
            diffuse_light = (1-diffuse_occ) * diffuse_light
            diffuse_light = rearrange(diffuse_light, "(B N) C -> B N C", N=self.diffuse_sample_num)
            diffuse_light = torch.mean(diffuse_light, dim=1)
        else:
            if hdr_rotation:
                normal_x = shading_normal[:,0].unsqueeze(-1)
                normal_y = shading_normal[:,1].unsqueeze(-1)
                normal_z = shading_normal[:,2].unsqueeze(-1)
                shading_normal = torch.cat([-normal_y, normal_z, -normal_x], dim=-1)
            diffuse_light = envlight(shading_normal)
        diffuse_color = diffuse_albedo * diffuse_light
        specular_light = envlight(reflective, roughness)

        # In no-occlusion mode, the logic is simply additive.
        if iteration > second_stage_step and self.occ_num_samples <= 0:
            specular_direct_light = specular_light
            specular_indirect_light = indirect_light
            
            specular_indirect_color = specular_indirect_light * specular_albedo
            specular_direct_color = specular_direct_light * specular_albedo
            
            specular_light = specular_direct_light + specular_indirect_light # Total light for visualization
            specular_color_pbr = specular_direct_color + specular_indirect_color # Total color
            
            # Set occ to a dummy tensor to ensure indirect components are saved in the result dict
            occ = torch.zeros_like(n_dot_v)
        elif occ is not None:
            specular_indirect_light = occ * indirect_light
            specular_direct_light = (1-occ) * specular_light
            specular_indirect_color = specular_indirect_light * specular_albedo
            specular_direct_color = specular_direct_light * specular_albedo
            specular_light = specular_indirect_light + specular_direct_light
            specular_color_pbr = specular_indirect_color + specular_direct_color
        else:
            specular_color_pbr = specular_light * specular_albedo

        # Background color via color SH (compute if available)
        color_sh_rgb = None
        if self._color_features_dc is not None and self._color_features_dc.numel() > 0:
            color_sh = torch.cat((self._color_features_dc, self._color_features_rest), dim=1)  # (P, S, 3)
            color_sh = color_sh.transpose(1, 2).contiguous().view(-1, 3, (self.max_sh_degree+1)**2)
            color_sh_rgb = torch.clamp(eval_sh(self.active_sh_degree, color_sh, shading_normal), 0.0, 1.0)

        # --- 开始：临时可视化修改 ---
        pbr_part_vis = None
        sh_part_vis = None
        # --- 结束：临时可视化修改 ---

        # Compose final color based on mode and foreground mask
        if force_color_sh_only and (color_sh_rgb is not None):
            specular_color = torch.zeros_like(diffuse_color)
            color = linear_to_srgb((color_sh_rgb)*2**exposure).clamp(0.0, 1.0)
        else:
            if getattr(self, 'use_fg_mask', False) and self.fg_mask.numel() == means3D.shape[0] and color_sh_rgb is not None:
                # PBR for FG
                pbr_color = diffuse_color + specular_color_pbr
                # SH color for BG
                composed_linear = torch.zeros_like(pbr_color)
                composed_linear[self.fg_mask] = pbr_color[self.fg_mask]
                composed_linear[~self.fg_mask] = color_sh_rgb[~self.fg_mask]
                color = linear_to_srgb(composed_linear*2**exposure).clamp(0.0, 1.0)
                specular_color = specular_color_pbr
                
                # --- 开始：临时可视化修改 ---
                # 创建可视化张量
                pbr_part_vis = torch.zeros_like(pbr_color)
                pbr_part_vis[self.fg_mask] = pbr_color[self.fg_mask]
                sh_part_vis = torch.zeros_like(pbr_color)
                sh_part_vis[~self.fg_mask] = color_sh_rgb[~self.fg_mask]
                # --- 结束：临时可视化修改 ---
            else:
                specular_color = specular_color_pbr
                color = linear_to_srgb((diffuse_color + specular_color)*2**exposure).clamp(0.0, 1.0)

        # 这里是法线校正，错误的法线方向通过mask2赋予随机颜色
        mask = torch.where(n_dot_v > 0, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
        mask2 = (torch.rand_like(mask) < 0.3).float()
        # 反转开始默认+5000
        if is_train and iteration > (first_stage_step+50000000):
                color = color * mask + torch.rand_like(color) * (1 - mask) * mask2
        else:
            color = color * mask

        result = {
            "color": color,
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
        }
        # --- 开始：临时可视化修改 ---
        if pbr_part_vis is not None:
            result["color_fg_pbr"] = linear_to_srgb(pbr_part_vis*2**exposure).clamp(0.0, 1.0)
        if sh_part_vis is not None:
            result["color_bg_sh"] = linear_to_srgb(sh_part_vis*2**exposure).clamp(0.0, 1.0)
        # --- 结束：临时可视化修改 ---
        if occ is not None:
            result.update({
            "specular_indirect_light": linear_to_srgb(specular_indirect_light).clamp(0.0, 1.0),
            "specular_direct_light": linear_to_srgb(specular_direct_light).clamp(0.0, 1.0),
            "specular_indirect_color": linear_to_srgb(specular_indirect_color).clamp(0.0, 1.0),
            "specular_direct_color": linear_to_srgb(specular_direct_color).clamp(0.0, 1.0),
            "occ": occ
            })
        return result

    @torch.no_grad()
    def count_specular_occluded_ellipsoids(self, camera_center):
        """
        统计当前视角下（基于反射方向的）被判定为遮挡的椭球数量。
        返回: (occ_count:int, total:int)
        """
        means3D = self.get_xyz
        if means3D.numel() == 0:
            return 0, 0
        shading_normal = self.get_eigenvector
        v = camera_center.to(means3D.device) - means3D
        v = torch.nn.functional.normalize(v, dim=1)
        n_dot_v = dot(v, shading_normal)
        reflective = n_dot_v * shading_normal * 2 - v
        occ = self.compute_occlusion(n_dot_v, reflective)
        if occ is None or occ.numel() == 0:
            return 0, int(means3D.shape[0])
        if occ.dim() == 3:
            # [P, sn, 1] -> per-ellipsoid any-occluded
            occ_mask = (occ.squeeze(-1) > 0).any(dim=1)
        else:
            occ_mask = (occ.squeeze(-1) > 0)
        return int(occ_mask.sum().item()), int(means3D.shape[0])

    @property
    def get_scaling(self):
        """对数尺度激活为正尺度"""
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        """四元数归一化"""
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        """返回点的世界坐标"""
        return self._xyz
    
    @property
    def get_features(self):
        """拼接SH直流与高频项，形状 (P, F, SH)"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        """不透明度经sigmoid激活"""
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        """基于缩放与旋转构造对称协方差矩阵（下三角乘自身转置）"""
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """逐步提升当前有效SH度数，直至上限"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        使用基础点云创建可训练的高斯集合：初始化位置、特征、缩放、旋转、不透明度与材质参数。
        并设置用于密度化与学习率缩放的相关状态。
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.zeros(fused_point_cloud.shape[0], 3).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # Initialize color SH (background color representation)
        color_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        color_features[:, :3, 0] = fused_color

        metallic = torch.rand(fused_point_cloud.shape[0], 1).float().cuda()
        roughness = torch.rand(fused_point_cloud.shape[0], 1).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._color_features_dc = nn.Parameter(color_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._color_features_rest = nn.Parameter(color_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.diffuse_occ = torch.ones((fused_point_cloud.shape[0], self.diffuse_sample_num), device="cuda")

        self._albedo_init = nn.Parameter(fused_color.requires_grad_(True))
        self._metallic_init = nn.Parameter(metallic.requires_grad_(True))
        self._roughness_init = nn.Parameter(roughness.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.visibility_score = torch.zeros((fused_point_cloud.shape[0]), device="cuda")
        self.aniso_score = torch.zeros_like(self.visibility_score)
        # Foreground scoring and mask
        self.fg_score = torch.zeros((fused_point_cloud.shape[0]), device="cuda")
        self.fg_mask = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.bool, device="cuda")

    def training_setup(self, training_args):
        """
        根据训练参数创建优化器与学习率调度器；将不同参数分配不同的lr与名称，便于后续调度。
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Albedo gradient accumulators
        self.albedo_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.albedo_grad_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._color_features_dc], 'lr': training_args.feature_lr, "name": "c_dc"},
            {'params': [self._color_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "c_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

            {'params': [self._albedo_init], 'lr': training_args.albedo_lr_init, "name": "albedo_init"},
            {'params': [self._metallic_init], 'lr': training_args.material_lr_init, "name": "metallic_init"},
            {'params': [self._roughness_init], 'lr': training_args.material_lr_init, "name": "roughness_init"},
            {'params': self.envlight.net.parameters(), 'lr': training_args.hdr_lr_init, "name": "hdr_net"},
            {'params': [self.envlight.init_base], 'lr': training_args.hdr_lr_init, "name": "hdr_init_base"},
            {'params': [self.envlight.base_train], 'lr': training_args.hdr_base_lr_init, "name": "hdr_base_train"},

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.hdr_scheduler_args = get_expon_lr_func(lr_init=training_args.hdr_lr_init,
                                                    lr_final=training_args.hdr_lr_final,
                                                    lr_delay_mult=training_args.hdr_lr_delay_mult,
                                                    max_steps=training_args.hdr_lr_max_steps)
        self.albedo_scheduler_args = get_expon_lr_func(lr_init=training_args.albedo_lr_init,
                                                    lr_final=training_args.albedo_lr_final,
                                                    lr_delay_mult=training_args.albedo_lr_delay_mult,
                                                    max_steps=training_args.albedo_lr_max_steps)
        self.material_scheduler_args = get_expon_lr_func(lr_init=training_args.material_lr_init,
                                                    lr_final=training_args.material_lr_final,
                                                    lr_delay_mult=training_args.material_lr_delay_mult,
                                                    max_steps=training_args.material_lr_max_steps)
        self.hdr_base_scheduler_args = get_expon_lr_func(lr_init=training_args.hdr_base_lr_init,
                                                    lr_final=training_args.hdr_base_lr_final,
                                                    lr_delay_mult=training_args.hdr_base_lr_delay_mult,
                                                    max_steps=training_args.hdr_base_lr_max_steps)
        self.densification_interval = getattr(training_args, 'densification_interval', 100)
        self.score_counter = 0


    def update_learning_rate(self, iteration):
        ''' 每步更新不同参数组的学习率 '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "c_dc":
                param_group['lr'] = self.hdr_scheduler_args(max(0, iteration-5000))
            if param_group["name"] == "c_rest":
                param_group['lr'] = self.hdr_scheduler_args(max(0, iteration-5000)) / 20.0
            if param_group["name"] == "hdr_net":
                lr = self.hdr_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "hdr_init_base":
                lr = self.hdr_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "hdr_base_train":
                lr = self.hdr_base_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "metallic_init":
                lr = self.material_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "roughness_init":
                lr = self.material_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "albedo_init":
                lr = self.albedo_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        """
        构建PLY导出所需的属性名称列表（xyz、法线、SH系数、不透明度、尺度、旋转、材质）。
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        # Color SH for background
        for i in range(self._color_features_dc.shape[1]*self._color_features_dc.shape[2]):
            l.append('c_dc_{}'.format(i))
        for i in range(self._color_features_rest.shape[1]*self._color_features_rest.shape[2]):
            l.append('c_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('albedo_r')
        l.append('albedo_g')
        l.append('albedo_b')
        l.append('metallic')
        l.append('roughness')
        return l

    def save_ply(self, path):
        """
        将当前高斯集合导出为PLY文件，包含几何、SH特征与材质。
        """
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        #normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        c_dc = self._color_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        c_rest = self._color_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        albedo_init = self._albedo_init.detach().cpu().numpy()
        metallic_init = self._metallic_init.detach().cpu().numpy()
        roughness_init = self._roughness_init.detach().cpu().numpy()
        normals = self.get_eigenvector.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, c_dc, c_rest, opacities, scale, rotation, albedo_init, metallic_init, roughness_init), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        将不透明度压低并替换到优化器中，以利于后续密度化与稳定性。
        """
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        从PLY文件加载高斯集合（几何、SH、材质等）并设置为可训练参数。
        """
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        albedo_init = np.stack((np.asarray(plydata.elements[0]["albedo_r"]),
                        np.asarray(plydata.elements[0]["albedo_g"]),
                        np.asarray(plydata.elements[0]["albedo_b"])),  axis=1)
        metallic_init = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
        roughness_init = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Load color SH if available; else initialize zeros
        c_dc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("c_dc_")]
        c_rest_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("c_rest_")]
        if len(c_dc_names) > 0 and len(c_rest_names) > 0:
            c_dc_names = sorted(c_dc_names, key = lambda x: int(x.split('_')[-1]))
            c_rest_names = sorted(c_rest_names, key = lambda x: int(x.split('_')[-1]))
            color_dc = np.zeros((xyz.shape[0], len(c_dc_names)))
            for idx, attr_name in enumerate(c_dc_names):
                color_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
            color_rest = np.zeros((xyz.shape[0], len(c_rest_names)))
            for idx, attr_name in enumerate(c_rest_names):
                color_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
            color_dc = color_dc.reshape((xyz.shape[0], 3, 1))
            color_rest = color_rest.reshape((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        else:
            color_dc = np.zeros((xyz.shape[0], 3, 1))
            color_rest = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._color_features_dc = nn.Parameter(torch.tensor(color_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._color_features_rest = nn.Parameter(torch.tensor(color_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._albedo_init = nn.Parameter(torch.tensor(albedo_init, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic_init = nn.Parameter(torch.tensor(metallic_init, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness_init = nn.Parameter(torch.tensor(roughness_init, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        将参数张量替换进优化器，同时保持一阶/二阶动量状态与组名一致。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        按掩码对优化器参数裁剪（排除HDR相关组），同步动量状态。
        用于点的剪枝后保持优化器一致性。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "hdr" in group["name"]:
                continue
            param_tensor = group['params'][0]
            # 对齐掩码与参数的第一维长度，防御潜在不一致
            if mask.shape[0] != param_tensor.shape[0]:
                if mask.shape[0] > param_tensor.shape[0]:
                    local_mask = mask[:param_tensor.shape[0]]
                else:
                    pad = torch.zeros((param_tensor.shape[0] - mask.shape[0],), dtype=torch.bool, device=mask.device)
                    local_mask = torch.cat([mask, pad], dim=0)
            else:
                local_mask = mask

            stored_state = self.optimizer.state.get(param_tensor, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][local_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][local_mask]

                del self.optimizer.state[param_tensor]
                group["params"][0] = nn.Parameter((param_tensor[local_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(param_tensor[local_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        根据布尔掩码删除点（保留为 ~mask），并更新所有相关可训练张量与缓存。
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._color_features_dc = optimizable_tensors["c_dc"]
        self._color_features_rest = optimizable_tensors["c_rest"]

        self._albedo_init = optimizable_tensors["albedo_init"]
        self._metallic_init = optimizable_tensors["metallic_init"]
        self._roughness_init = optimizable_tensors["roughness_init"]
        self.diffuse_occ = self.diffuse_occ[valid_points_mask]

        self.visibility_score = self.visibility_score[valid_points_mask]
        self.aniso_score = self.aniso_score[valid_points_mask]
        self.fg_score = self.fg_score[valid_points_mask]
        self.fg_mask = self.fg_mask[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # prune albedo accumulators (keep running sums; do not reset elsewhere)
        self.albedo_grad_accum = self.albedo_grad_accum[valid_points_mask]
        self.albedo_grad_denom = self.albedo_grad_denom[valid_points_mask]

    def prune_floaters_by_knn(self, threshold):
        """
        通过KNN距离移除离群点（浮点）。
        计算每个点到其最近邻的距离，若大于阈值则判定为离群点并进行剪枝。
        """
        with torch.no_grad():
            points = self.get_xyz
            if points.shape[0] < 2:  # KNN需要至少2个点
                return

            # distCUDA2 计算到1-最近邻的平方距离
            # 输入张量必须是连续的
            distances_sq = distCUDA2(points.contiguous())
            distances = torch.sqrt(distances_sq.squeeze(-1))  # Squeeze to shape (N,)

            # 创建距离大于阈值的点的掩码
            outlier_mask = (distances > threshold)

            if torch.any(outlier_mask):
                num_outliers = outlier_mask.sum().item()
                print(f"\n[KNN Pruning] 移除了 {num_outliers} 个离群点 (距离 > {threshold:.4f}).")
                self.prune_points(outlier_mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新点的参数拼接到优化器参数组中，并扩展动量状态到相同形状。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "hdr" in group["name"]:
                continue
            assert len(group["params"]) == 1
            # 如果本次未提供该组的扩展张量，则跳过
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_albedo_init, new_metallic_init, new_roughness_init, new_diffuse_occ, new_color_dc=None, new_color_rest=None, new_fg_score=None, new_fg_mask=None):
        """
        在密度化（新增点）后，统一将新增参数合并进当前模型与优化器中，并重置统计缓存。
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "albedo_init": new_albedo_init,
        "metallic_init": new_metallic_init,
        "roughness_init": new_roughness_init,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._albedo_init = optimizable_tensors["albedo_init"]
        self._metallic_init = optimizable_tensors["metallic_init"]
        self._roughness_init = optimizable_tensors["roughness_init"]
        self.diffuse_occ = torch.cat([self.diffuse_occ, new_diffuse_occ], 0)

        # Extend color SH robustly: extend each group independently, creating zero-like
        # extensions when inputs are missing or have zero-sized SH-rest.
        num_new = int(new_xyz.shape[0])
        device = self._xyz.device
        # c_dc
        if hasattr(self, "_color_features_dc") and self._color_features_dc is not None:
            if new_color_dc is None:
                # 创建与现有形状匹配的零扩展 (num_new, 1, 3)
                shape_dc = (num_new, self._color_features_dc.shape[1], self._color_features_dc.shape[2])
                new_color_dc = torch.zeros(shape_dc, dtype=self._color_features_dc.dtype, device=device)
            color_dc_tensors = self.cat_tensors_to_optimizer({"c_dc": new_color_dc})
            if "c_dc" in color_dc_tensors:
                self._color_features_dc = color_dc_tensors["c_dc"]

        # c_rest（可能为0宽度，但仍需在第一维对齐）
        if hasattr(self, "_color_features_rest") and self._color_features_rest is not None:
            if new_color_rest is None:
                shape_rest = (num_new, self._color_features_rest.shape[1], self._color_features_rest.shape[2])
                # 即使第二维为0，这里也会创建 (num_new, 0, 3) 的空张量，以保持第一维一致
                new_color_rest = torch.zeros(shape_rest, dtype=self._color_features_rest.dtype, device=device)
            color_rest_tensors = self.cat_tensors_to_optimizer({"c_rest": new_color_rest})
            if "c_rest" in color_rest_tensors:
                self._color_features_rest = color_rest_tensors["c_rest"]

        # Debug: record counts of added points before resetting caches
        added = int(new_xyz.shape[0])
        if hasattr(self, "_adding_source") and self._adding_source == "split":
            self.debug_split_added += added
        elif hasattr(self, "_adding_source") and self._adding_source == "clone":
            self.debug_clone_added += added
        elif hasattr(self, "_adding_source") and self._adding_source == "extra_split":
            self.debug_extra_split_added += added
        self._adding_source = None

        self.visibility_score = torch.cat([self.visibility_score, torch.zeros(new_xyz.shape[0], device="cuda")], 0)
        self.aniso_score = torch.cat([self.aniso_score, torch.zeros(new_xyz.shape[0], device="cuda")], 0)
        # foreground score and mask extend (inherit from sources if provided)
        if new_fg_score is not None:
            self.fg_score = torch.cat([self.fg_score, new_fg_score], 0)
        else:
            self.fg_score = torch.cat([self.fg_score, torch.zeros(new_xyz.shape[0], device="cuda")], 0)
        if new_fg_mask is not None:
            self.fg_mask = torch.cat([self.fg_mask, new_fg_mask], 0)
        else:
            self.fg_mask = torch.cat([self.fg_mask, torch.zeros(new_xyz.shape[0], dtype=torch.bool, device="cuda")], 0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Extend albedo accumulators for new points, keep history for existing points
        self.albedo_grad_accum = torch.cat([
            self.albedo_grad_accum,
            torch.zeros((new_xyz.shape[0], 1), device="cuda")
        ], dim=0)
        self.albedo_grad_denom = torch.cat([
            self.albedo_grad_denom,
            torch.zeros((new_xyz.shape[0], 1), device="cuda")
        ], dim=0)

    def extra_densify_and_prune(self, visibility_score: torch.Tensor, albedo_gradient: torch.Tensor = None, scene_extent: float = 1.0, split_times: int = 2, prune_ratio_limit: float = 0.05, prune_low_visibility: bool = True):
        """混合驱动的物理致密化（基于分位数，分裂实现）：
        分裂集合：高可见（前 vs_top_percent）且高AG（前 ag_top_percent，且 ≥ 固定阈值）。
        剪枝集合：低可见（后 vs_low_percent）。
        生成新点时，采用与 densify_and_split 一致的分裂方式：
        在父点的椭球坐标系中按尺度采样扰动并映射回世界坐标；
        新点缩放按 densify_and_split 的规则缩小。
        """
        with torch.no_grad():
            P = self.get_xyz.shape[0]
            if P == 0:
                return

            # 设备与数据准备
            device = self.get_xyz.device
            if albedo_gradient is None or albedo_gradient.numel() == 0:
                albedo_gradient = torch.zeros_like(visibility_score)

            # 百分位阈值（分位数）
            vs_top = max(1e-6, min(0.999999, float(self.vs_top_percent)))
            ag_top = max(1e-6, min(0.999999, float(self.ag_top_percent)))
            vs_low = max(1e-6, min(0.999999, float(self.vs_low_percent)))

            vs_high_th = torch.quantile(visibility_score.float(), 1.0 - vs_top) if visibility_score.numel() > 0 else torch.tensor(0.0, device=device)
            ag_high_th = torch.quantile(albedo_gradient.float(), 1.0 - ag_top) if albedo_gradient.numel() > 0 else torch.tensor(0.0, device=device)
            vs_low_th = torch.quantile(visibility_score.float(), vs_low) if visibility_score.numel() > 0 else torch.tensor(0.0, device=device)

            # 候选致密化集合：高可见 ∩ 高AG(分位数) ∩ 高AG(固定阈值)
            ag_fixed_th = float(getattr(self, "ag_fixed_threshold", 0.0))
            densify_candidates = (visibility_score >= vs_high_th) & (albedo_gradient >= ag_high_th) & (albedo_gradient >= ag_fixed_th)

            # 进一步区分为 split 和 clone
            scales = self.get_scaling[densify_candidates]
            scale_threshold = self.percent_dense * scene_extent
            
            split_mask_local = torch.max(scales, dim=1).values > scale_threshold
            clone_mask_local = torch.max(scales, dim=1).values <= scale_threshold

            # 将局部掩码映射回全局索引
            densify_indices = torch.where(densify_candidates)[0]
            split_mask = torch.zeros_like(densify_candidates)
            clone_mask = torch.zeros_like(densify_candidates)
            
            if densify_indices.numel() > 0:
                split_mask[densify_indices[split_mask_local]] = True
                clone_mask[densify_indices[clone_mask_local]] = True

            num_split = int(split_mask.sum().item())
            num_clone = int(clone_mask.sum().item())

            # 剪枝候选集合：低可见（后分位）
            prune_candidates = (visibility_score <= vs_low_th)
            
            print(f"[extra_densification_and_prune] 克隆集合={num_clone}, 分裂集合={num_split}, 剪枝候选={int(prune_candidates.sum().item())}")

            # 执行分裂
            if torch.any(split_mask):
                N = split_times
                stds = self.get_scaling[split_mask].repeat(N, 1)
                means = torch.zeros((stds.size(0), 3), device=device)
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[split_mask]).repeat(N, 1, 1)
                new_xyz_split = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[split_mask].repeat(N, 1)
                new_scaling_split = self.scaling_inverse_activation(self.get_scaling[split_mask].repeat(N, 1) / (0.8 * N))
                # 继承其他属性
                new_rotation_split = self._rotation[split_mask].repeat(N, 1)
                new_features_dc_split = self._features_dc[split_mask].repeat(N, 1, 1)
                new_features_rest_split = self._features_rest[split_mask].repeat(N, 1, 1)
                new_opacity_split = self._opacity[split_mask].repeat(N, 1)
                new_albedo_init_split = self._albedo_init[split_mask].repeat(N, 1)
                new_metallic_init_split = self._metallic_init[split_mask].repeat(N, 1)
                new_roughness_init_split = self._roughness_init[split_mask].repeat(N, 1)
                new_diffuse_occ_split = self.diffuse_occ[split_mask].repeat(N, 1) if self.diffuse_occ.numel() > 0 else torch.zeros((new_xyz_split.shape[0], self.diffuse_sample_num), device=device)
                new_color_dc_split = self._color_features_dc[split_mask].repeat(N, 1, 1) if self._color_features_dc.numel() > 0 else None
                new_color_rest_split = self._color_features_rest[split_mask].repeat(N, 1, 1) if self._color_features_rest.numel() > 0 else None
            else:
                # 保持变量定义以便于后续拼接
                new_xyz_split = torch.empty((0, 3), device=device)
                new_features_dc_split = torch.empty((0, self._features_dc.shape[1], self._features_dc.shape[2]), device=device)
                new_features_rest_split = torch.empty((0, self._features_rest.shape[1], self._features_rest.shape[2]), device=device)
                new_opacity_split = torch.empty((0, 1), device=device)
                new_scaling_split = torch.empty((0, 3), device=device)
                new_rotation_split = torch.empty((0, 4), device=device)
                new_albedo_init_split = torch.empty((0, 3), device=device)
                new_metallic_init_split = torch.empty((0, 1), device=device)
                new_roughness_init_split = torch.empty((0, 1), device=device)
                new_diffuse_occ_split = torch.empty((0, self.diffuse_sample_num), device=device)
                new_color_dc_split = torch.empty((0, self._color_features_dc.shape[1], self._color_features_dc.shape[2]), device=device) if self._color_features_dc is not None and self._color_features_dc.numel() > 0 else None
                new_color_rest_split = torch.empty((0, self._color_features_rest.shape[1], self._color_features_rest.shape[2]), device=device) if self._color_features_rest is not None and self._color_features_rest.numel() > 0 else None
            
            # 执行克隆
            if torch.any(clone_mask):
                new_xyz_clone = self._xyz[clone_mask]
                new_scaling_clone = self._scaling[clone_mask] # clone 不改变大小
                new_rotation_clone = self._rotation[clone_mask]
                new_features_dc_clone = self._features_dc[clone_mask]
                new_features_rest_clone = self._features_rest[clone_mask]
                new_opacity_clone = self._opacity[clone_mask]
                new_albedo_init_clone = self._albedo_init[clone_mask]
                new_metallic_init_clone = self._metallic_init[clone_mask]
                new_roughness_init_clone = self._roughness_init[clone_mask]
                new_diffuse_occ_clone = self.diffuse_occ[clone_mask] if self.diffuse_occ.numel() > 0 else torch.zeros((new_xyz_clone.shape[0], self.diffuse_sample_num), device=device)
                new_color_dc_clone = self._color_features_dc[clone_mask] if self._color_features_dc.numel() > 0 else None
                new_color_rest_clone = self._color_features_rest[clone_mask] if self._color_features_rest.numel() > 0 else None
            else:
                new_xyz_clone = torch.empty((0, 3), device=device)
                new_features_dc_clone = torch.empty((0, self._features_dc.shape[1], self._features_dc.shape[2]), device=device)
                new_features_rest_clone = torch.empty((0, self._features_rest.shape[1], self._features_rest.shape[2]), device=device)
                new_opacity_clone = torch.empty((0, 1), device=device)
                new_scaling_clone = torch.empty((0, 3), device=device)
                new_rotation_clone = torch.empty((0, 4), device=device)
                new_albedo_init_clone = torch.empty((0, 3), device=device)
                new_metallic_init_clone = torch.empty((0, 1), device=device)
                new_roughness_init_clone = torch.empty((0, 1), device=device)
                new_diffuse_occ_clone = torch.empty((0, self.diffuse_sample_num), device=device)
                new_color_dc_clone = torch.empty((0, self._color_features_dc.shape[1], self._color_features_dc.shape[2]), device=device) if self._color_features_dc is not None and self._color_features_dc.numel() > 0 else None
                new_color_rest_clone = torch.empty((0, self._color_features_rest.shape[1], self._color_features_rest.shape[2]), device=device) if self._color_features_rest is not None and self._color_features_rest.numel() > 0 else None

            # 合并所有新点
            if num_split > 0 or num_clone > 0:
                new_xyz = torch.cat((new_xyz_split, new_xyz_clone), dim=0)
                new_scaling = torch.cat((new_scaling_split, new_scaling_clone), dim=0)
                new_rotation = torch.cat((new_rotation_split, new_rotation_clone), dim=0)
                new_features_dc = torch.cat((new_features_dc_split, new_features_dc_clone), dim=0)
                new_features_rest = torch.cat((new_features_rest_split, new_features_rest_clone), dim=0)
                new_opacity = torch.cat((new_opacity_split, new_opacity_clone), dim=0)
                new_albedo_init = torch.cat((new_albedo_init_split, new_albedo_init_clone), dim=0)
                new_metallic_init = torch.cat((new_metallic_init_split, new_metallic_init_clone), dim=0)
                new_roughness_init = torch.cat((new_roughness_init_split, new_roughness_init_clone), dim=0)
                new_diffuse_occ = torch.cat((new_diffuse_occ_split, new_diffuse_occ_clone), dim=0)
                
                # 安全地合并 color SH
                def safe_cat(a, b):
                    if a is None and b is None: return None
                    a_empty = (a is None or a.numel() == 0)
                    b_empty = (b is None or b.numel() == 0)
                    if a_empty and b_empty: return torch.empty(0)
                    if a_empty: return b
                    if b_empty: return a
                    return torch.cat((a, b), dim=0)

                new_color_dc = safe_cat(new_color_dc_split, new_color_dc_clone)
                new_color_rest = safe_cat(new_color_rest_split, new_color_rest_clone)

                self._adding_source = "extra_split_clone"
                self.debug_extra_split_added += new_xyz.shape[0] # 记录总新增
                self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_albedo_init, new_metallic_init, new_roughness_init, new_diffuse_occ, new_color_dc, new_color_rest)

            # --- 剪枝逻辑重构 ---
            P_new = self.get_xyz.shape[0]
            
            # 1. 确定需要剪枝的父点 (来自致密化)
            prune_from_densify = torch.logical_or(split_mask, clone_mask) # size P
            
            # 2. 确定低可见性候选剪枝点
            prune_low_vis_candidates = prune_candidates # size P

            # 3. 如果增加了新点，将剪枝掩码填充到新尺寸
            if P_new > P:
                pad = torch.zeros((P_new - P,), dtype=torch.bool, device=device)
                prune_from_densify = torch.cat([prune_from_densify, pad], dim=0)
                prune_low_vis_candidates = torch.cat([prune_low_vis_candidates, pad], dim=0)
            
            # 4. & 5. 合并剪枝逻辑
            if prune_low_visibility:
                # 4. 对低可见性点应用比例限制
                final_prune_low_vis_mask = torch.zeros_like(prune_low_vis_candidates)
                max_prune = int(prune_ratio_limit * P) # 限制基于原始点数
                num_can_prune = torch.sum(prune_low_vis_candidates).item()

                if num_can_prune > max_prune and max_prune > 0:
                    cand_idx = torch.where(prune_low_vis_candidates)[0]
                    perm = torch.randperm(cand_idx.numel(), device=device)
                    selected = cand_idx[perm[:max_prune]]
                    final_prune_low_vis_mask[selected] = True
                else:
                    final_prune_low_vis_mask = prune_low_vis_candidates
                
                # 5. 合并两类剪枝点：父点总是被剪，低可见性点按比例剪
                final_prune_mask = torch.logical_or(final_prune_low_vis_mask, prune_from_densify)
            else:
                final_prune_mask = prune_from_densify
            
            if torch.any(final_prune_mask):
                actual_pruned = int(final_prune_mask.sum().item())
                print(f"[extra_densification_and_prune] 实际剪枝数量={actual_pruned}")
                self.prune_points(final_prune_mask)
                self.debug_extra_pruned += actual_pruned

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        基于梯度阈值选择点并沿协方差椭球采样生成新点（分裂N次），随后剪枝原点：
        - 被选点需满足梯度较大且尺寸大于阈值
        - 新点参数继承并做尺度缩放
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_color_dc = self._color_features_dc[selected_pts_mask].repeat(N,1,1)
        new_color_rest = self._color_features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_albedo_init = self._albedo_init[selected_pts_mask].repeat(N,1)
        new_metallic_init = self._metallic_init[selected_pts_mask].repeat(N,1)
        new_roughness_init = self._roughness_init[selected_pts_mask].repeat(N,1)
        new_diffuse_occ = self.diffuse_occ[selected_pts_mask].repeat(N,1)
        # Inherit foreground score/mask from parents
        parent_fg_score = self.fg_score[selected_pts_mask].repeat(N)
        parent_fg_mask = self.fg_mask[selected_pts_mask].repeat(N)

        print(f"[densify_and_split] 当前总数={self.get_xyz.shape[0]}")
        self._adding_source = "split"
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_albedo_init, new_metallic_init, new_roughness_init, new_diffuse_occ, new_color_dc, new_color_rest, parent_fg_score, parent_fg_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        print(f"[densify_and_split] 新增: split={new_xyz.shape[0]}")
        before = int(self.get_xyz.shape[0])
        self.prune_points(prune_filter)
        after = int(self.get_xyz.shape[0])
        self.debug_pruned += (before - after)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        基于梯度范数阈值选择较小点并克隆（复制）加入集合。
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_color_dc = self._color_features_dc[selected_pts_mask]
        new_color_rest = self._color_features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_albedo_init = self._albedo_init[selected_pts_mask]
        new_metallic_init = self._metallic_init[selected_pts_mask]
        new_roughness_init = self._roughness_init[selected_pts_mask]
        new_diffuse_occ = self.diffuse_occ[selected_pts_mask]
        # Inherit foreground score/mask from parents
        parent_fg_score = self.fg_score[selected_pts_mask]
        parent_fg_mask = self.fg_mask[selected_pts_mask]

        self._adding_source = "clone"
        print(f"[densify_and_clone] 当前总数={self.get_xyz.shape[0]}")
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_albedo_init, new_metallic_init, new_roughness_init, new_diffuse_occ, new_color_dc, new_color_rest, parent_fg_score, parent_fg_mask)
        print(f"[densify_and_clone] 新增: clone={new_xyz.shape[0]}")

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        综合执行克隆与分裂进行密度化，并依据不透明度与屏幕尺寸阈值进行剪枝。
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            # 支持传入 (screen_thresh, world_factor) 或单个 screen_thresh
            screen_thresh = None
            world_factor = 0.1
            if isinstance(max_screen_size, (tuple, list)) and len(max_screen_size) == 2:
                screen_thresh = float(max_screen_size[0])
                world_factor = float(max_screen_size[1])
            else:
                screen_thresh = float(max_screen_size)
            big_points_vs = self.max_radii2D > screen_thresh
            big_points_ws = self.get_scaling.max(dim=1).values > world_factor * float(extent)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        before = int(self.get_xyz.shape[0])
        self.prune_points(prune_mask)
        after = int(self.get_xyz.shape[0])
        self.debug_pruned += (before - after)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        统计视空间梯度，用于后续密度化的梯度阈值选择。
        只对当前可见（update_filter）的点累积。
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def update_foreground_scores(self, camera_center, visibility_filter):
        """前景打分：忽略可见性，所有点按 +1/distance 累积。"""
        # with torch.no_grad():
        #     xyz = self.get_xyz  # (P,3)
        #     camera_center = camera_center.to(xyz.device)
        #     distances = torch.norm(xyz - camera_center, dim=1)  # (P)
        #     self.fg_score += 1.0 / (distances + 1e-6)
            # 原实现（保留参考）：仅对可见点累积 1/distance
            # effective_filter = visibility_filter & (self.get_opacity.squeeze(-1) > 1e-4)
            # inc = torch.zeros_like(self.fg_score)
            # inc[effective_filter] = 1.0 / (distances[effective_filter] + 1e-6)
            # self.fg_score += inc
        pass

    def finalize_foreground_mask(self, percentile: float = 0.5, scene_center=None):
        """
        根据当前 fg_score 生成 fg_mask。percentile 表示按得分分位数阈值划分前景。
        """
        # with torch.no_grad():
        #     if self.fg_score.numel() == 0:
        #         return
        #     q = torch.quantile(self.fg_score, max(1e-6, min(0.999999, float(percentile))))
        #     self.fg_mask = (self.fg_score >= q)
        with torch.no_grad():
            if self._xyz.numel() == 0:
                return
            
            # 1. 确定场景中心。优先使用传入的基于相机位置的中心，否则回退到高斯点云的几何中心。
            if scene_center is None:
                center = torch.mean(self._xyz, dim=0)
            else:
                center = scene_center.to(self._xyz.device)
            
            # 2. 计算每个椭球到中心位置的距离
            distances = torch.norm(self._xyz - center, dim=1)
            
            # 3. 根据fg_percentile找到距离阈值
            #    percentile代表离中心最近的n%
            #    因此我们直接在距离上取分位数
            distance_threshold = torch.quantile(distances, max(1e-6, min(0.999999, float(percentile))))
            
            # 4. 距离小于等于阈值的为前景
            self.fg_mask = (distances <= distance_threshold)

    def add_albedo_grad_stats(self, rendered_albedo, update_filter):
        """累计每点的 albedo 梯度范数，用于混合驱动的材质应力信号。
        rendered_albedo: 渲染阶段得到的 per-Gaussian albedo 映射到屏幕后的聚合梯度源。
        这里简化使用 self._albedo_init 的梯度范数聚合到可见点上。
        """
        if self._albedo_init.grad is None:
            return
        grad_norm = torch.norm(self._albedo_init.grad, dim=-1, keepdim=True)  # [P,1]
        self.albedo_grad_accum[update_filter] += grad_norm[update_filter]
        self.albedo_grad_denom[update_filter] += 1

    def accumulate_scores(self, camera_center, visibility_filter):
        """在每个视角仅对参与成像的高斯累积可见性与各向异性得分。
        参与成像的判定：可见(由栅格化返回的 visibility_filter) 且 不近似全透明。
        """
        with torch.no_grad():
            xyz = self.get_xyz  # (P,3)
            # 确保相机中心与点云在同一设备
            camera_center = camera_center.to(xyz.device)
            distances = torch.norm(xyz - camera_center, dim=1)  # (P)
            # 进一步过滤：排除几乎全透明的高斯
            opacity_mask = (self.get_opacity.squeeze(-1) > 1e-4)
            effective_filter = visibility_filter & opacity_mask
            self.visibility_score[effective_filter] += 1.0 / (distances[effective_filter] + 1e-6)

            features_rest = self._features_rest.transpose(1, 2).contiguous()  # (P, F, SH-1)
            high_sh_norm = torch.norm(features_rest, dim=(1, 2))  # (P)
            # 作为间接光高频异常代理信号累积（不做可见性门控）
            self.aniso_score += high_sh_norm

            # 分数仅累积，不在此处触发额外 densification/prune