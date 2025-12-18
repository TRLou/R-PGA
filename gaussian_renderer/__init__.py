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
高斯渲染器入口

提供 render 函数用于将 3D 高斯点云（含SH或预计算颜色、材质等）在给定相机与管线配置下栅格化为图像。
返回包括渲染图像、深度、透明度以及（在特定阶段）法线、材质、分解光照/颜色等多种中间可视化结果。
"""

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import RGB2SH

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, random_bg_color = None, iteration=None, scaling_modifier = 1.0, is_train=None, first_stage_step=5000, second_stage_step=30000, remove_noise=False, hdr_rotation=False, color_sh_only: bool = False):
    """
    渲染场景主函数

    参数:
    - viewpoint_camera: 相机对象，包含视锥、投影、相机位姿、分辨率等
    - pc: 高斯模型(点云)，提供位置、缩放、旋转、不透明度、颜色/材质等
    - pipe: 渲染管线配置，含 debug、是否用Python计算协方差/SH等
    - bg_color: 背景颜色 (GPU Tensor)
    - random_bg_color: 随机背景颜色 (用于数据增强)，若为 None 则与 bg_color 相同
    - iteration: 当前迭代步，用于阶段控制与特定初始化/清零逻辑
    - scaling_modifier: 尺度修正系数，影响协方差/表观大小
    - is_train: 是否处于训练模式（控制颜色/材质计算分支）
    - first_stage_step: 第一阶段结束步数（仅使用初始化albedo等）
    - second_stage_step: 第二阶段结束步数（之后输出更多可视化/监督）
    - remove_noise: 是否在颜色/材质计算时去噪
    - hdr_rotation: 是否对环境贴图执行旋转

    返回:
    - 字典，键包括：
      - "render": 最终渲染图像 (C,H,W)
      - "depth": 深度图
      - "alpha": 透明度图
      - 以及在后续阶段/每隔若干步额外导出的法线/材质/分解颜色与光照/遮挡等
    注意: bg_color 必须在 GPU 上。
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 计算视场角正切，准备栅格化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 主渲染背景配置（固定背景）
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 随机背景配置（若提供），常用于训练时数据增强
    if random_bg_color is not None:
        random_raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=random_bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        random_rasterizer = GaussianRasterizer(raster_settings=random_raster_settings)
    else:
        random_rasterizer = rasterizer

    # 从模型读取基础属性
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    diffuse_color = None
    specular_indirect_light = None
    specular_indirect_color = None
    result = {}  # 初始化result为空字典
    if iteration <= first_stage_step:
        colors_precomp = pc.get_albedo_init
        # 若启用稳定过渡，不进行硬性置零
        if iteration == first_stage_step:
            if not getattr(viewpoint_camera, 'stable_stage_transition', False):
                pc._albedo_init.data = torch.zeros_like(pc._albedo_init)
    else:
        if iteration == second_stage_step+1:
            if not getattr(viewpoint_camera, 'stable_stage_transition', False):
                pc._albedo_init.data = torch.zeros_like(pc._albedo_init)
                pc._features_dc.data = torch.zeros_like(pc._features_dc)
                pc._features_rest.data = torch.zeros_like(pc._features_rest)
                pc._metallic_init.data = torch.rand_like(pc._metallic_init) * 0.2
                pc._roughness_init.data = torch.rand_like(pc._roughness_init)
        # 兼容不同模型：若支持 force_color_sh_only 参数则传入，否则回退
        try:
            # result = pc.compute_color(viewpoint_camera.camera_center, iteration, is_train, first_stage_step, second_stage_step, remove_noise, hdr_rotation, viewpoint_camera.exposure, force_color_sh_only=color_sh_only)
            result = pc.compute_color(viewpoint_camera.camera_center, iteration, is_train, first_stage_step, second_stage_step, remove_noise, hdr_rotation, viewpoint_camera.exposure)
        except TypeError:
            result = pc.compute_color(viewpoint_camera.camera_center, iteration, is_train, first_stage_step, second_stage_step, remove_noise, hdr_rotation, viewpoint_camera.exposure)

        colors_precomp = result["color"]
        albedo = result["albedo"]
        diffuse_albedo = result["diffuse_albedo"]
        diffuse_light = result["diffuse_light"]
        diffuse_color = result["diffuse_color"]
        specular_albedo = result["specular_albedo"]
        specular_indirect_light = result["specular_indirect_light"]
        specular_direct_light = result["specular_direct_light"]
        specular_indirect_color = result["specular_indirect_color"]
        specular_direct_color = result["specular_direct_color"]
        specular_light = result["specular_light"]
        specular_color = result["specular_color"]
        occ = result["occ"]

    # 主渲染：输出最终图像、半径、深度、透明度
    rendered_image, radii, depth, alpha = random_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # 额外可视化/监督结果默认置空，仅在阶段满足时渲染
    rendered_normal = None
    rendered_metallic = None
    rendered_roughness = None
    rendered_albedo = None
    rendered_diffuse_color = None
    rendered_specular_color = None
    rendered_diffuse_albedo=None
    rendered_specular_albedo=None
    rendered_diffuse_light = None
    rendered_specular_light = None
    rendered_specular_indirect_color = None
    rendered_specular_indirect_light = None
    rendered_specular_direct_light = None
    rendered_specular_direct_color = None
    rendered_occ = None
    rendered_fg_mask = None

    # --- 开始：临时可视化修改 ---
    rendered_color_fg_pbr = None
    rendered_color_bg_sh = None
    # --- 结束：临时可视化修改 ---

    # 第二阶段后：每步输出法线/材质等以提供监督和可视化
    if iteration > second_stage_step:
        render_normal = (pc.get_eigenvector + 1) / 2
        render_material = torch.cat([pc.get_metallic_init, pc.get_roughness_init.clamp(0.08, 0.5), torch.zeros((render_normal.shape[0],1), device="cuda")], -1)
        rendered_normal, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = render_normal,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        rendered_material, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = render_material,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        rendered_metallic = rendered_material[0:1,...].repeat(3,1,1)
        rendered_roughness = rendered_material[1:2,...].repeat(3,1,1)
        rendered_albedo, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = albedo,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    # 第一阶段结束至第二阶段：每500步导出若干可视化，避免频繁计算开销
    elif iteration > first_stage_step:
        if iteration % 500 == 0:
            with torch.no_grad():
                render_normal = (pc.get_eigenvector + 1) / 2
                render_material = torch.cat([pc.get_metallic_init, pc.get_roughness_init.clamp(0.08, 0.5), torch.zeros((render_normal.shape[0],1), device="cuda")], -1)
                rendered_normal, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = render_normal,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_material, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = render_material,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_metallic = rendered_material[0:1,...].repeat(3,1,1)
                rendered_roughness = rendered_material[1:2,...].repeat(3,1,1)
                rendered_albedo, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = albedo,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)

    # 每500步（且有对应数据）导出漫反射/镜面分解的颜色、反照率、光照等
    if iteration % 500 == 0:
        with torch.no_grad():
            if diffuse_color is not None:
                rendered_diffuse_color, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = diffuse_color,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_specular_color, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_color,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_diffuse_albedo, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = diffuse_albedo,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_specular_albedo, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_albedo,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_diffuse_light, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = diffuse_light,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_specular_light, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_light,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
            if specular_indirect_color is not None:
                rendered_specular_indirect_color, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = specular_indirect_color,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
                rendered_specular_direct_color, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = specular_direct_color,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
            if specular_indirect_light is not None:
                rendered_specular_indirect_light, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = specular_indirect_light,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
                rendered_specular_direct_light, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_direct_light,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_occ, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = occ.repeat(1,3),
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
            # Render foreground mask image if available and mask is in use
            if hasattr(pc, 'fg_mask') and getattr(pc, 'use_fg_mask', False) and pc.fg_mask.numel() == means3D.shape[0]:
                fg_color = pc.fg_mask.float().unsqueeze(-1).repeat(1,3)
                rendered_fg_mask, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = fg_color,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)

            # --- 开始：临时可视化修改 ---
            color_fg_pbr = result.get("color_fg_pbr")
            if color_fg_pbr is not None:
                rendered_color_fg_pbr, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = color_fg_pbr,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)

            color_bg_sh = result.get("color_bg_sh")
            if color_bg_sh is not None:
                rendered_color_bg_sh, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = color_bg_sh,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
            # --- 结束：临时可视化修改 ---

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "alpha": alpha,
            "rendered_normal": rendered_normal,
            "rendered_albedo": rendered_albedo,
            "rendered_metallic": rendered_metallic,
            "rendered_roughness": rendered_roughness,
            "rendered_diffuse_color": rendered_diffuse_color,
            "rendered_specular_color": rendered_specular_color,
            "rendered_diffuse_light": rendered_diffuse_light,
            "rendered_specular_light": rendered_specular_light,
            "rendered_diffuse_albedo": rendered_diffuse_albedo,
            "rendered_specular_albedo": rendered_specular_albedo,
            "rendered_specular_indirect_light": rendered_specular_indirect_light,
            "rendered_specular_direct_light": rendered_specular_direct_light,
            "rendered_specular_indirect_color": rendered_specular_indirect_color,
            "rendered_specular_direct_color": rendered_specular_direct_color,
            "rendered_occ": rendered_occ,
            "rendered_fg_mask": rendered_fg_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            # --- 开始：临时可视化修改 ---
            "rendered_color_fg_pbr": rendered_color_fg_pbr,
            "rendered_color_bg_sh": rendered_color_bg_sh
            # --- 结束：临时可视化修改 ---
            }
