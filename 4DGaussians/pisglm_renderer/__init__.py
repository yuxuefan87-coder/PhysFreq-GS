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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. PI-SGLM Optimized Pipeline.
    """
    # 1. 解析相机参数与时间戳
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        time_scalar = viewpoint_camera.time
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
    else:
        raster_settings = viewpoint_camera['camera']
        time_scalar = viewpoint_camera['time']

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ==========================================
    # PI-SGLM 核心重构: 物理级的数据流旁路劫持
    # ==========================================
    if hasattr(pc, '_xyz_dynamic'):
        # --- A. 我们处于 PI-SGLM 物理拆分状态 ---
        
        # 提取静态常量 (永远不进入时序形变)
        shs_static = torch.cat([pc._features_dc_static, pc._features_rest_static], dim=1)
        
        # 预留 Phase 3 接口: 如果启用了解析 DCT 频域轨迹
        if hasattr(pc, 'calculate_dct_deformation'):
            deformed_xyz_dynamic = pc.calculate_dct_deformation(time_scalar)
            deformed_scales_dynamic = pc._scaling_dynamic
            deformed_rotations_dynamic = pc._rotation_dynamic
            deformed_opacity_dynamic = pc._opacity_dynamic
            deformed_shs_dynamic = torch.cat([pc._features_dc_dynamic, pc._features_rest_dynamic], dim=1)
            
        # 兼容 Phase 2 测试: MLP 网格 (仅喂入动态点)
        else:
            time_tensor = torch.tensor(time_scalar).to(pc._xyz_dynamic.device).repeat(pc._xyz_dynamic.shape[0], 1)
            shs_dynamic = torch.cat([pc._features_dc_dynamic, pc._features_rest_dynamic], dim=1)
            
            if "fine" in stage:
                deformed_xyz_dynamic, deformed_scales_dynamic, deformed_rotations_dynamic, deformed_opacity_dynamic, deformed_shs_dynamic = pc._deformation(
                    pc._xyz_dynamic, pc._scaling_dynamic, pc._rotation_dynamic, pc._opacity_dynamic, shs_dynamic, time_tensor
                )
            else:
                deformed_xyz_dynamic = pc._xyz_dynamic
                deformed_scales_dynamic = pc._scaling_dynamic
                deformed_rotations_dynamic = pc._rotation_dynamic
                deformed_opacity_dynamic = pc._opacity_dynamic
                deformed_shs_dynamic = shs_dynamic

        # --- B. ConcatBackward 零内存开销拼接 ---
        means3D_final = torch.cat([pc._xyz_static, deformed_xyz_dynamic], dim=0)
        scales_final = torch.cat([pc._scaling_static, deformed_scales_dynamic], dim=0)
        rotations_final = torch.cat([pc._rotation_static, deformed_rotations_dynamic], dim=0)
        opacity_final = torch.cat([pc._opacity_static, deformed_opacity_dynamic], dim=0)
        shs_final = torch.cat([shs_static, deformed_shs_dynamic], dim=0)

    else:
        # --- C. 降级回退 (Baseline 消融实验模式) ---
        means3D = pc.get_xyz
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity
        shs = pc.get_features
        time_tensor = torch.tensor(time_scalar).to(means3D.device).repeat(means3D.shape[0], 1)

        if "coarse" in stage:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        elif "fine" in stage:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
                means3D, scales, rotations, opacity, shs, time_tensor
            )
        else:
            raise NotImplementedError
    # ==========================================

    # 梯度追踪初始化 (必须在拼接之后进行以匹配 shape)
    screenspace_points = torch.zeros_like(means3D_final, dtype=means3D_final.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 激活函数处理
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    # SH 颜色转换
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    # 送入底层光栅化算子
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = screenspace_points,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}