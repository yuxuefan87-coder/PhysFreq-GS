import torch
import json
import numpy as np
from PIL import Image
from plyfile import PlyData

def generate_3d_dynamic_mask_multiview(ply_path, cameras_json_path, mask_dict, save_path):
    """
    mask_dict 格式: {"hama1": "path_to_mask1", "hama2": "path_to_mask2"}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载 3D 初始点云
    plydata = PlyData.read(ply_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    pts_3d = torch.tensor(xyz, dtype=torch.float32, device=device) # [N, 3]
    pts_3d_homo = torch.cat([pts_3d, torch.ones((pts_3d.shape[0], 1), device=device)], dim=1) # [N, 4]
    
    # 初始化全局 1D 动态掩码 (全 False)
    global_dynamic_mask = torch.zeros(pts_3d.shape[0], dtype=torch.bool, device=device)
    
    with open(cameras_json_path, 'r') as f:
        cam_data = json.load(f)
    
    # 2. 遍历所有提供的相机视角进行投射
    for cam_name, mask_path in mask_dict.items():
        # 加载当前视角的 2D 掩码 (单通道，>128 为动态)
        mask_img = np.array(Image.open(mask_path).convert('L'))
        H, W = mask_img.shape
        mask_tensor = torch.tensor(mask_img, device=device) > 128 
        
        # 提取当前相机的内外参
        cam = next(c for c in cam_data if cam_name in c["img_name"])
        W2C = torch.tensor(cam["W2C"], dtype=torch.float32, device=device) 
        K = torch.tensor(cam["K"], dtype=torch.float32, device=device)
        
        # 矩阵化前向投影
        pts_cam = (W2C @ pts_3d_homo.T).T # [N, 4]
        pts_2d_homo = (K @ pts_cam[:, :3].T).T # [N, 3]
        
        z = pts_2d_homo[:, 2]
        valid_depth = z > 0.01 # 必须在相机前方
        
        u = torch.round(pts_2d_homo[:, 0] / z).long()
        v = torch.round(pts_2d_homo[:, 1] / z).long()
        
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid_mask = valid_depth & valid_uv
        
        # 提取当前视角下判定为动态的点
        current_view_dynamic = torch.zeros(pts_3d.shape[0], dtype=torch.bool, device=device)
        current_view_dynamic[valid_mask] = mask_tensor[v[valid_mask], u[valid_mask]]
        
        # 核心防漏点：对多视角动态结果取逻辑或 (|)
        global_dynamic_mask = global_dynamic_mask | current_view_dynamic
        
        print(f"[PI-SGLM] 视角 {cam_name} 投射完毕。当前累积动态点数: {global_dynamic_mask.sum().item()}")

    # 3. 保存最终用于物理拆分的张量
    torch.save(global_dynamic_mask.cpu(), save_path)
    print(f"[PI-SGLM] 双目物理拆分掩码已生成，总点数: {pts_3d.shape[0]} | 最终动态点数: {global_dynamic_mask.sum().item()}")

# 运行示例
# mask_inputs = {
#     "hama1": "data/images/sglm_dynamic_mask_hama1.png",
#     "hama2": "data/images/sglm_dynamic_mask_hama2.png"
# }
# generate_3d_dynamic_mask_multiview("data/points3D.ply", "data/cameras.json", mask_inputs, "data/dynamic_mask_3d.pt")