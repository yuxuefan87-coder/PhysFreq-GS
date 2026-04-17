import json
import os
import numpy as np

def build_4dgs_cameras_json(input_calib_path, output_json_path):
    with open(input_calib_path, 'r') as f:
        raw_calib = json.load(f)

    cameras_list = []
    
    # 注意大小写匹配，JSON里的键可能是首字母大写
    target_cameras = ['Hama1', 'Hama2', 'hand'] 
    
    for cam_id in target_cameras:
        # 兼容大小写不一致的脏数据
        dict_key = cam_id
        if cam_id not in raw_calib:
            # 尝试首字母大写或全小写
            alt_keys = [cam_id.lower(), cam_id.capitalize(), cam_id.upper()]
            for alt in alt_keys:
                if alt in raw_calib:
                    dict_key = alt
                    break
            else:
                continue
            
        cam_data_list = raw_calib[dict_key]
        
        # 核心修正：直接取索引 0 为平移，索引 1 为四元数
        t = np.array(cam_data_list[0]) # [x, y, z]
        q = np.array(cam_data_list[1]) # [qx, qy, qz, qw]
        
        # 将四元数 [qx, qy, qz, qw] 转换为 旋转矩阵 R
        qx, qy, qz, qw = q
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
            [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        # 构造 c2w 矩阵
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        
        # 转为 4DGS 需要的 w2c 格式
        w2c = np.linalg.inv(c2w)

        cam_node = {
            "id": cam_id.lower(),
            "img_name": f"{cam_id.lower()}_frame",
            "width": 1920,
            "height": 1080,
            "position": c2w[:3, 3].tolist(),
            "rotation": w2c[:3, :3].tolist(), 
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0
        }
        cameras_list.append(cam_node)

    with open(output_json_path, 'w') as f:
        json.dump(cameras_list, f, indent=4)
        
    print(f"[Engine] 位姿洗地完成，成功提取 {len(cameras_list)} 个机位，已输出至 {output_json_path}")

if __name__ == "__main__":
    build_4dgs_cameras_json('./data/2025-01-09-13-59-54_poses.json', './data/cameras.json')