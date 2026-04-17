import torch
import numpy as np
import cv2
import os
from glob import glob

def process_single_view(video_dir, output_dir):
    frame_paths = sorted(
        glob(os.path.join(video_dir, "*.png")) + 
        glob(os.path.join(video_dir, "*.jpg"))
    )
    if not frame_paths:
        return False
        
    cam_name = os.path.basename(video_dir)
    print(f"[{cam_name}] 加载 {len(frame_paths)} 帧，执行物理降噪与时空剥离...")

    frames = []
    for p in frame_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img_blurred = cv2.GaussianBlur(img, (9, 9), 0)
        frames.append(img_blurred.astype(np.float32) / 255.0)

    frames_tensor = torch.tensor(np.stack(frames)) # Shape: [T, H, W]
    variance = torch.var(frames_tensor, dim=0)

    # 维持 80% 静态背景的强力显存预算
    threshold = torch.quantile(variance.view(-1), 0.80).item()
    dynamic_mask_2d = (variance > threshold).numpy().astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dynamic_mask_2d = cv2.morphologyEx(dynamic_mask_2d, cv2.MORPH_OPEN, kernel)
    dynamic_mask_2d = cv2.morphologyEx(dynamic_mask_2d, cv2.MORPH_CLOSE, kernel)

    # 将各个视角的 mask 统一保存在根目录下，带上相机后缀
    output_path = os.path.join(output_dir, f"sglm_dynamic_mask_{cam_name}.png")
    cv2.imwrite(output_path, dynamic_mask_2d)
    
    static_ratio = 100 - (np.sum(dynamic_mask_2d > 0) / dynamic_mask_2d.size * 100)
    print(f"[{cam_name}] 掩码生成完毕 -> {output_path} | 静态背景冻结比: {static_ratio:.2f}%\n")
    return True

def main():
    # 锁定双目/多目序列的根目录
    base_dir = "/home/data/yuxuefan/PhysFreq-GS/data/images"
    
    # 自动检索所有以 image 开头的视角文件夹 (完全兼容 image01, image02)
    cam_dirs = sorted(glob(os.path.join(base_dir, "hama*")))
    
    if not cam_dirs:
        print("未找到任何相机视角文件夹，请检查根目录路径。")
        return

    print(f"检测到 {len(cam_dirs)} 个相机视角，正在执行多视角同步解耦...\n")

    for cam_dir in cam_dirs:
        process_single_view(cam_dir, base_dir)

    print("所有视角的动静隔离掩码已全部生成，随时准备接入 SGLM 3D 初始化反投影。")

if __name__ == "__main__":
    main()