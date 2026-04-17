import cv2
import os
from glob import glob

def create_video_from_images(image_dir, output_path, fps=30.0):
    frame_paths = sorted(
        glob(os.path.join(image_dir, "*.png")) + 
        glob(os.path.join(image_dir, "*.jpg"))
    )
    
    if not frame_paths:
        return False
        
    # 读取第一帧以获取分辨率元数据
    first_frame = cv2.imread(frame_paths[0])
    h, w, layers = first_frame.shape
    size = (w, h)
    
    print(f"正在压制 [{os.path.basename(image_dir)}] 序列，共 {len(frame_paths)} 帧，分辨率 {w}x{h}...")

    # 使用 mp4v 编码器，保证在绝大多数 Linux 服务器和本地播放器上的兼容性
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for p in frame_paths:
        img = cv2.imread(p)
        out.write(img)
        
    out.release()
    print(f"[{os.path.basename(image_dir)}] 视频已生成 -> {output_path}")
    return True

def main():
    # 锁定双目/多目序列的根目录
    base_dir = "/home/data/yuxuefan/PhysFreq-GS/data/Hamlyn_Rectified_Dataset/rectified01/rectified01"
    
    cam_dirs = sorted(glob(os.path.join(base_dir, "image*")))
    
    if not cam_dirs:
        print("未找到相机序列文件夹。")
        return

    # 通常内窥镜数据集的帧率为 25 或 30 fps
    fps = 30.0  
    
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        output_path = os.path.join(base_dir, f"GT_Motion_{cam_name}.mp4")
        create_video_from_images(cam_dir, output_path, fps)

if __name__ == "__main__":
    main()