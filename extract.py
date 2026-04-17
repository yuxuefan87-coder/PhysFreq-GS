import h5py
import os

def surgical_mp4_extraction(h5_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    target_cameras = ['hama1', 'hama2', 'hand']
    
    print(f"[Engine] 正在切入 {h5_file_path} ...")
    with h5py.File(h5_file_path, 'r') as f:
        for cam in target_cameras:
            if cam in f:
                mp4_bytes = f[cam][()]
                out_video_path = os.path.join(output_dir, f"{cam}.mp4")
                with open(out_video_path, 'wb') as vid_file:
                    vid_file.write(mp4_bytes)
                print(f"[Success] 成功剥离视频: {out_video_path}")
            else:
                print(f"[Warning] 未找到视角 {cam}")

if __name__ == "__main__":
    surgical_mp4_extraction('./data/raw_h5/2025-01-09-13-59-54.h5', './data/raw_videos')