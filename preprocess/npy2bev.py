import numpy as np
import cv2
import os
import argparse
from tqdm import trange

# --- 1. 参数设置 ---
parser = argparse.ArgumentParser(description='Snail-Radar-Gen-BEV-Images')
parser.add_argument('--npy_path', type=str, 
                    default=r'/home/kaiyan/BEVPlace3/datasets/snail/radar/if_20240116_5/enhanced_rich_npy', 
                    help='path to your radar npy files')
parser.add_argument('--bev_save_path', type=str, 
                    default=r'/home/kaiyan/BEVPlace3/datasets/snail/radar/if_20240116_5/bev_image', 
                    help='path to save images')

# --- 2. 核心参数 ---
VOXEL_SIZE = 0.4
MAX_RANGE = 100.0
Z_MIN = -5.0
Z_MAX = 10.0

def getBEV_Radar(points_n5): 
    """
    输入: points_n5 -> (N, 5) [x, y, z, power, doppler]
    输出: (H, W, 3) BGR 图像
          Ch0 (Blue):  Density (点密度)
          Ch1 (Green): Height (最大高度)
          Ch2 (Red):   Intensity (最大强度)
    """
    
    # --- 预计算网格尺寸 ---
    x_min, x_max = -MAX_RANGE, MAX_RANGE
    y_min, y_max = -MAX_RANGE, MAX_RANGE
    
    grid_h = int(np.round((x_max - x_min) / VOXEL_SIZE))
    grid_w = int(np.round((y_max - y_min) / VOXEL_SIZE))
    
    # 初始化画布
    # Density 初始化为 0
    map_density = np.zeros((grid_h, grid_w), dtype=np.float32)
    # Intensity 初始化为 0
    map_intensity = np.zeros((grid_h, grid_w), dtype=np.float32)
    # Height 初始化为地面最低点 (Z_MIN)，确保能正确取到最大值
    map_height = np.full((grid_h, grid_w), Z_MIN, dtype=np.float32)
    
    # 1. 空间过滤
    mask = (np.abs(points_n5[:, 0]) < MAX_RANGE) & \
           (np.abs(points_n5[:, 1]) < MAX_RANGE) & \
           (points_n5[:, 2] > Z_MIN) & \
           (points_n5[:, 2] < Z_MAX)
    points = points_n5[mask]

    # 如果没有点，直接返回全黑图
    if points.shape[0] == 0:
        return np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # 2. 坐标映射 (World -> Image Grid)
    img_r = np.floor((x_max - points[:, 0]) / VOXEL_SIZE).astype(int)
    img_c = np.floor((y_max - points[:, 1]) / VOXEL_SIZE).astype(int)
    
    # 3. 边界安全检查
    img_r = np.clip(img_r, 0, grid_h - 1)
    img_c = np.clip(img_c, 0, grid_w - 1)
    
    # 提取特征值
    z_vals = points[:, 2]      # 高度
    power_vals = points[:, 3]  # 强度

    # 4. 填充 Density 通道 (累加)
    np.add.at(map_density, (img_r, img_c), 1.0)
    
    # 5. 填充 Intensity 和 Height 通道 (取最大值)
    flat_indices = img_r * grid_w + img_c
    
    # Intensity Max Pooling
    flat_intensity = map_intensity.ravel()
    np.maximum.at(flat_intensity, flat_indices, power_vals)
    map_intensity = flat_intensity.reshape(grid_h, grid_w)

    # Height Max Pooling (关键新增: 你的需求 G=高度)
    flat_height = map_height.ravel()
    np.maximum.at(flat_height, flat_indices, z_vals)
    map_height = flat_height.reshape(grid_h, grid_w)

    # 6. 归一化与增强
    # Density: Log 增强 -> 归一化到 0-255
    map_density = np.log1p(map_density)
    max_d = map_density.max()
    if max_d > 0: map_density = map_density / max_d * 255.0
    
    # Intensity: 截断 0~25 -> 归一化到 0-255
    map_intensity = np.clip(map_intensity, 0, 25)
    map_intensity = map_intensity / 25.0 * 255.0
    
    # Height: 线性映射 Z_MIN~Z_MAX -> 0-255
    # 例如 -5m -> 0, 10m -> 255
    height_range = Z_MAX - Z_MIN
    map_height = (map_height - Z_MIN) / height_range
    map_height = np.clip(map_height, 0, 1) * 255.0
    
    # 7. 组合通道 (BGR 顺序，适配 OpenCV)
    bev_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Channel 0 (Blue): 密度 (Density)
    bev_image[:, :, 0] = map_density.astype(np.uint8)
    
    # Channel 1 (Green): 高度 (Height)
    bev_image[:, :, 1] = map_height.astype(np.uint8)
    
    # Channel 2 (Red): 强度 (Intensity)
    bev_image[:, :, 2] = map_intensity.astype(np.uint8) 
    
    return bev_image

if __name__ == "__main__":
    args = parser.parse_args()
    
    if not os.path.exists(args.bev_save_path):
        os.makedirs(args.bev_save_path)
    
    npy_files = [f for f in os.listdir(args.npy_path) if f.endswith('.npy')]
    npy_files.sort()
    
    print(f"Start processing {len(npy_files)} files...")
    
    for i in trange(len(npy_files)):
        file_name = npy_files[i]
        file_path = os.path.join(args.npy_path, file_name)
        
        try:
            pc_data = np.load(file_path)
        except:
            continue

        if pc_data.ndim == 2 and pc_data.shape[1] >= 4:
            img = getBEV_Radar(pc_data)
            save_name = file_name.replace('.npy', '.png')
            cv2.imwrite(os.path.join(args.bev_save_path, save_name), img)
        else:
            pass

    print("Done!")