import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import csv
from typing import List, Dict, Tuple

# 导入您之前提到的自车速度估计函数
# 假设该文件在路径中，或直接将逻辑集成进来
from ego_vel_estimate import estimate_ego_vel

def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """根据 rpy 重建旋转矩阵 (Rz @ Ry @ Rx)，匹配 scipy.as_euler('xyz')"""
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def load_gps_poses(csv_path: str) -> Dict[int, np.ndarray]:
    """修改：使用 int 作为 Key"""
    poses = {}
    if not os.path.exists(csv_path):
        print(f"❌ 错误：未找到位姿文件 {csv_path}")
        return poses
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 关键修改 1：直接转为 int（处理微秒级整数）
            ts = int(float(row['timestamp'])) 
            T = np.eye(4)
            T[:3, :3] = _rotation_matrix_from_rpy(float(row['roll']), float(row['pitch']), float(row['yaw']))
            T[:3, 3] = [float(row['easting']), float(row['northing']), float(row['height'])]
            poses[ts] = T
    print(f"✅ 成功加载位姿数量: {len(poses)}")
    return poses

def extract_rich_radar_submaps(src_folder, dst_folder, gps_csv_path, window_size=7):
    """
    改进版：实现多普勒去噪 + 子图堆叠对齐
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 1. 加载所有位姿
    all_poses = load_gps_poses(gps_csv_path)
    
    # 2. 筛选并排序 PCD 文件
    files = sorted([f for f in os.listdir(src_folder) if f.endswith('.pcd') and not f.startswith('.')])
    if len(files) < window_size:
        print("错误：PCD 文件数量不足以构成一个窗口")
        return

    print(f"开始处理：移除动态点 + 堆叠对齐 (窗口大小: {window_size})...")
    half_win = window_size // 2
    success_count = 0

    # 3. 以滑动窗口模式进行堆叠
    for i in tqdm(range(half_win, len(files) - half_win), desc="聚合子图"):
        center_file = files[i]

        
        # 在 pcd2npy_submap.py 中
        # 将文件名时间戳也转为微秒整数进行匹配
        center_ts = int(round(float(center_file[:-4]) * 1e6))
        
        if center_ts not in all_poses:
            continue
        
        U_T_center = all_poses[center_ts]
        inv_U_T_center = np.linalg.inv(U_T_center)
        
        submap_points = []

        # 遍历窗口内的每一帧进行去噪和对齐
        for j in range(i - half_win, i + half_win + 1):
            curr_file = files[j]
            curr_ts = int(round(float(curr_file[:-4]) * 1e6))
            pcd_path = os.path.join(src_folder, curr_file)

            try:
                # 使用 Tensor API 读取富信息
                pcd = o3d.t.io.read_point_cloud(pcd_path)
                if pcd.is_empty(): continue
                
                xyz = pcd.point["positions"].numpy().astype(np.float32)
                power = pcd.point["Power"].numpy().astype(np.float32).reshape(-1, 1)
                doppler = pcd.point["Doppler"].numpy().astype(np.float32).reshape(-1, 1)
                
                # 组合为 estimate_ego_vel 期望的格式 [x, y, z, doppler, power]
                # 注意：estimate_ego_vel 内部索引通常为 [x,y,z,doppler,power]
                raw_scan = np.hstack([xyz, doppler, power])
                
                # --- 功能 1: 移除多普勒移动点 ---
                # 利用 RANSAC 估计自车速度并滤除外点
                success, v_e, static_scan = estimate_ego_vel(raw_scan, retain_vel=True)
                if not success:
                    static_scan = raw_scan # 若估计失败，回退至原始数据
                
                # --- 功能 2: 对点云进行堆叠对齐 ---
                if curr_ts in all_poses:
                    U_T_curr = all_poses[curr_ts]
                    # 正确变换: 当前帧→世界→中心帧
                    # U_T_R 表示 World→Radar，所以 Radar_curr→Radar_center:
                    # T_rel = Rcenter_T_World @ World_T_Rcurr = inv(U_T_center) @ U_T_curr
                    inv_U_T_center = np.linalg.inv(U_T_center)
                    T_rel = inv_U_T_center @ U_T_curr
                    
                    curr_xyz = static_scan[:, :3]
                    h_pts = np.hstack([curr_xyz, np.ones((curr_xyz.shape[0], 1))])
                    # 变换到中心帧坐标系
                    aligned_xyz = (T_rel @ h_pts.T).T[:, :3]
                    
                    # 重新组合 [x, y, z, power, doppler] (保持您之前的输出格式)
                    aligned_rich = np.hstack([
                        aligned_xyz.astype(np.float32), 
                        static_scan[:, 4:5], # Power
                        static_scan[:, 3:4]  # Doppler (注意索引位置)
                    ])
                    submap_points.append(aligned_rich)

            except Exception as e:
                print(f"跳过文件 {curr_file}: {e}")
                continue

        if submap_points:
            # 合并所有对齐后的帧
            final_submap = np.vstack(submap_points)
            
            # 保存为 .npy
            npy_name = center_file.replace('.pcd', '.npy')
            np.save(os.path.join(dst_folder, npy_name), final_submap)
            success_count += 1

    print(f"\n处理完成！生成子图数：{success_count}")

# --- 配置路径 ---
input_dir = r'/home/kaiyan/BEVPlace3/datasets/snail/radar/if_20240116_5/pcd'
gps_csv = r'/home/kaiyan/BEVPlace3/datasets/snail/radar/if_20240116_5_preprocessed/gps.csv'
output_dir = os.path.join(os.path.dirname(input_dir), "submap_rich_npy")

if __name__ == "__main__":
    extract_rich_radar_submaps(input_dir, output_dir, gps_csv, window_size=15)