import os
import csv
from typing import Dict, List, Tuple

import concurrent.futures as cf
import numpy as np

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# 导入 BEV/Range 转换器
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from radar_image_converter.converters import BEVConverter, RangeImageConverter
    from radar_image_converter.utils import normalize_image
except ImportError:
    from preprocess.radar_image_converter.converters import BEVConverter, RangeImageConverter
    from preprocess.radar_image_converter.utils import normalize_image

from PIL import Image


def _rotation_matrix_from_rpy_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """根据 SciPy 的 'xyz' 欧拉角约定重建旋转矩阵：R = Rx(roll) @ Ry(pitch) @ Rz(yaw)。"""
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return (Rx @ Ry) @ Rz


def load_gps_pose_entries(csv_path: str) -> List[Dict]:
    """从 gps.csv 中加载位姿"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"GPS pose file not found: {csv_path}")

    entries: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"timestamp", "northing", "easting", "height", "roll", "pitch", "yaw"}
        if reader.fieldnames is None or not required_fields.issubset(set(reader.fieldnames)):
            raise ValueError(
                "GPS CSV 缺少必要字段，期望包含: timestamp,northing,easting,height,roll,pitch,yaw"
            )

        for row in reader:
            try:
                ts = float(row["timestamp"])
                northing = float(row["northing"])
                easting = float(row["easting"])
                height = float(row["height"])
                roll = float(row["roll"])
                pitch = float(row["pitch"])
                yaw = float(row["yaw"])
            except (TypeError, ValueError):
                continue

            T = np.eye(4, dtype=float)
            T[:3, :3] = _rotation_matrix_from_rpy_xyz(roll, pitch, yaw)
            T[:3, 3] = np.array([easting, northing, height], dtype=float)

            entries.append({
                "timestamp": ts,
                "T": T,
                "t": T[:3, 3].copy(),
                "rpy": (roll, pitch, yaw),
            })

    entries.sort(key=lambda item: item["timestamp"])
    return entries


def find_nearest_entry(ts: float, entries: List[Dict], max_diff: float = 0.1) -> Dict:
    """根据时间戳找最近的 GPS 位姿"""
    if not entries:
        return None
    idx = min(range(len(entries)), key=lambda i: abs(entries[i]["timestamp"] - ts))
    if abs(entries[idx]["timestamp"] - ts) > max_diff:
        return None
    return entries[idx]


def load_radar_npy(npy_path: str) -> np.ndarray:
    """加载 Radar NPY 文件，返回 (N, 5) 数组：[x, y, z, doppler, intensity]"""
    return np.load(npy_path).astype(np.float32)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """坐标变换：points = T @ [x, y, z, 1]^T"""
    if points.size == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    pts_homo = np.hstack([points[:, :3], ones])
    transformed = (T @ pts_homo.T).T
    return np.hstack([transformed[:, :3], points[:, 3:]])  # 保留属性列


def mask_points_in_fov(points: np.ndarray, max_range: float = 120.0) -> np.ndarray:
    """简单的 FOV 过滤：只保留范围内的点"""
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    return r <= max_range


def run_for_base_dir(base_dir: str) -> None:
    preprocessed_base_dir = f"{base_dir}_preprocessed"
    radar_dir = f"{preprocessed_base_dir}/pointclouds"
    gps_csv_path = f"{preprocessed_base_dir}/gps.csv"
    radar_calib_path = f"{base_dir}/body_T_oculii.txt"

    if not os.path.isdir(preprocessed_base_dir):
        raise RuntimeError(f"预处理目录不存在: {preprocessed_base_dir}")
    if not os.path.isdir(radar_dir):
        raise RuntimeError(f"Radar 目录不存在: {radar_dir}")

    # 参数配置
    submap_window_size = 7           # 子地图聚合窗口帧数
    submap_stride = 1                # 中心帧步长
    max_range_m = 120.0
    max_time_diff = 0.1

    bev_resolution_m = 0.5
    bev_x_range_m = (0.0, 120.0)
    bev_y_range_m = (-60.0, 60.0)
    bev_height_range = (-5.0, 5.0)

    # BEV/Range 输出参数
    generate_png_images = True
    norm_method = 'minmax'
    azimuth_bins = 512
    elevation_bins = 64

    batch_num_workers = 4

    if submap_window_size <= 0 or submap_window_size % 2 == 0:
        raise ValueError("submap_window_size 必须为正奇数")

    submap_half = submap_window_size // 2

    print("Loading GPS-based poses...")
    entries = load_gps_pose_entries(gps_csv_path)
    if not entries:
        raise RuntimeError("位姿文件为空或无法解析有效条目。")

    print("Loading Radar extrinsics...")
    T_body_radar = np.loadtxt(radar_calib_path, dtype=float).reshape(4, 4)

    # 预计算位姿矩阵
    for entry in entries:
        U_T_body = entry["T"]
        entry["T_body"] = U_T_body
        entry["T_radar"] = U_T_body @ T_body_radar

    # 加载 Radar 文件列表
    radar_files_all = sorted(
        [f for f in os.listdir(radar_dir) if f.endswith(".npy")],
        key=lambda f: float(os.path.splitext(f)[0]),
    )
    if not radar_files_all:
        raise RuntimeError("Radar 目录下未找到 .npy 文件。")

    if len(radar_files_all) < submap_window_size:
        print(
            f"可用雷达帧数量不足以构建窗口 (需要 {submap_window_size}, 实际 {len(radar_files_all)})，跳过。"
        )
        return

    # 创建输出目录
    radar_bev_dir = os.path.join(preprocessed_base_dir, "radar_submap_bev_images")
    radar_range_dir = os.path.join(preprocessed_base_dir, "radar_submap_range_images")
    os.makedirs(radar_bev_dir, exist_ok=True)
    os.makedirs(radar_range_dir, exist_ok=True)

    submap_indices = list(range(submap_half, len(radar_files_all) - submap_half, submap_stride))
    if not submap_indices:
        print("未生成任何 submap 中心索引，检查窗口与步长配置。")
        return

    def process_one_submap(center_idx: int) -> int:
        try:
            center_file = radar_files_all[center_idx]
            ts_center = float(os.path.splitext(center_file)[0])
        except Exception:
            return 0

        # 找中心帧位姿
        center_pose = find_nearest_entry(ts_center, entries, max_time_diff)
        if center_pose is None:
            return 0

        U_T_R_center = center_pose.get("T_radar")
        if U_T_R_center is None:
            U_T_body_center = center_pose["T"]
            U_T_R_center = U_T_body_center @ T_body_radar
            center_pose["T_radar"] = U_T_R_center

        # 聚合子地图窗口内的所有 Radar 点
        window_start = center_idx - submap_half
        window_end = center_idx + submap_half

        radar_pts_world_list: List[np.ndarray] = []
        for ridx in range(window_start, window_end + 1):
            rf = radar_files_all[ridx]
            rp = os.path.join(radar_dir, rf)
            try:
                ts_r = float(os.path.splitext(rf)[0])
            except Exception:
                continue

            # 找该帧的位姿
            pose_r = find_nearest_entry(ts_r, entries, max_time_diff)
            if pose_r is None:
                continue

            # 加载 Radar 点云
            try:
                pts_radar = load_radar_npy(rp)
            except Exception:
                continue

            if pts_radar.size == 0:
                continue

            # 变换到世界坐标系
            U_T_R_i = pose_r.get("T_radar")
            if U_T_R_i is None:
                U_T_body_r = pose_r["T"]
                U_T_R_i = U_T_body_r @ T_body_radar
                pose_r["T_radar"] = U_T_R_i
            
            pts_world = transform_points(U_T_R_i, pts_radar)
            radar_pts_world_list.append(pts_world)

        # 合并所有帧的点
        if radar_pts_world_list:
            radar_pts_world = np.concatenate(radar_pts_world_list, axis=0)
            # FOV 过滤
            mask_fov = mask_points_in_fov(radar_pts_world, max_range_m)
            radar_pts_world = radar_pts_world[mask_fov]
        else:
            radar_pts_world = np.empty((0, 5), dtype=float)

        if radar_pts_world.size == 0:
            return 0

        # 变换回中心帧 Radar 坐标系
        T_radar_world_center = np.linalg.inv(U_T_R_center)
        radar_pts_radar = transform_points(T_radar_world_center, radar_pts_world)

        out_name_base = f"{ts_center:.9f}"

        # 生成 BEV 和 Range 图像
        try:
            # BEV 图像
            bev_converter = BEVConverter(
                x_range=bev_x_range_m, y_range=bev_y_range_m, resolution=bev_resolution_m,
                height_range=bev_height_range
            )
            bev_img = bev_converter(radar_pts_radar)
            if norm_method and norm_method != 'none':
                bev_img = normalize_image(bev_img, method=norm_method)
            bev_img = (np.clip(bev_img, 0, 1) * 255).astype(np.uint8)
            bev_pil = Image.fromarray(bev_img)
            bev_pil.save(os.path.join(radar_bev_dir, f"{out_name_base}_bev.png"))

            # Range 图像
            range_converter = RangeImageConverter(
                azimuth_bins=azimuth_bins, elevation_bins=elevation_bins, max_range=max_range_m
            )
            range_img = range_converter(radar_pts_radar)
            range_img = (np.clip(range_img, 0, 1) * 255).astype(np.uint8)
            range_pil = Image.fromarray(range_img)
            range_pil.save(os.path.join(radar_range_dir, f"{out_name_base}_range.png"))

            return 1
        except Exception as e:
            print(f"  ⚠️  生成图像失败 {out_name_base}: {e}")
            return 0

    saved_submaps = 0
    iterator = submap_indices
    pbar = None
    if _HAS_TQDM:
        pbar = tqdm(total=len(iterator), desc="Radar submap", unit="frame")

    with cf.ThreadPoolExecutor(max_workers=max(1, int(batch_num_workers))) as executor:
        futures = [executor.submit(process_one_submap, idx) for idx in iterator]
        for fut in cf.as_completed(futures):
            try:
                saved_submaps += int(fut.result() or 0)
            except Exception:
                pass
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    print(f"✨ Radar 子地图生成完成！总共生成: {saved_submaps} 个子地图")


def main():
    # 修改为你的数据路径
    base_dir = "/home/kaiyan/BEVPlace3/datasets/snail/radar/if_20240116_5"
    
    print(f"\n==== Processing dataset: {base_dir} ====")
    try:
        run_for_base_dir(base_dir)
    except Exception as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
