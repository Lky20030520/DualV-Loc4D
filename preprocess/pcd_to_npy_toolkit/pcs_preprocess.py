#!/usr/bin/env python3
import argparse
import os
import numpy as np
import csv
from os.path import join
from tqdm import tqdm
import bisect
from multiprocessing import Pool
import shutil
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from ego_vel_estimate import estimate_ego_vel

# globals for stage 1
GLOBAL_RADAR_FOLDER = None
GLOBAL_SAVE_PC       = None
GLOBAL_W             = None
GLOBAL_GEN_ORIG      = None
GLOBAL_GEN_IMG       = None
GLOBAL_IMG_FOLDER    = None
GLOBAL_SORTED_IMAGES = None
GLOBAL_TARGET_POINTS = None
GLOBAL_INTERP_POSES  = None
GLOBAL_BT_L          = None
GLOBAL_BT_R          = None
GLOBAL_NORM_FUNC     = None
GLOBAL_MAXIMUM_RANGE = None

# globals for stage 2
GLOBAL_PREPROC = None
GLOBAL_INTERP_POSES = None
GLOBAL_BT_L = None
GLOBAL_BT_R = None
GLOBAL_W = None
GLOBAL_SAVE_PC = None
GLOBAL_GEN_ORIG = None
GLOBAL_GEN_IMG = None
GLOBAL_IMG_FOLDER = None
GLOBAL_SORTED_IMAGES = None
GLOBAL_TARGET_POINTS = None
GLOBAL_PCD_FILES = None
GLOBAL_NORM_FUNC = None


def init_stage1(radar_folder, save_pc_folder, W, generate_original,
                generate_images, image_folder, sorted_image_entries,
                target_points, interp_poses, Body_T_L, Body_T_R,
                norm_func, maximum_range):
    global GLOBAL_RADAR_FOLDER, GLOBAL_SAVE_PC, GLOBAL_W, GLOBAL_GEN_ORIG
    global GLOBAL_GEN_IMG, GLOBAL_IMG_FOLDER, GLOBAL_SORTED_IMAGES
    global GLOBAL_TARGET_POINTS, GLOBAL_INTERP_POSES, GLOBAL_BT_L
    global GLOBAL_BT_R, GLOBAL_NORM_FUNC, GLOBAL_MAXIMUM_RANGE

    GLOBAL_RADAR_FOLDER = radar_folder
    GLOBAL_SAVE_PC       = save_pc_folder
    GLOBAL_W             = W
    GLOBAL_GEN_ORIG      = generate_original
    GLOBAL_GEN_IMG       = generate_images
    GLOBAL_IMG_FOLDER    = image_folder
    GLOBAL_SORTED_IMAGES = sorted_image_entries
    GLOBAL_TARGET_POINTS = target_points
    GLOBAL_INTERP_POSES  = interp_poses
    GLOBAL_BT_L          = Body_T_L
    GLOBAL_BT_R          = Body_T_R
    GLOBAL_NORM_FUNC     = norm_func
    GLOBAL_MAXIMUM_RANGE = maximum_range


def init_stage2(preproc_list, interp_poses, Body_T_L, Body_T_R,
                W, save_pc_folder,
                generate_original, generate_images,
                image_folder, sorted_image_entries,
                target_points, pcd_files, norm_func):
    global GLOBAL_PREPROC, GLOBAL_INTERP_POSES, GLOBAL_BT_L, GLOBAL_BT_R
    global GLOBAL_W, GLOBAL_SAVE_PC, GLOBAL_GEN_ORIG, GLOBAL_GEN_IMG
    global GLOBAL_IMG_FOLDER, GLOBAL_SORTED_IMAGES, GLOBAL_TARGET_POINTS
    global GLOBAL_PCD_FILES, GLOBAL_NORM_FUNC

    GLOBAL_PREPROC       = {item[0]: item for item in preproc_list}
    GLOBAL_INTERP_POSES  = interp_poses
    GLOBAL_BT_L          = Body_T_L
    GLOBAL_BT_R          = Body_T_R
    GLOBAL_W             = W
    GLOBAL_SAVE_PC       = save_pc_folder
    GLOBAL_GEN_ORIG      = generate_original
    GLOBAL_GEN_IMG       = generate_images
    GLOBAL_IMG_FOLDER    = image_folder
    GLOBAL_SORTED_IMAGES = sorted_image_entries
    GLOBAL_TARGET_POINTS = target_points
    GLOBAL_PCD_FILES     = pcd_files
    GLOBAL_NORM_FUNC     = norm_func


def rot_slerp_batch(keytimes, keyquats, querytimes):
    keyrots = R.from_quat(keyquats)
    slerp = Slerp(keytimes, keyrots)
    interp_rots = slerp(querytimes)
    return interp_rots.as_quat()

def pos_interpolate_batch(keytimes, keypositions, querytimes):
    interp_positions = np.zeros((len(querytimes), 3))
    for i in range(keypositions.shape[1]):
        interp_positions[:, i] = np.interp(querytimes, keytimes, keypositions[:, i])
    return interp_positions

def interpolation(gt_times, gt_positions, gt_quats, radar_times):
    interp_positions = pos_interpolate_batch(gt_times, gt_positions, radar_times)
    interp_quats = rot_slerp_batch(gt_times, gt_quats, radar_times)
    poses = []
    for i, q_i in enumerate(interp_quats):
        quat_max = R.from_quat(q_i).as_matrix()
        xyz = interp_positions[i].reshape(3, 1)
        pose = np.hstack((quat_max, xyz))
        pose = np.vstack((pose, [0, 0, 0, 1]))
        poses.append(pose)
    return poses

def load_pcd(filename):
    with open(filename, 'rb') as file:
        fields = []
        while True:
            line = file.readline().decode('utf-8')
            if not line:
                break
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            if "DATA" in line:
                break
        if not fields:
            raise ValueError(f"No FIELDS found in file {filename}")
        binary_data = file.read()
        num_floats = len(binary_data) // 4
        data = np.frombuffer(binary_data, dtype=np.float32, count=num_floats)
        if num_floats % len(fields) != 0:
            raise ValueError("Binary data size is not a multiple of the number of fields.")
        points = data.reshape(-1, len(fields))
    return fields, points

def quaternion_to_euler(qx, qy, qz, qw):
    rot = R.from_quat([qx, qy, qz, qw])
    return rot.as_euler('xyz', degrees=False)

def find_closest_image(sorted_image_entries, target_time):
    image_times = [entry[0] for entry in sorted_image_entries]
    pos = bisect.bisect_left(image_times, target_time)
    if pos == 0:
        return sorted_image_entries[0][1]
    if pos == len(image_times):
        return sorted_image_entries[-1][1]
    before = sorted_image_entries[pos - 1]
    after = sorted_image_entries[pos]
    return before[1] if abs(before[0] - target_time) <= abs(after[0] - target_time) else after[1]

def process_timestamp(timestamp_str):
    """
    将时间戳转换为 seconds_microseconds 格式，与 poses 文件命名规则一致
    例如: 1702473515.979433 -> "1702473515_979433"
    """
    timestamp_float = float(timestamp_str)
    seconds = int(timestamp_float)
    microseconds = int((timestamp_float - seconds) * 1e6)
    return f"{seconds}_{microseconds:06d}"

def normalize_sphere(points):
    """
    Shape-centric normalization:
    1) subtract the point-cloud centroid (so shape is centered at 0)
    2) divide by the furthest point’s distance (so shape fits inside unit ball)
    Loses all absolute translation information.
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.linalg.norm(points, axis=1))
    if furthest_distance > 0:
        points = points / furthest_distance
    return points

def normalize_range(points, range=120.0):
    """
    Sensor-centric range normalization:
    1) divide all points by max_range so that the sensor origin remains at 0
       and every coordinate lies in [-1,1].
    Preserves translation and uses the full [–1,1]³ cube for quantization.
    """
    return points / range

def normalize_raw(points):
    return points

def resample_pointcloud(points, target_size=4096):
    if target_size <= 0:  # -1 表示保留全部点
        return points
    current_size = points.shape[0]
    if current_size > target_size:
        indices = np.random.choice(current_size, target_size, replace=False)
    elif current_size < target_size:
        indices = np.random.choice(current_size, target_size, replace=True)
    else:
        return points
    return points[indices, :]

def preprocess_frame(args):
    idx, filename = args
    radar_folder       = GLOBAL_RADAR_FOLDER
    save_pc_folder     = GLOBAL_SAVE_PC
    W                  = GLOBAL_W
    generate_original  = GLOBAL_GEN_ORIG
    generate_images    = GLOBAL_GEN_IMG
    image_folder       = GLOBAL_IMG_FOLDER
    sorted_image_entries = GLOBAL_SORTED_IMAGES
    target_points      = GLOBAL_TARGET_POINTS
    interp_poses       = GLOBAL_INTERP_POSES
    Body_T_L           = GLOBAL_BT_L
    Body_T_R           = GLOBAL_BT_R
    norm_func          = GLOBAL_NORM_FUNC
    maximum_range      = GLOBAL_MAXIMUM_RANGE

    raw_timestamp = filename[:-4]
    processed_timestamp = process_timestamp(raw_timestamp)
    ts_float = float(raw_timestamp)

    fields, pts = load_pcd(join(radar_folder, filename))
    try:
        ix = fields.index('x')
        iy = fields.index('y')
        iz = fields.index('z')
        idop = fields.index('Doppler')
        ipow = fields.index('Power')
    except ValueError:
        return None

    radar_scan = pts[:, [ix, iy, iz, idop, ipow]].astype(np.float32)
    flag, _, proc_scan = estimate_ego_vel(radar_scan, maximum_range=maximum_range)
    if not flag:
        return None

    if W == 1:
        proc_scan[:, :3] = norm_func(proc_scan[:, :3])
        proc_scan = resample_pointcloud(proc_scan, target_size=target_points)
        if np.unique(proc_scan[:, :3], axis=0).shape[0] < 10:
            return None

        np.save(join(save_pc_folder, f"{processed_timestamp}.npy"), proc_scan)
        if generate_original:
            radar_scan[:, :3] = norm_func(radar_scan[:, :3])
            np.save(join(save_pc_folder, f"{processed_timestamp}_org.npy"), radar_scan)

        if generate_images and image_folder and sorted_image_entries:
            try:
                img = find_closest_image(sorted_image_entries, ts_float)
                dst = join(os.path.dirname(save_pc_folder), "images")
                os.makedirs(dst, exist_ok=True)
                ext = img.split('.')[-1]
                shutil.copy(join(image_folder, img),
                            join(dst, f"{processed_timestamp}.{ext}"))
            except:
                pass

        U_T_L_c = interp_poses[idx]
        L_T_R = np.linalg.inv(Body_T_L) @ Body_T_R
        U_T_R_c = U_T_L_c @ L_T_R
        tx, ty, tz = U_T_R_c[:3, 3]
        roll, pitch, yaw = quaternion_to_euler(*R.from_matrix(U_T_R_c[:3, :3]).as_quat())

        return (processed_timestamp, ty, tx, tz, roll, pitch, yaw)
    else:
        return (idx, processed_timestamp, ts_float, radar_scan, proc_scan)

def process_accumulate_window(args):
    window_idxs, center_idx = args
    preproc              = GLOBAL_PREPROC
    interp_poses         = GLOBAL_INTERP_POSES
    Body_T_L             = GLOBAL_BT_L
    Body_T_R             = GLOBAL_BT_R
    save_pc_folder       = GLOBAL_SAVE_PC
    generate_original    = GLOBAL_GEN_ORIG
    generate_images      = GLOBAL_GEN_IMG
    image_folder         = GLOBAL_IMG_FOLDER
    sorted_image_entries = GLOBAL_SORTED_IMAGES
    target_points        = GLOBAL_TARGET_POINTS
    pcd_files            = GLOBAL_PCD_FILES
    norm_func            = GLOBAL_NORM_FUNC

    center_file = pcd_files[center_idx]
    raw_timestamp = center_file[:-4]
    processed_timestamp = process_timestamp(raw_timestamp)
    
    # 检查中心帧是否有效（ego-velocity估计成功）
    if center_idx not in preproc:
        return None

    U_T_L_c = interp_poses[center_idx]
    L_T_R = np.linalg.inv(Body_T_L) @ Body_T_R
    U_T_R_c = U_T_L_c @ L_T_R
    Rc_T_U = np.linalg.inv(U_T_R_c)

    subpc = []
    for j in window_idxs:
        # 跳过ego-velocity估计失败的帧
        if j not in preproc:
            continue
        
        _, _, _, _, proc_scan = preproc[j]
        xyz = proc_scan[:, :3]
        inten = proc_scan[:, 4][:, None]

        U_T_L_j = interp_poses[j]
        U_T_R_j = U_T_L_j @ L_T_R
        Rc_T_Rj = Rc_T_U @ U_T_R_j

        hpts = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
        xyz_c = (Rc_T_Rj @ hpts.T).T[:, :3]
        speed = np.zeros((xyz_c.shape[0], 1), dtype=np.float32)
        subpc.append(np.hstack([xyz_c, speed, inten]))
    
    # 如果窗口内有效帧太少，跳过此窗口
    if len(subpc) == 0:
        return None

    subpc = np.vstack(subpc)
    subpc = resample_pointcloud(subpc, target_size=target_points)
    subpc[:, :3] = norm_func(subpc[:, :3])
    if np.unique(subpc[:, :3], axis=0).shape[0] < 10:
        return None

    np.save(join(save_pc_folder, f"{processed_timestamp}.npy"), subpc)
    if generate_original and center_idx in preproc:
        _, _, _, radar_scan, _ = preproc[center_idx]
        radar_scan[:, :3] = norm_func(radar_scan[:, :3])
        np.save(join(save_pc_folder, f"{processed_timestamp}_org.npy"), radar_scan)

    if generate_images and image_folder and sorted_image_entries:
        try:
            ts_float = float(raw_timestamp)
            img = find_closest_image(sorted_image_entries, ts_float)
            dst = join(os.path.dirname(save_pc_folder), "images")
            os.makedirs(dst, exist_ok=True)
            ext = img.split('.')[-1]
            shutil.copy(join(image_folder, img),
                        join(dst, f"{processed_timestamp}.{ext}"))
        except:
            pass

    tx, ty, tz = U_T_R_c[:3, 3]
    roll, pitch, yaw = quaternion_to_euler(*R.from_matrix(U_T_R_c[:3, :3]).as_quat())
    return (processed_timestamp, ty, tx, tz, roll, pitch, yaw)

def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess radar dataset with optional windowed accumulation"
    )
    parser.add_argument("--dataset_root", type=str, default="/datasets/snail-radar/if/20240116_5",
                        help="Root directory of the dataset")
    parser.add_argument("--radar_rel_path", type=str, default="eagleg7/enhanced",
                        help="Relative path to radar .pcd files")
    parser.add_argument("--lidar_pose_rel_path", type=str, default="utm50r_T_xt32.txt",
                        help="Relative path to utm50r->LiDAR pose file")
    parser.add_argument("--lidar_calib_rel_path",type=str, default="body_T_xt32.txt",
                        help="Relative path to LiDAR calibration (body->LiDAR) file")
    parser.add_argument("--radar_calib_rel_path",type=str, default="body_T_oculii.txt",
                        help="Relative path to Radar calibration (body->Radar) file")
    parser.add_argument("--img_rel_path", type=str, default="zed2i/left",
                        help="Relative path to image camera image files")
    parser.add_argument("--save_folder", type=str, default='datasets/snail/radar/if',
                        help="Folder to save preprocessed data")
    parser.add_argument("--accum_win", type=int, default=7,
                        help="If >1, number of consecutive frames to accumulate")
    parser.add_argument("-o", "--generate_original", action="store_true",
                        help="Generate original point clouds without outlier removal at the same time.")
    parser.add_argument("-i", "--generate_images", action="store_true",
                        help="Copy images from raw data to new folder.")
    parser.add_argument("--target_points", type=int, default=-1,
                        help="The desired number of points in the processed point cloud (for up/down sampling).")
    parser.add_argument(
        "--norm_type",
        type=str,
        default="raw",
        choices=["range", "sphere", "raw"],
        help=(
            "Normalization type for point clouds: "
            "'range'   – range normalization, "
            "'sphere'– unit-sphere scaling, "
            "'raw'   – leave data unchanged"
        )
    )
    parser.add_argument("--maximum_range", type=float, default=120.0,
                        help="Maximum range of pts to be kept.")
    parser.add_argument("--add_suffix", type=str, default="",
                        help="Additional info to be added to the output folder name.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gap_size", type=float, default=4,
                       help="Sample gap size between frames.")
    group.add_argument("--interval_dist", type=float, default=None,
                       help="Sample minimum distance between frames (meters).")

    args = parser.parse_args()
    if args.save_folder is None:
        suffix = "" if args.accum_win == 1 else f"_accm{args.accum_win}"
        add_suffix = f"_{args.add_suffix}" if args.add_suffix else ""
        args.save_folder = f"{args.dataset_root}_preprocessed{suffix}{add_suffix}"
    os.makedirs(args.save_folder, exist_ok=True)
    with open(join(args.save_folder, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    return args

def main():
    args = get_args()
    radar_folder = join(args.dataset_root, args.radar_rel_path)
    save_folder = args.save_folder
    save_pc_folder = join(save_folder, "pointclouds")
    os.makedirs(save_pc_folder, exist_ok=True)

    if args.norm_type == "range":
        norm_func = lambda points: normalize_range(points, range=args.maximum_range)
    elif args.norm_type == "sphere":
        norm_func = normalize_sphere
    else:
        norm_func = normalize_raw

    # load poses
    pose_file = join(args.dataset_root, args.lidar_pose_rel_path)
    times, positions, quats = [], [], []
    with open(pose_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#') or len(parts) < 8:
                continue
            t = float(parts[0]); times.append(t)
            positions.append(list(map(float, parts[1:4])))
            quats.append(list(map(float, parts[4:8])))
    times = np.array(times)
    t_min, t_max = times[0], times[-1]
    positions = np.array(positions)
    quats = np.array(quats)

    # list & filter radar files
    all_files = sorted([f for f in os.listdir(radar_folder) if f.endswith('.pcd')])
    all_times = np.array([float(f[:-4]) for f in all_files])
    valid_mask = (all_times >= t_min) & (all_times <= t_max)
    pcd_files = [all_files[i] for i in np.nonzero(valid_mask)[0]]
    radar_times = all_times[valid_mask]

    interp_poses = interpolation(times, positions, quats, radar_times)
    Body_T_L = np.loadtxt(join(args.dataset_root, args.lidar_calib_rel_path)).reshape(4, 4)
    Body_T_R = np.loadtxt(join(args.dataset_root, args.radar_calib_rel_path)).reshape(4, 4)

    # prepare images
    if args.generate_images:
        image_folder = join(args.dataset_root, args.img_rel_path)
        image_files = [f for f in os.listdir(image_folder)
                       if f.lower().endswith(('.jpg', '.png'))]
        sorted_image_entries = []
        for im in image_files:
            try:
                ts = float(im[:-4]); sorted_image_entries.append((ts, im))
            except:
                pass
        sorted_image_entries.sort(key=lambda x: x[0])
    else:
        image_folder = None
        sorted_image_entries = None

    W = args.accum_win
    gap = args.gap_size
    sample_dist = args.interval_dist

    # Stage 1: per-frame preprocessing
    init1_args = (
        radar_folder, save_pc_folder, W, args.generate_original,
        args.generate_images, image_folder, sorted_image_entries,
        args.target_points, interp_poses, Body_T_L, Body_T_R,
        norm_func, args.maximum_range
    )
    # build tasks list
    tasks1 = []
    last_pos = None
    for idx, fn in enumerate(pcd_files):
        include = False
        if W == 1:
            if gap is not None and idx % int(gap) == 0:
                include = True
            if sample_dist is not None:
                curr_pos = interp_poses[idx][:3, 3]
                if last_pos is None or np.linalg.norm(curr_pos - last_pos) >= sample_dist:
                    include = True
        else:
            include = True
        if include:
            tasks1.append((idx, fn))
            if include and sample_dist is not None:
                last_pos = interp_poses[idx][:3, 3]

    results = []
    with Pool(processes=16,initializer=init_stage1, initargs=init1_args) as pool:
        if W == 1:
            for gps_tuple in tqdm(pool.imap_unordered(preprocess_frame, tasks1),
                                  total=len(tasks1), desc="Processing frames"):
                if gps_tuple:
                    results.append(gps_tuple)
        else:
            preproc_list = list(tqdm(pool.imap_unordered(preprocess_frame, tasks1),
                                     total=len(tasks1), desc="Estimating ego velocity"))
            preproc_list = [r for r in preproc_list if r]

    # Stage 2: window accumulation
    if W > 1:
        N = len(pcd_files)
        centers = []
        if gap is not None:
            for start in range(0, N - W + 1, int(gap)):
                centers.append(start + W // 2)
        else:
            last_pos = None
            for idx in range(N):
                curr_pos = interp_poses[idx][:3, 3]
                if last_pos is None or np.linalg.norm(curr_pos - last_pos) >= sample_dist:
                    if idx - W // 2 >= 0 and idx + W // 2 < N:
                        centers.append(idx)
                    last_pos = curr_pos
        windows = []
        for c in centers:
            start = c - W // 2
            windows.append(list(range(start, start + W)))

        init2_args = (preproc_list, interp_poses, Body_T_L, Body_T_R,
                      W, save_pc_folder, args.generate_original,
                      args.generate_images, image_folder,
                      sorted_image_entries, args.target_points,
                      pcd_files, norm_func)
        with Pool(processes=16,initializer=init_stage2, initargs=init2_args) as pool:
            for gps_tuple in tqdm(pool.imap_unordered(process_accumulate_window,
                                                      zip(windows, centers)),
                                  total=len(windows), desc="Accumulating windows"):
                if gps_tuple:
                    results.append(gps_tuple)

    # write GPS csv
    results.sort(key=lambda x: int(x[0]))
    gps_csv = join(save_folder, "gps.csv")
    with open(gps_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "northing", "easting", "height", "roll", "pitch", "yaw"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
