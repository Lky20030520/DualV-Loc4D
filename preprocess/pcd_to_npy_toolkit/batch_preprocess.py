# #!/usr/bin/env python3
# import os
# import sys
# import subprocess
# import argparse
# import logging
# import config_defaults


# script_dir = os.path.dirname(os.path.abspath(__file__))
# preprocess_script = os.path.join(script_dir, "pcs_preprocess.py")

# if not os.path.isfile(preprocess_script):
#     print(f"pcs_preprocess.py not found in the current directory: {script_dir}")
#     sys.exit(1)

# from config_defaults import DATA_DICT as data_dict

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )

# def main():
#     setup_logging()
    
#     # Parse command line arguments for the base directory
#     parser = argparse.ArgumentParser(
#         description="Process dataset folders with pcs_preprocess.py using a specified base directory."
#     )
#     parser.add_argument(
#         "--base_dir",
#         type=str,
#         default=config_defaults.BASE_DIR,
#         help="Base directory containing the dataset folders."
#     )

#     parser.add_argument("--radar_rel_path", type=str, default=config_defaults.RADAR_REL_PATH,
#                         help="Relative path to radar .pcd files")
#     parser.add_argument("--lidar_pose_rel_path", type=str, default=config_defaults.LIDAR_POSE_REL_PATH,
#                         help="Relative path to utm50r->LiDAR pose file")
#     parser.add_argument("--lidar_calib_rel_path", type=str, default=config_defaults.LIDAR_CALIB_REL_PATH,
#                         help="Relative path to LiDAR calibration (body->LiDAR) file")
#     parser.add_argument("--radar_calib_rel_path", type=str, default=config_defaults.RADAR_CALIB_REL_PATH,
#                         help="Relative path to Radar calibration (body->Radar) file")
#     parser.add_argument("--img_rel_path", type=str, default=config_defaults.IMG_REL_PATH,
#                         help="Relative path to image camera image files")
#     parser.add_argument("--save_folder", type=str, default=config_defaults.SAVE_FOLDER,
#                         help="Base folder to save preprocessed data (will append place name)")
#     parser.add_argument("--accum_win", type=int, default=config_defaults.ACCUM_WIN,
#                         help="If >1, number of consecutive frames to accumulate")
#     parser.add_argument("-o", "--generate_original", action="store_true",
#                         help="Generate original point clouds without outlier removal at the same time.")
#     parser.add_argument("-i", "--generate_images", action="store_true",
#                         help="Copy images from raw data to new folder.")
#     parser.add_argument("--target_points", type=int, default=config_defaults.TARGET_POINTS,
#                         help="The desired number of points in the processed point cloud (for up/down sampling).")
#     parser.add_argument(
#         "--norm_type",
#         type=str,
#         default=config_defaults.NORM_TYPE,
#         choices=["range", "sphere", "raw"],
#         help=(
#             "Normalization type for point clouds: "
#             "'range'   – range normalization, "
#             "'sphere'– unit-sphere scaling, "
#             "'raw'   – leave data unchanged"
#         )
#     )
#     parser.add_argument("--maximum_range", type=float, default=config_defaults.MAXIMUM_RANGE,
#                         help="Maximum range of pts to be kept.")
#     parser.add_argument("--add_suffix", type=str, default=config_defaults.ADD_SUFFIX,
#                         help="Additional info to be added to the output folder name.")
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument("--gap_size", type=float, default=config_defaults.GAP_SIZE,
#                        help="Sample gap size between frames.")
#     group.add_argument("--interval_dist", type=float, default=config_defaults.INTERVAL_DIST,
#                        help="Sample minimum distance between frames (meters).")
        
#     args = parser.parse_args()
#     base_dir = args.base_dir

#     # Verify that the base directory exists
#     if not os.path.isdir(base_dir):
#         logging.error(f"The base directory does not exist: {base_dir}")
#         sys.exit(1)
    

#     for place, folder_list in data_dict.items():
#         for folder in folder_list:
#             processed_folder = folder.replace("/", "_")
#             datasets_root = os.path.join(base_dir, place, processed_folder)
#             logging.info(f"Processing folder: {datasets_root}")

#             if not os.path.isdir(datasets_root):
#                 logging.warning(f"Directory does not exist: {datasets_root}. Creating directory.")
#                 try:
#                     os.makedirs(datasets_root, exist_ok=True)
#                 except Exception as e:
#                     logging.error(f"Failed to create directory {datasets_root}: {e}")
#                     continue

#             # 构建完整的保存路径（包含数据集名称）
#             suffix = "" if args.accum_win == 1 else f"_accm{args.accum_win}"
#             add_suffix = f"_{args.add_suffix}" if args.add_suffix else ""
#             save_folder_full = os.path.join(args.save_folder, place, f"{processed_folder}_preprocessed{suffix}{add_suffix}")

#             command = [
#                 sys.executable, preprocess_script, 
#                 "--dataset_root", datasets_root,
#                 "--radar_rel_path", args.radar_rel_path,
#                 "--lidar_pose_rel_path", args.lidar_pose_rel_path,
#                 "--lidar_calib_rel_path", args.lidar_calib_rel_path,
#                 "--radar_calib_rel_path", args.radar_calib_rel_path,
#                 "--img_rel_path", args.img_rel_path,
#                 "--save_folder", save_folder_full,
#                 "--accum_win", str(args.accum_win),
#                 "--target_points", str(args.target_points),
#                 "--norm_type", args.norm_type,
#                 "--maximum_range", str(args.maximum_range),
#                 "--add_suffix", args.add_suffix
#             ]
            
#             logging.info(f"  → Output will be saved to: {save_folder_full}")
#             if args.generate_original:
#                 command.append("-o")
#             if args.generate_images:
#                 command.append("-i")
#             if args.gap_size is not None:
#                 command += ["--gap_size", str(args.gap_size)]
#             if args.interval_dist is not None:
#                 command += ["--interval_dist", str(args.interval_dist)]
            
#             try:
#                 result = subprocess.run(
#                     command,
#                     check=True,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True
#                 )
#                 logging.info(f"Success processing {datasets_root}:\n{result.stdout}")
#             except subprocess.CalledProcessError as e:
#                 logging.error(f"Error processing {datasets_root} (exit code {e.returncode}):\n{e.stderr}")
#             except Exception as e:
#                 logging.error(f"Unexpected error processing {datasets_root}: {e}")

#     logging.info("All processing tasks have been completed.")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import logging
import config_defaults

script_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_script = os.path.join(script_dir, "pcs_preprocess.py")

if not os.path.isfile(preprocess_script):
    print(f"pcs_preprocess.py not found in the current directory: {script_dir}")
    sys.exit(1)

from config_defaults import DATA_DICT as data_dict

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def main():
    setup_logging()
    
    # Parse command line arguments for the base directory
    parser = argparse.ArgumentParser(
        description="Process dataset folders with pcs_preprocess.py using a specified base directory."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=config_defaults.BASE_DIR,
        help="Base directory containing the dataset folders."
    )

    parser.add_argument("--radar_rel_path", type=str, default=config_defaults.RADAR_REL_PATH,
                        help="Relative path to radar .pcd files")
    parser.add_argument("--lidar_pose_rel_path", type=str, default=config_defaults.LIDAR_POSE_REL_PATH,
                        help="Relative path to utm50r->LiDAR pose file")
    parser.add_argument("--lidar_calib_rel_path", type=str, default=config_defaults.LIDAR_CALIB_REL_PATH,
                        help="Relative path to LiDAR calibration (body->LiDAR) file")
    parser.add_argument("--radar_calib_rel_path", type=str, default=config_defaults.RADAR_CALIB_REL_PATH,
                        help="Relative path to Radar calibration (body->Radar) file")
    parser.add_argument("--img_rel_path", type=str, default=config_defaults.IMG_REL_PATH,
                        help="Relative path to image camera image files")
    parser.add_argument("--save_folder", type=str, default=config_defaults.SAVE_FOLDER,
                        help="Base folder to save preprocessed data")
    parser.add_argument("--accum_win", type=int, default=config_defaults.ACCUM_WIN,
                        help="If >1, number of consecutive frames to accumulate")
    parser.add_argument("-o", "--generate_original", action="store_true",
                        help="Generate original point clouds without outlier removal at the same time.")
    parser.add_argument("-i", "--generate_images", action="store_true",
                        help="Copy images from raw data to new folder.")
    parser.add_argument("--target_points", type=int, default=config_defaults.TARGET_POINTS,
                        help="The desired number of points in the processed point cloud (for up/down sampling).")
    parser.add_argument(
        "--norm_type",
        type=str,
        default=config_defaults.NORM_TYPE,
        choices=["range", "sphere", "raw"],
        help=(
            "Normalization type for point clouds: "
            "'range'   – range normalization, "
            "'sphere'– unit-sphere scaling, "
            "'raw'   – leave data unchanged"
        )
    )
    parser.add_argument("--maximum_range", type=float, default=config_defaults.MAXIMUM_RANGE,
                        help="Maximum range of pts to be kept.")
    parser.add_argument("--add_suffix", type=str, default=config_defaults.ADD_SUFFIX,
                        help="Additional info to be added to the output folder name.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gap_size", type=float, default=config_defaults.GAP_SIZE,
                       help="Sample gap size between frames.")
    group.add_argument("--interval_dist", type=float, default=config_defaults.INTERVAL_DIST,
                       help="Sample minimum distance between frames (meters).")
        
    args = parser.parse_args()
    base_dir = args.base_dir

    # Verify that the base directory exists
    if not os.path.isdir(base_dir):
        logging.error(f"The base directory does not exist: {base_dir}")
        sys.exit(1)
    
    for place, folder_list in data_dict.items():
        for folder in folder_list:
            processed_folder = folder.replace("/", "_")
            
            # 【修改 1：拍扁输入路径】
            # 将 base_dir/if/20240115_3 变成 base_dir/if_20240115_3
            dataset_name = f"{place}_{processed_folder}"
            datasets_root = os.path.join(base_dir, dataset_name)
            
            logging.info(f"Processing folder: {datasets_root}")

            # 如果原始数据文件夹不存在，应该跳过而不是创建空文件夹
            if not os.path.isdir(datasets_root):
                logging.warning(f"Raw data directory does not exist: {datasets_root}. Skipping...")
                continue

            # 【修改 2：拍扁并自定义输出路径】
            # 直接生成如：/workspace/.../if_20240115_3_accum_7
            save_folder_full = os.path.join(args.save_folder, f"{dataset_name}_accum_{args.accum_win}")

            command = [
                sys.executable, preprocess_script, 
                "--dataset_root", datasets_root,
                "--radar_rel_path", args.radar_rel_path,
                "--lidar_pose_rel_path", args.lidar_pose_rel_path,
                "--lidar_calib_rel_path", args.lidar_calib_rel_path,
                "--radar_calib_rel_path", args.radar_calib_rel_path,
                "--img_rel_path", args.img_rel_path,
                "--save_folder", save_folder_full,
                "--accum_win", str(args.accum_win),
                "--target_points", str(args.target_points),
                "--norm_type", args.norm_type,
                "--maximum_range", str(args.maximum_range),
            ]
            
            # 处理可选后缀
            if args.add_suffix:
                command.extend(["--add_suffix", args.add_suffix])
            
            logging.info(f"  → Output will be saved to: {save_folder_full}")
            if args.generate_original:
                command.append("-o")
            if args.generate_images:
                command.append("-i")
            if args.gap_size is not None:
                command += ["--gap_size", str(args.gap_size)]
            if args.interval_dist is not None:
                command += ["--interval_dist", str(args.interval_dist)]
            
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    # stdout=subprocess.PIPE,
                    # stderr=subprocess.PIPE,
                    # text=True
                )
                logging.info(f"Success processing {datasets_root}:\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error processing {datasets_root} (exit code {e.returncode}):\n{e.stderr}")
            except Exception as e:
                logging.error(f"Unexpected error processing {datasets_root}: {e}")

    logging.info("All processing tasks have been completed.")

if __name__ == "__main__":
    main()