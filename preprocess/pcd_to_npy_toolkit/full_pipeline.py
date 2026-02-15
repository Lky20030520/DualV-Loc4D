#!/usr/bin/env python3
"""
完整数据处理流程
1. 批量预处理所有数据集（PCD -> NPY）
2. 为每个数据集生成 BEV 图像
3. 为每个数据集生成 Range 图像
"""
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import config_defaults

# 获取脚本目录
script_dir = Path(__file__).parent.absolute()
preprocess_dir = script_dir.parent.absolute()

from config_defaults import DATA_DICT as data_dict

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def run_batch_preprocess(args):
    """步骤1: 调用 batch_preprocess.py 生成 NPY 文件"""
    logging.info("=" * 60)
    logging.info("STEP 1: Running batch_preprocess.py to generate NPY files")
    logging.info("=" * 60)
    
    batch_script = script_dir / "batch_preprocess.py"
    
    # 构建命令
    cmd = [
        sys.executable,
        str(batch_script),
        "--base_dir", args.base_dir,
        "--save_folder", args.save_folder,
        "--accum_win", str(args.accum_win),
        "--target_points", str(args.target_points),
        "--norm_type", args.norm_type,
        "--maximum_range", str(args.maximum_range),
    ]
    
    if args.add_suffix:
        cmd += ["--add_suffix", args.add_suffix]
    
    if args.generate_original:
        cmd.append("-o")
    
    if args.generate_images:
        cmd.append("-i")
    
    if args.gap_size is not None:
        cmd += ["--gap_size", str(args.gap_size)]
    
    if args.interval_dist is not None:
        cmd += ["--interval_dist", str(args.interval_dist)]
    
    # 运行批处理脚本
    if not args.skip_preprocess:
        try:
            logging.info(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            logging.info("✅ Batch preprocessing completed successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Batch preprocessing failed with exit code {e.returncode}")
            return False
        except KeyboardInterrupt:
            logging.warning("⚠️ Batch preprocessing interrupted by user")
            return False
    else:
        logging.info("⏭️  Skipping preprocessing (--skip_preprocess enabled)")
    
    return True

def generate_images_for_dataset(npy_folder, place, dataset_name, args):
    """为单个数据集生成 BEV 和 Range 图像"""
    npy_path = Path(npy_folder) / "pointclouds"
    
    if not npy_path.exists():
        logging.warning(f"⚠️ Pointcloud folder not found: {npy_path}")
        return
    
    logging.info(f"📁 Processing: {place}/{dataset_name}")
    
    # 生成 BEV 图像
    if not args.skip_bev:
        bev_save_path = npy_path.parent / "bev_image"
        bev_save_path.mkdir(exist_ok=True)
        
        bev_script = preprocess_dir / "npy2bev.py"
        bev_cmd = [
            sys.executable,
            str(bev_script),
            "--npy_path", str(npy_path),
            "--bev_save_path", str(bev_save_path)
        ]
        
        logging.info(f"  🗺️  Generating BEV images -> {bev_save_path}")
        try:
            subprocess.run(bev_cmd, check=True, capture_output=True, text=True)
            logging.info(f"  ✅ BEV images generated")
        except subprocess.CalledProcessError as e:
            logging.error(f"  ❌ BEV generation failed: {e.stderr}")
    
    # 生成 Range 图像
    if not args.skip_range:
        range_save_path = npy_path.parent / "range_image"
        range_save_path.mkdir(exist_ok=True)
        
        # npy2range.py 需要修改环境变量或脚本内容
        # 这里我们创建一个临时的调用脚本
        range_script = preprocess_dir / "npy2range.py"
        
        # 读取 npy2range.py 并动态修改路径（更稳健地替换变量定义行）
        range_script_content = range_script.read_text()

        # 创建临时脚本
        temp_range_script = script_dir / f"_temp_npy2range_{place}_{dataset_name.replace('/', '_')}.py"

        # 按行替换 INPUT_DIR / OUTPUT_DIR 定义（避免依赖硬编码的原始路径）
        new_lines = []
        replaced_input = False
        replaced_output = False
        for line in range_script_content.splitlines():
            stripped = line.strip()
            if stripped.startswith('INPUT_DIR') and '=' in line:
                new_lines.append(f"INPUT_DIR = r'{npy_path}'")
                replaced_input = True
                continue
            if stripped.startswith('OUTPUT_DIR') and '=' in line:
                new_lines.append(f"OUTPUT_DIR = r'{range_save_path}'")
                replaced_output = True
                continue
            new_lines.append(line)

        # 如果源脚本没有明确的 OUTPUT_DIR 行，则在文件顶部插入定义（保证存在）
        if not replaced_input or not replaced_output:
            header = []
            if not replaced_input:
                header.append(f"INPUT_DIR = r'{npy_path}'")
            if not replaced_output:
                header.append(f"OUTPUT_DIR = r'{range_save_path}'")
            modified_content = '\n'.join(header + [''] + new_lines)
        else:
            modified_content = '\n'.join(new_lines)

        temp_range_script.write_text(modified_content)
        
        range_cmd = [sys.executable, str(temp_range_script)]
        
        logging.info(f"  📊 Generating Range images -> {range_save_path}")
        try:
            subprocess.run(range_cmd, check=True, capture_output=True, text=True)
            logging.info(f"  ✅ Range images generated")
        except subprocess.CalledProcessError as e:
            logging.error(f"  ❌ Range generation failed: {e.stderr}")
        finally:
            # 清理临时文件
            if temp_range_script.exists():
                temp_range_script.unlink()

def run_image_generation(args):
    """步骤2&3: 为所有数据集生成图像"""
    logging.info("=" * 60)
    logging.info("STEP 2&3: Generating BEV and Range images")
    logging.info("=" * 60)
    
    # 计算输出文件夹名称后缀
    suffix = "" if args.accum_win == 1 else f"_accm{args.accum_win}"
    add_suffix = f"_{args.add_suffix}" if args.add_suffix else ""
    
    total_datasets = sum(len(folders) for folders in data_dict.values())
    processed = 0
    
    for place, folder_list in data_dict.items():
        for folder in folder_list:
            processed += 1
            dataset_name = folder.replace("/", "_")
            
            # 构建预处理后的文件夹路径
            npy_folder = Path(args.save_folder) / place / f"{dataset_name}_preprocessed{suffix}{add_suffix}"
            
            logging.info(f"[{processed}/{total_datasets}] {place}/{dataset_name}")
            
            if not npy_folder.exists():
                logging.warning(f"  ⚠️ Folder not found (skipping): {npy_folder}")
                continue
            
            generate_images_for_dataset(npy_folder, place, dataset_name, args)
    
    logging.info("=" * 60)
    logging.info("✅ All image generation completed!")
    logging.info("=" * 60)

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Full pipeline: Preprocess all datasets and generate BEV/Range images"
    )
    
    # 数据路径参数
    parser.add_argument("--base_dir", type=str, default=config_defaults.BASE_DIR,
                        help="Base directory containing raw dataset folders")
    parser.add_argument("--save_folder", type=str, default=config_defaults.SAVE_FOLDER,
                        help="Base folder to save preprocessed data")
    
    # 预处理参数（与 batch_preprocess.py 一致）
    parser.add_argument("--accum_win", type=int, default=config_defaults.ACCUM_WIN,
                        help="Number of consecutive frames to accumulate")
    parser.add_argument("--target_points", type=int, default=config_defaults.TARGET_POINTS,
                        help="Target number of points (-1 = keep all)")
    parser.add_argument("--norm_type", type=str, default=config_defaults.NORM_TYPE,
                        choices=["range", "sphere", "raw"],
                        help="Normalization type for point clouds")
    parser.add_argument("--maximum_range", type=float, default=config_defaults.MAXIMUM_RANGE,
                        help="Maximum range of points to keep (meters)")
    parser.add_argument("--add_suffix", type=str, default=config_defaults.ADD_SUFFIX,
                        help="Additional suffix for output folder name")
    parser.add_argument("-o", "--generate_original", action="store_true",
                        help="Generate original point clouds without outlier removal")
    parser.add_argument("-i", "--generate_images", action="store_true",
                        help="Copy images from raw data to new folder")
    
    # 采样参数（互斥）
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gap_size", type=float, default=config_defaults.GAP_SIZE,
                       help="Sample gap size between frames")
    group.add_argument("--interval_dist", type=float, default=config_defaults.INTERVAL_DIST,
                       help="Sample minimum distance between frames (meters)")
    
    # 流程控制参数
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip batch preprocessing (only generate images)")
    parser.add_argument("--skip_bev", action="store_true",
                        help="Skip BEV image generation")
    parser.add_argument("--skip_range", action="store_true",
                        help="Skip Range image generation")
    
    args = parser.parse_args()
    
    # 步骤1: 批量预处理
    if not run_batch_preprocess(args):
        logging.error("Pipeline failed at preprocessing stage")
        return 1
    
    # 步骤2&3: 生成图像
    run_image_generation(args)
    
    logging.info("🎉 Full pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
