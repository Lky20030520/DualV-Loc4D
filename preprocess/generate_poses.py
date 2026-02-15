#!/usr/bin/env python3
"""Generate poses folders from times_kitti.txt and utm50r_T_xt32_kitti.txt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

TIMES_NAME = "times_kitti.txt"
POSES_SRC_NAME = "utm50r_T_xt32_kitti.txt"
POSES_DIR_NAME = "poses"


def build_pose_filename(time_str: str) -> str:
    """Convert a timestamp line to a pose filename with no extension."""
    clean = time_str.strip()
    if not clean:
        return ""
    return f"{clean.replace('.', '_')}_txt"


def generate_poses_in_dir(dataset_dir: Path, output_dir: Path, overwrite: bool) -> int:
    """Convert a timestamp line to individual pose files.
    
    Args:
        dataset_dir: Source directory to read times and poses from
        output_dir: Output directory to save pose files to
        overwrite: Whether to overwrite existing files
    """
    times_path = dataset_dir / TIMES_NAME
    poses_src_path = dataset_dir / POSES_SRC_NAME
    if not (times_path.is_file() and poses_src_path.is_file()):
        return 0

    try:
        times = times_path.read_text(encoding="utf-8").splitlines()
        poses = poses_src_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"⚠️ 读取失败 {dataset_dir}: {e}")
        return 0

    times = [t.strip() for t in times if t.strip()]
    poses = [p.strip() for p in poses if p.strip()]

    if len(times) != len(poses):
        print(
            f"⚠️ 数量不匹配 {dataset_dir}: {len(times)} 时间戳 vs {len(poses)} 位姿"
        )
        return 0

    # 保存到输出目录而非源目录
    poses_dir = output_dir / POSES_DIR_NAME
    poses_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for time_str, pose_line in zip(times, poses):
        filename = build_pose_filename(time_str)
        if not filename:
            continue
        out_path = poses_dir / filename
        if out_path.exists() and not overwrite:
            continue
        try:
            out_path.write_text(pose_line + "\n", encoding="utf-8")
            written += 1
        except Exception as e:
            print(f"⚠️ 写入失败 {out_path}: {e}")
            continue

    return written


def find_dataset_dirs(root: Path) -> list[Path]:
    """
    优化版：直接搜索已知的目录结构，避免递归遍历
    SNAIL数据集结构: /datasets/snail-radar/{place}/{date_run}/
    """
    dataset_dirs: list[Path] = []
    
    # 限制搜索深度到3层以内
    if not root.exists():
        return dataset_dirs
    
    print(f"🔍 搜索数据集目录: {root}")
    count = 0
    
    # 层级1: place (bc, if, 81r, etc.)
    try:
        for place_dir in root.iterdir():
            if not place_dir.is_dir():
                continue
            
            # 层级2: date_run (20231213/4, etc.)
            try:
                for date_dir in place_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    # 层级3: run number (1, 2, 3, etc.) 或直接包含文件
                    # 先检查当前目录
                    if (date_dir / TIMES_NAME).exists() and (date_dir / POSES_SRC_NAME).exists():
                        dataset_dirs.append(date_dir)
                        count += 1
                        if count % 10 == 0:
                            print(f"  已找到 {count} 个数据集...")
                    
                    # 再检查子目录
                    try:
                        for run_dir in date_dir.iterdir():
                            if not run_dir.is_dir():
                                continue
                            if (run_dir / TIMES_NAME).exists() and (run_dir / POSES_SRC_NAME).exists():
                                dataset_dirs.append(run_dir)
                                count += 1
                                if count % 10 == 0:
                                    print(f"  已找到 {count} 个数据集...")
                    except PermissionError:
                        continue
                        
            except PermissionError:
                continue
                
    except PermissionError:
        print(f"⚠️ 权限不足，无法访问: {root}")
    
    print(f"✅ 总共找到 {len(dataset_dirs)} 个数据集")
    return dataset_dirs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate poses/ files from times_kitti and utm50r_T_xt32_kitti."
    )
    parser.add_argument(
        "--root",
        default="/datasets/snail-radar",
        help="Source root path to read datasets from (default: /datasets/snail-radar)",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output root path to save poses to. If not specified, saves to --root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pose files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")
    
    # 如果未指定输出目录，则直接保存到源目录
    output_root = Path(args.output_root) if args.output_root else root
    if args.output_root:
        print(f"📂 源目录: {root}")
        print(f"💾 输出目录: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(root)
    if not dataset_dirs:
        print("No datasets found with times_kitti.txt and utm50r_T_xt32_kitti.txt")
        return 0

    total = 0
    for i, dataset_dir in enumerate(dataset_dirs, 1):
        # 计算相对路径并映射到输出目录
        try:
            rel_path = dataset_dir.relative_to(root)
            output_dir = output_root / rel_path
        except ValueError:
            # dataset_dir 不在 root 下，跳过
            continue
        
        print(f"[{i}/{len(dataset_dirs)}] 处理: {rel_path}")
        count = generate_poses_in_dir(dataset_dir, output_dir, args.overwrite)
        if count:
            print(f"  ✅ 写入 {count} 个位姿文件 -> {output_dir / POSES_DIR_NAME}")
            total += count
        else:
            print(f"  ⏭️  跳过")

    print(f"\n🎉 完成! 总共写入 {total} 个位姿文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())