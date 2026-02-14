#!/usr/bin/env python3
"""
最简单的转换脚本 - 单个NPY文件转图像

用法:
    python simple_convert.py input.npy output_bev.npy
    python simple_convert.py input.npy output_bev.npy --range output_range.npy
"""
import argparse
import numpy as np
from converters import BEVConverter, RangeImageConverter
from utils import normalize_image


def main():
    parser = argparse.ArgumentParser(description="将单个NPY点云转换为图像")
    parser.add_argument("input", help="输入NPY文件路径")
    parser.add_argument("output_bev", help="输出BEV图像路径(.npy)")
    parser.add_argument("--range", dest="output_range", help="输出Range图像路径(.npy，可选)")
    parser.add_argument("--resolution", type=float, default=0.2, help="BEV分辨率(默认0.2米)")
    
    args = parser.parse_args()
    
    # 加载点云
    print(f"加载: {args.input}")
    pc = np.load(args.input).astype(np.float32)
    print(f"  形状: {pc.shape}")
    
    if pc.shape[1] != 5:
        print(f"错误: 点云应该是(N,5)形状，当前是{pc.shape}")
        return
    
    # 转换BEV
    print(f"\n生成BEV图像...")
    bev_converter = BEVConverter(resolution=args.resolution)
    bev = bev_converter(pc)
    bev_norm = normalize_image(bev, method='percentile')
    
    np.save(args.output_bev, bev_norm)
    print(f"✓ 保存BEV: {args.output_bev}")
    print(f"  形状: {bev_norm.shape}")
    
    # 转换Range（可选）
    if args.output_range:
        print(f"\n生成Range图像...")
        range_converter = RangeImageConverter()
        range_img = range_converter(pc)
        range_norm = normalize_image(range_img, method='percentile')
        
        np.save(args.output_range, range_norm)
        print(f"✓ 保存Range: {args.output_range}")
        print(f"  形状: {range_norm.shape}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
