#!/usr/bin/env python3
"""
Batch conversion script for converting NPY point clouds to images
"""
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from .converters import BEVConverter, RangeImageConverter
from .utils import normalize_image, save_visualization


def process_single_file(args):
    """Process a single NPY file"""
    npy_path, output_bev_dir, output_range_dir, bev_cfg, range_cfg, norm_method, save_vis = args
    
    try:
        # Load point cloud
        pc = np.load(npy_path).astype(np.float32)
        
        if pc.shape[1] != 5:
            print(f"Warning: {npy_path} has shape {pc.shape}, expected (N, 5). Skipping.")
            return None
        
        filename = Path(npy_path).stem
        
        # Generate BEV image
        if output_bev_dir:
            bev_converter = BEVConverter(**bev_cfg)
            bev_img = bev_converter(pc)
            bev_img_norm = normalize_image(bev_img, method=norm_method)
            
            # Save as NPY
            np.save(os.path.join(output_bev_dir, f"{filename}.npy"), bev_img_norm.astype(np.float32))
            
            # Save visualization
            if save_vis:
                save_visualization(bev_img_norm, os.path.join(output_bev_dir, f"{filename}.png"))
        
        # Generate Range image
        if output_range_dir:
            range_converter = RangeImageConverter(**range_cfg)
            range_img = range_converter(pc)
            range_img_norm = normalize_image(range_img, method=norm_method)
            
            # Save as NPY
            np.save(os.path.join(output_range_dir, f"{filename}.npy"), range_img_norm.astype(np.float32))
            
            # Save visualization
            if save_vis:
                save_visualization(range_img_norm, os.path.join(output_range_dir, f"{filename}.png"))
        
        return filename
    
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch convert NPY point clouds to BEV/Range images")
    
    # Input/Output
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing NPY files")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Output folder (default: input_folder/../images)")
    parser.add_argument("--image_types", type=str, nargs='+', default=['bev', 'range'],
                        choices=['bev', 'range'],
                        help="Types of images to generate")
    
    # BEV parameters
    parser.add_argument("--bev_x_range", type=float, nargs=2, default=[-50, 50])
    parser.add_argument("--bev_y_range", type=float, nargs=2, default=[-50, 50])
    parser.add_argument("--bev_resolution", type=float, default=0.2)
    parser.add_argument("--bev_height_range", type=float, nargs=2, default=[-2, 5])
    
    # Range image parameters
    parser.add_argument("--range_azimuth_bins", type=int, default=512)
    parser.add_argument("--range_elevation_bins", type=int, default=64)
    parser.add_argument("--range_max_range", type=float, default=100.0)
    
    # Processing
    parser.add_argument("--norm_method", type=str, default="percentile",
                        choices=["minmax", "percentile", "standardize"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_visualization", action="store_true",
                        help="Don't save PNG visualizations")
    
    args = parser.parse_args()
    
    # Setup output folders
    if args.output_folder is None:
        args.output_folder = os.path.join(os.path.dirname(args.input_folder), "images")
    
    output_bev_dir = os.path.join(args.output_folder, "bev") if 'bev' in args.image_types else None
    output_range_dir = os.path.join(args.output_folder, "range") if 'range' in args.image_types else None
    
    if output_bev_dir:
        os.makedirs(output_bev_dir, exist_ok=True)
    if output_range_dir:
        os.makedirs(output_range_dir, exist_ok=True)
    
    # Get NPY files
    npy_files = sorted([
        os.path.join(args.input_folder, f) 
        for f in os.listdir(args.input_folder) 
        if f.endswith('.npy')
    ])
    
    if len(npy_files) == 0:
        print(f"No NPY files found in {args.input_folder}")
        return
    
    print(f"Found {len(npy_files)} NPY files")
    print(f"Output folder: {args.output_folder}")
    
    # Configs
    bev_cfg = {
        'x_range': tuple(args.bev_x_range),
        'y_range': tuple(args.bev_y_range),
        'resolution': args.bev_resolution,
        'height_range': tuple(args.bev_height_range)
    }
    
    range_cfg = {
        'azimuth_bins': args.range_azimuth_bins,
        'elevation_bins': args.range_elevation_bins,
        'max_range': args.range_max_range
    }
    
    # Prepare tasks
    tasks = [
        (npy_path, output_bev_dir, output_range_dir, bev_cfg, range_cfg, 
         args.norm_method, not args.no_visualization)
        for npy_path in npy_files
    ]
    
    # Process
    num_workers = args.num_workers or cpu_count()
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, tasks),
            total=len(tasks),
            desc="Converting"
        ))
    
    # Summary
    success_count = sum(1 for r in results if r is not None)
    print(f"\n✓ Successfully processed {success_count}/{len(npy_files)} files")
    
    if output_bev_dir:
        print(f"  BEV images: {output_bev_dir}")
    if output_range_dir:
        print(f"  Range images: {output_range_dir}")


if __name__ == "__main__":
    main()
