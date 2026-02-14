#!/usr/bin/env python3
"""
示例: 如何使用 radar_image_converter

演示基本用法和常见场景
"""
import numpy as np
import os
from radar_image_converter import BEVConverter, RangeImageConverter, normalize_image


def example1_single_file():
    """示例1: 转换单个文件"""
    print("=" * 50)
    print("示例1: 转换单个文件")
    print("=" * 50)
    
    # 创建模拟点云数据 (实际使用时从文件加载)
    # 格式: (N, 5) = [x, y, z, doppler, intensity]
    N = 5000
    pc = np.random.randn(N, 5).astype(np.float32)
    pc[:, 0] = pc[:, 0] * 20  # x: -40 ~ 40m
    pc[:, 1] = pc[:, 1] * 20  # y: -40 ~ 40m
    pc[:, 2] = pc[:, 2] * 2   # z: -4 ~ 4m
    pc[:, 3] = pc[:, 3] * 5   # doppler: -10 ~ 10 m/s
    pc[:, 4] = np.abs(pc[:, 4]) * 50  # intensity: 0 ~ 100
    
    print(f"点云形状: {pc.shape}")
    print(f"坐标范围: X[{pc[:,0].min():.1f}, {pc[:,0].max():.1f}], "
          f"Y[{pc[:,1].min():.1f}, {pc[:,1].max():.1f}]")
    
    # 创建BEV转换器
    bev_converter = BEVConverter(
        x_range=(-50, 50),
        y_range=(-50, 50),
        resolution=0.2
    )
    
    # 转换
    bev_img = bev_converter(pc)
    print(f"\nBEV图像形状: {bev_img.shape}")
    
    # 归一化
    bev_norm = normalize_image(bev_img, method='percentile')
    print(f"归一化后范围: [{bev_norm.min():.3f}, {bev_norm.max():.3f}]")
    
    # 保存
    os.makedirs("output_examples", exist_ok=True)
    np.save("output_examples/bev_example.npy", bev_norm)
    print(f"\n✓ 保存到: output_examples/bev_example.npy")
    
    # 可视化
    try:
        import cv2
        vis = (bev_norm[:, :, :3] * 255).astype(np.uint8)
        cv2.imwrite("output_examples/bev_example.png", vis)
        print(f"✓ 可视化: output_examples/bev_example.png")
    except:
        print("(跳过可视化，需要opencv)")


def example2_range_image():
    """示例2: 生成Range图像"""
    print("\n" + "=" * 50)
    print("示例2: 生成Range图像")
    print("=" * 50)
    
    # 模拟点云
    N = 8000
    # 球面分布
    theta = np.random.rand(N) * 2 * np.pi  # 方位角
    phi = (np.random.rand(N) - 0.5) * np.pi / 4  # 俯仰角
    r = np.random.rand(N) * 80 + 5  # 距离 5-85m
    
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    
    doppler = np.random.randn(N) * 3
    intensity = np.random.rand(N) * 100
    
    pc = np.stack([x, y, z, doppler, intensity], axis=1).astype(np.float32)
    
    print(f"点云形状: {pc.shape}")
    
    # Range转换
    range_converter = RangeImageConverter(
        azimuth_bins=512,
        elevation_bins=64,
        max_range=100.0
    )
    
    range_img = range_converter(pc)
    print(f"\nRange图像形状: {range_img.shape}")
    
    # 归一化
    range_norm = normalize_image(range_img, method='percentile')
    
    # 统计
    non_empty = (range_img[:, :, 0] > 0).sum()
    total_pixels = range_img.shape[0] * range_img.shape[1]
    print(f"填充率: {non_empty}/{total_pixels} = {100*non_empty/total_pixels:.1f}%")
    
    # 保存
    os.makedirs("output_examples", exist_ok=True)
    np.save("output_examples/range_example.npy", range_norm)
    print(f"\n✓ 保存到: output_examples/range_example.npy")


def example3_custom_config():
    """示例3: 自定义配置"""
    print("\n" + "=" * 50)
    print("示例3: 不同分辨率对比")
    print("=" * 50)
    
    # 模拟点云
    N = 6000
    pc = np.random.randn(N, 5).astype(np.float32) * 15
    
    configs = [
        {"name": "高分辨率", "resolution": 0.1, "x_range": (-25, 25)},
        {"name": "标准分辨率", "resolution": 0.2, "x_range": (-50, 50)},
        {"name": "低分辨率", "resolution": 0.5, "x_range": (-50, 50)},
    ]
    
    for cfg in configs:
        converter = BEVConverter(
            x_range=cfg['x_range'],
            y_range=cfg['x_range'],
            resolution=cfg['resolution']
        )
        bev = converter(pc)
        print(f"{cfg['name']:8s}: 分辨率={cfg['resolution']}m, "
              f"图像尺寸={bev.shape[0]}x{bev.shape[1]}")


def example4_batch_processing():
    """示例4: 批量处理多个文件"""
    print("\n" + "=" * 50)
    print("示例4: 批量处理")
    print("=" * 50)
    
    # 创建示例数据
    os.makedirs("output_examples/pointclouds", exist_ok=True)
    
    print("生成示例点云文件...")
    for i in range(5):
        N = np.random.randint(3000, 8000)
        pc = np.random.randn(N, 5).astype(np.float32) * 20
        np.save(f"output_examples/pointclouds/scan_{i:03d}.npy", pc)
    
    print("创建了5个示例文件")
    
    # 批量转换
    print("\n使用命令行工具批量转换:")
    print("-" * 50)
    cmd = """python -m radar_image_converter.batch_convert \\
    --input_folder output_examples/pointclouds \\
    --output_folder output_examples/images \\
    --image_types bev \\
    --bev_resolution 0.2 \\
    --norm_method percentile"""
    print(cmd)
    print("-" * 50)
    print("\n(复制上述命令执行)")


def example5_integration():
    """示例5: 在自己项目中集成"""
    print("\n" + "=" * 50)
    print("示例5: 项目集成代码模板")
    print("=" * 50)
    
    code = '''
# 在你的项目中:
from radar_image_converter import BEVConverter, normalize_image
import numpy as np

class MyRadarProcessor:
    def __init__(self):
        self.converter = BEVConverter(
            x_range=(-50, 50),
            y_range=(-50, 50),
            resolution=0.2
        )
    
    def process(self, pointcloud_path):
        """处理单个点云文件"""
        # 加载
        pc = np.load(pointcloud_path)
        
        # 转为BEV
        bev = self.converter(pc)
        
        # 归一化
        bev_norm = normalize_image(bev, method='percentile')
        
        return bev_norm
    
    def batch_process(self, input_folder, output_folder):
        """批量处理"""
        import glob
        import os
        
        files = glob.glob(f"{input_folder}/*.npy")
        os.makedirs(output_folder, exist_ok=True)
        
        for f in files:
            bev = self.process(f)
            out_name = os.path.basename(f)
            np.save(f"{output_folder}/{out_name}", bev)

# 使用
processor = MyRadarProcessor()
result = processor.process("my_pointcloud.npy")
'''
    
    print(code)


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print(" radar_image_converter 使用示例")
    print("=" * 60)
    
    # 运行示例
    example1_single_file()
    example2_range_image()
    example3_custom_config()
    example4_batch_processing()
    example5_integration()
    
    print("\n" + "=" * 60)
    print("✓ 所有示例运行完成!")
    print("=" * 60)
    print("\n查看输出文件:")
    print("  - output_examples/bev_example.npy")
    print("  - output_examples/bev_example.png")
    print("  - output_examples/range_example.npy")
    print("  - output_examples/pointclouds/ (示例数据)")


if __name__ == "__main__":
    main()
