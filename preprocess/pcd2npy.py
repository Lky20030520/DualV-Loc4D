import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

def extract_rich_radar_data(src_folder, dst_folder):
    # 检查输入目录是否存在
    if not os.path.exists(src_folder):
        print(f"错误：输入目录不存在 → {src_folder}")
        return
    
    # 创建输出目录（若不存在）
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"创建输出目录 → {dst_folder}")

    # 筛选 .pcd 文件
    files = [f for f in os.listdir(src_folder) if f.endswith('.pcd') and not f.startswith('.')]
    if len(files) == 0:
        print(f"警告：输入目录中未找到 .pcd 文件 → {src_folder}")
        return
    
    print(f"正在处理 {len(files)} 个 PCD 文件 (使用 Open3D Tensor API)...")
    success_count = 0 

    for file_name in tqdm(files, desc="处理进度"):
        pcd_path = os.path.join(src_folder, file_name)
        npy_path = os.path.join(dst_folder, file_name.replace('.pcd', '.npy'))

        try:
            # --- 关键修改 1: 使用 Tensor API 读取 (o3d.t.io) ---
            # 只有这个 API 能保留 Power 和 Doppler 等自定义字段
            pcd = o3d.t.io.read_point_cloud(pcd_path)
            
            # 2. 验证点云是否为空 (Tensor API 使用 is_empty())
            if pcd.is_empty():
                print(f"\n警告：文件 {file_name} 为空，跳过")
                continue

            # --- 关键修改 2: 获取坐标 (positions) ---
            # Tensor API 中坐标通常存储在 "positions" 键下，需转为 numpy
            if "positions" not in pcd.point:
                print(f"\n警告：文件 {file_name} 缺少坐标数据，跳过")
                continue
                
            xyz = pcd.point["positions"].numpy().astype(np.float32)
            n_points = len(xyz)

            # --- 关键修改 3: 检查字段并转换 ---
            # 打印可用的键值以便调试 (如果字段名大小写不匹配)
            # print(f"Available fields: {pcd.point.keys()}") 

            if "Power" not in pcd.point:
                # 尝试容错：有时候可能是小写 "power"
                if "power" in pcd.point:
                    pcd.point["Power"] = pcd.point["power"]
                else:
                    print(f"\n警告：文件 {file_name} 缺少 Power 字段。现有字段: {list(pcd.point.keys())}")
                    continue
            
            if "Doppler" not in pcd.point:
                if "doppler" in pcd.point:
                    pcd.point["Doppler"] = pcd.point["doppler"]
                else:
                    print(f"\n警告：文件 {file_name} 缺少 Doppler 字段。现有字段: {list(pcd.point.keys())}")
                    continue

            # 提取并 reshape
            power = pcd.point["Power"].numpy().astype(np.float32).reshape(-1, 1)
            doppler = pcd.point["Doppler"].numpy().astype(np.float32).reshape(-1, 1)

            # 5. 验证长度
            if len(power) != n_points or len(doppler) != n_points:
                print(f"\n警告：文件 {file_name} 字段长度不匹配")
                continue

            # 6. 组合数据 [x, y, z, Power, Doppler]
            point_cloud = np.hstack([xyz, power, doppler])

            # 7. 过滤无效点
            # 检查是否有 NaN
            valid_mask = ~np.isnan(point_cloud).any(axis=1)
            # 检查是否有 Inf
            valid_mask &= ~np.isinf(point_cloud).any(axis=1)
            
            point_cloud = point_cloud[valid_mask]

            if len(point_cloud) == 0:
                print(f"\n警告：文件 {file_name} 过滤后无有效数据")
                continue

            # 8. 保存
            np.save(npy_path, point_cloud)
            success_count += 1

        except Exception as e:
            print(f"\n错误：处理 {file_name} 失败 → {str(e)}")
            import traceback
            traceback.print_exc() # 打印详细报错堆栈帮助调试
            continue

    print(f"\n转换完成！")
    print(f"成功处理文件数：{success_count}/{len(files)}")
    print(f"输出目录：{dst_folder}")

# --- 配置路径 ---
input_dir = r'/home/kaiyan/BEVPlace3/datasets/snail/radar/if_20240116_5/pcd' 
OUTPUT_DIR = os.path.join(os.path.dirname(input_dir), "enhanced_rich_npy")
output_dir = OUTPUT_DIR

if __name__ == "__main__":
    extract_rich_radar_data(input_dir, output_dir)