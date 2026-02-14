import numpy as np
import os

def analyze_npy(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    # 加载数据
    data = np.load(file_path)

    print("-" * 50)
    print(f"📁 文件名: {os.path.basename(file_path)}")
    print(f"📊 形状 (Shape): {data.shape}  ->  ({data.shape[0]} 个点)")
    print(f"🔢 数据类型 (dtype): {data.dtype}")
    print("-" * 50)

    # 假设格式是 [x, y, z, power, doppler] 或 [x, y, z, intensity, speed]
    columns = ['X', 'Y', 'Z', 'Power/Inten', 'Doppler/Speed']
    
    print("📈 各维度统计信息:")
    for i in range(min(data.shape[1], len(columns))):
        col_data = data[:, i]
        print(f"  {columns[i]:<15}: Min={col_data.min():>8.4f}, Max={col_data.max():>8.4f}, Mean={col_data.mean():>8.4f}")

    print("-" * 50)
    print("📋 前 5 行数据示例:")
    print(data[:5, :])
    print("-" * 50)

if __name__ == "__main__":
    # 替换为你实际生成的 npy 路径
    test_file = r'/home/kaiyan/BEVPlace3/datasets/snail/radar/if/test/database/pointclouds/1702474358169133.npy'
    analyze_npy(test_file)