import numpy as np

# 替换为你要查看的npy文件名
file_path = "/mnt/kaiyan/datasets/snail_radar/81r/20240116_2/eagleg7_npy/1705391895.642129975.npy"

# 加载npy文件
data = np.load(file_path)

# 打印数组形状
print(f"文件 {file_path} 的形状为: {data.shape}")

# 打印前25行数据
print("\n前25行数据:")
print(data[:25])

# 打印第四列（索引3）不为0的行
try:
    # 检查数组是否至少有4列（确保第四列存在）
    if data.ndim < 2 or data.shape[1] < 4:
        print("\n数据列数不足4列，无法查询第四列")
    else:
        # 筛选第四列（索引3）不等于0的行
        mask = data[:, 3] != 0
        non_zero_rows = data[mask]
        
        if len(non_zero_rows) == 0:
            print("\n第四列（索引3）中没有值不为0的行")
        else:
            print(f"\n第四列（索引3）值不为0的行（共{len(non_zero_rows)}行）:")
            # 若符合条件的行太多，可限制打印数量（如前50行），避免输出过长
            print(non_zero_rows[:50])  # 只打印前50行，可根据需要调整
except Exception as e:
    print(f"\n查询第四列非0行时出错: {e}")