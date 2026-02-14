# 快速使用指南 - NPY转图像

## 📦 安装

### 步骤1: 复制文件夹
```bash
# 复制 radar_image_converter 到你的项目
cp -r radar_image_converter /your/project/
```

### 步骤2: 安装依赖
```bash
pip install -r radar_image_converter/requirements.txt
```

完成！

---

## 🚀 三种使用方式

### 方式1: 作为Python包导入（推荐）

```python
from radar_image_converter import BEVConverter, normalize_image
import numpy as np

# 加载点云
pc = np.load("pointcloud.npy")  # (N, 5): x,y,z,doppler,intensity

# 转换为BEV
converter = BEVConverter()
bev_img = converter(pc)  # (500, 500, 4)

# 归一化
bev_norm = normalize_image(bev_img, method='percentile')

# 保存
np.save("bev.npy", bev_norm)
```

### 方式2: 命令行批量处理

```bash
python -m radar_image_converter.batch_convert \
    --input_folder ./my_pointclouds \
    --output_folder ./my_images \
    --norm_method percentile
```

### 方式3: 运行示例学习

```bash
python radar_image_converter/example_usage.py
```

---

## 📋 输入NPY文件格式

必须满足:
- **形状**: `(N, 5)` - N为点数
- **列**: `[x, y, z, doppler, intensity]`
- **单位**: 米（真实坐标）

验证:
```python
import numpy as np
pc = np.load("your_file.npy")
print(f"形状: {pc.shape}")  # 应该是 (N, 5)
```

---

## 🎯 常见任务

### 任务1: 生成BEV图像（默认配置）
```bash
python -m radar_image_converter.batch_convert \
    --input_folder ./pointclouds \
    --image_types bev
```

生成: `./images/bev/*.npy` 和 `*.png`

### 任务2: 生成Range图像
```bash
python -m radar_image_converter.batch_convert \
    --input_folder ./pointclouds \
    --image_types range
```

### 任务3: 同时生成BEV和Range
```bash
python -m radar_image_converter.batch_convert \
    --input_folder ./pointclouds \
    --image_types bev range
```

### 任务4: 自定义BEV分辨率
```bash
python -m radar_image_converter.batch_convert \
    --input_folder ./pointclouds \
    --image_types bev \
    --bev_resolution 0.1 \           # 更高分辨率
    --bev_x_range -30 30 \           # 更小范围
    --bev_y_range -30 30
```

---

## 🔧 关键参数速查

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--bev_resolution` | 0.2 | BEV分辨率(m/pixel) | 城市:0.1, 高速:0.3 |
| `--bev_x_range` | -50 50 | X轴范围(米) | 根据场景调整 |
| `--range_azimuth_bins` | 512 | 水平分辨率 | 512或1024 |
| `--norm_method` | percentile | 归一化方法 | percentile(推荐) |
| `--num_workers` | CPU核数 | 并行进程数 | 8-16 |

---

## 💡 集成到你的训练代码

### PyTorch Dataset示例

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import glob

class RadarBEVDataset(Dataset):
    def __init__(self, image_folder):
        self.images = sorted(glob.glob(f"{image_folder}/*.npy"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像 (H, W, 4)
        img = np.load(self.images[idx])
        
        # 转为PyTorch格式 (4, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        return img, idx

# 使用
dataset = RadarBEVDataset("./images/bev")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch_img, batch_idx in loader:
    # batch_img: (32, 4, 500, 500)
    # 你的训练代码...
    pass
```

---

## ❓ 问题排查

### 问题: 图像全黑
**原因**: 点云坐标被预先归一化了  
**解决**: 确保输入点云是真实米制坐标
解决**: 检查点云坐标范围，调整BEV的x_range/y_range参数
**解决**: 降低分辨率或减少并行数
```bash
--bev_resolution 0.5 --num_workers 4
```

### 问题: 找不到模块
**解决**: 确保在包含 `radar_image_converter` 的目录下运行
```bash
cd /path/to/your/project
python -m radar_image_converter.batch_convert ...
```

### 问题: 处理速度慢
**解决**: 增加并行进程
```bash
--num_workers 16
```

---

## 📞 获取帮助

查看完整文档:
```bash
cat radar_image_converter/README.md
```

查看所有命令行选项:
```bash
python -m radar_image_converter.batch_convert --help
```

运行示例:
```bash
python radar_image_converter/example_usage.py
```

---

## ✅ 检查清单

复制到新项目后，确认:

- [ ] 复制了整个 `radar_image_converter` 文件夹
- [ ] 安装了依赖 `pip install -r requirements.txt`
- [ ] 点云文件是 `(N, 5)` 格式
- [ ] 坐标是真实米制（未归一化）
- [ ] 运行示例成功 `python radar_image_converter/example_usage.py`

全部勾选后就可以使用了！
