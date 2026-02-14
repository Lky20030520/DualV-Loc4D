# NPY to Image Converter

将NumPy点云文件(NPY)转换为BEV/Range图像的独立工具。

## 📦 功能

- ✅ **BEV (Bird's Eye View)** 鸟瞰图投影
- ✅ **Range Image** 球面距离图投影  
- ✅ 多通道特征（密度、强度、高度、多普勒等）
- ✅ 批量处理和并行加速

## 🚀 快速开始

### 作为Python包使用

```python
import numpy as np
from radar_image_converter import BEVConverter, RangeImageConverter, normalize_image

# 加载点云 (N, 5): x, y, z, doppler, intensity
pc = np.load("pointcloud.npy")

# 生成BEV图像
bev_converter = BEVConverter(
    x_range=(-50, 50),
    y_range=(-50, 50),
    resolution=0.2
)
bev_img = bev_converter(pc)  # (500, 500, 4)

# 归一化
bev_norm = normalize_image(bev_img, method='percentile')

# 保存
np.save("bev_image.npy", bev_norm)
```

### 批量转换（命令行）

```bash
python -m radar_image_converter.batch_convert \
    --input_folder ./pointclouds \
    --output_folder ./images \
    --bev_resolution 0.2 \
    --norm_method percentile \
    --num_workers 8
```

## 📊 输入输出格式

### 输入
- **格式**: NPY文件 (使用 `numpy.load()` 加载)
- **形状**: `(N, 5)` 其中 N 是点数
- **通道顺序**: `[x, y, z, doppler, intensity]`
- **坐标单位**: 米（真实坐标）

### 输出

#### BEV图像
- **形状**: `(H, W, 4)` 
- **默认大小**: `(500, 500, 4)` 表示100m×100m，0.2m分辨率
- **通道**:
  - `[0]` 点密度 (log scale)
  - `[1]` 平均强度
  - `[2]` 最大高度
  - `[3]` 平均多普勒速度

#### Range图像
- **形状**: `(H, W, 6)`
- **默认大小**: `(64, 512, 6)` 
- **通道**:
  - `[0]` 深度
  - `[1]` 强度
  - `[2]` 多普勒速度
  - `[3-5]` XYZ坐标

## 🔧 参数配置

### BEV参数

```python
BEVConverter(
    x_range=(-50, 50),      # X轴范围（米）
    y_range=(-50, 50),      # Y轴范围（米）
    resolution=0.2,         # 分辨率（米/像素）
    height_range=(-2, 5)    # 高度过滤范围（米）
)
```

**常用配置**:
- 城市环境: `x_range=(-30, 30), resolution=0.1` (60m×60m, 高分辨率)
- 高速场景: `x_range=(-80, 80), resolution=0.3` (160m×160m, 大范围)

### Range参数

```python
RangeImageConverter(
    azimuth_bins=512,        # 水平分辨率（360°）
    elevation_bins=64,       # 垂直分辨率
    max_range=100.0          # 最大距离（米）
)
```

### 归一化方法

- **`percentile`** (推荐): 基于99分位数，鲁棒性强
- **`minmax`**: 线性归一化到[0,1]
- **`standardize`**: 零均值标准化

## 📂 目录结构

```
radar_image_converter/
├── __init__.py              # 包初始化
├── converters.py            # BEV和Range转换器
├── utils.py                 # 工具函数
├── batch_convert.py         # 批量处理脚本
├── README.md                # 本文件
├── requirements.txt         # 依赖
└── example_usage.py         # 使用示例
```

## 💻 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装:
```bash
pip install numpy opencv-python tqdm
```

## 🎯 使用案例

### 案例1: 单个文件转换

```python
from radar_image_converter import BEVConverter, normalize_image
import numpy as np

# 加载
pc = np.load("scan_001.npy")

# 转换
converter = BEVConverter()
bev = converter(pc)

# 归一化并保存
bev_norm = normalize_image(bev, method='percentile')
np.save("bev_001.npy", bev_norm)
```

### 案例2: 批量处理

```bash
# 处理整个文件夹
python -m radar_image_converter.batch_convert \
    --input_folder /data/pointclouds \
    --output_folder /data/images \
    --image_types bev range \
    --num_workers 16
```

### 案例3: 自定义参数

```python
from radar_image_converter import BEVConverter, RangeImageConverter
import numpy as np

# 大范围BEV
bev_converter = BEVConverter(
    x_range=(-100, 100),
    y_range=(-100, 100),
    resolution=0.5  # 0.5m/pixel
)

# 高分辨率Range
range_converter = RangeImageConverter(
    azimuth_bins=1024,
    elevation_bins=128
)

pc = np.load("pointcloud.npy")
bev = bev_converter(pc)
range_img = range_converter(pc)
```

### 案例4: PyTorch数据集集成

```python
import torch
from torch.utils.data import Dataset
import numpy as np
from radar_image_converter import normalize_image

class RadarImageDataset(Dataset):
    def __init__(self, image_folder):
        self.images = sorted(glob.glob(f"{image_folder}/*.npy"))
    
    def __getitem__(self, idx):
        img = np.load(self.images[idx])  # (H, W, C)
        img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
        return img
    
    def __len__(self):
        return len(self.images)

# 使用
dataset = RadarImageDataset("./images/bev")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

## ⚙️ 命令行选项

完整选项列表:

```bash
python -m radar_image_converter.batch_convert --help
```

常用选项:
- `--input_folder`: 输入NPY文件夹 (必需)
- `--output_folder`: 输出文件夹
- `--image_types`: 生成类型 [bev, range]
- `--bev_resolution`: BEV分辨率
- `--norm_method`: 归一化方法
- `--num_workers`: 并行进程数
- `--no_visualization`: 不生成PNG可视化

## ❓ 常见问题

### Q1: 图像全黑？
确保输入点云使用真实米制坐标（不要预先归一化）。

### Q2: 内存不足？
降低分辨率或减少并行进程数:
```bash
--bev_resolution 0.5 --num_workers 4
```

### Q3: 处理速度慢？
增加并行进程:
```bash
--num_workers 16
```

### Q4: 如何移植到其他项目？
直接复制整个 `radar_image_converter` 文件夹即可，确保安装了依赖。

## 📄 LICENSE

Same as TransLoc4D project

## 🙏 致谢

本工具从 [TransLoc4D](https://github.com/transmissible/TransLoc4D) 项目提取并改进。
