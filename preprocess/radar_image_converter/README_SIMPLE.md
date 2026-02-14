# NPY to Image - 极简使用说明

只做一件事：**NPY点云 → BEV/Range图像**

---

## 快速开始

### 1. 安装依赖
```bash
pip install numpy opencv-python tqdm
```

### 2. 转换单个文件
```bash
python simple_convert.py input.npy output_bev.npy
```

### 3. 批量转换
```bash
python -m batch_convert --input_folder ./npys --output_folder ./images
```

完成！

---

## 输入格式

你的NPY文件:
```python
import numpy as np
pc = np.load("your_file.npy")
# 形状: (N, 5)
# 列: [x, y, z, doppler, intensity]
# x,y,z单位: 米
```

---

## 输出格式

### BEV图像
- 形状: `(500, 500, 4)`
- 通道: [密度, 强度, 高度, 多普勒]

### Range图像  
- 形状: `(64, 512, 6)`
- 通道: [深度, 强度, 多普勒, x, y, z]

---

## 使用示例

### Python代码
```python
from converters import BEVConverter
from utils import normalize_image
import numpy as np

# 加载
pc = np.load("scan.npy")

# 转换
converter = BEVConverter()
bev = converter(pc)

# 归一化
bev_norm = normalize_image(bev, method='percentile')

# 保存
np.save("bev.npy", bev_norm)
```

### 命令行
```bash
# 单文件
python simple_convert.py scan.npy bev.npy

# 批量
python -m batch_convert \
    --input_folder ./my_npys \
    --output_folder ./my_images
```

---

## 参数调整

### 改变分辨率
```python
# 高分辨率
converter = BEVConverter(resolution=0.1)  # 0.1米/像素

# 低分辨率  
converter = BEVConverter(resolution=0.5)  # 0.5米/像素
```

### 改变范围
```python
# 大范围
converter = BEVConverter(x_range=(-100, 100), y_range=(-100, 100))

# 小范围
converter = BEVConverter(x_range=(-20, 20), y_range=(-20, 20))
```

---

## 所有文件说明

| 文件 | 用途 |
|------|------|
| `converters.py` | 核心转换器 |
| `utils.py` | 归一化函数 |
| `simple_convert.py` | 单文件转换脚本 ⭐ |
| `batch_convert.py` | 批量转换脚本 ⭐ |
| `example_usage.py` | 代码示例 |

⭐ 标记的是直接可运行的脚本

---

## 完成！

就这么简单。有NPY文件，运行脚本，得到图像。
