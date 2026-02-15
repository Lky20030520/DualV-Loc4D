# Full Pipeline 使用说明

## 脚本功能

`full_pipeline.py` 是一个自动化全流程处理脚本，整合了以下三个步骤：

1. **批量预处理（PCD → NPY）**: 调用 `batch_preprocess.py` 处理所有 46 个数据集
2. **生成 BEV 图像**: 对每个数据集调用 `npy2bev.py` 
3. **生成 Range 图像**: 对每个数据集调用 `npy2range.py`

## 快速开始

### 最简单用法（使用默认参数）

```bash
cd /home/kaiyan/BEVPlace3/preprocess/pcd_to_npy_toolkit
python full_pipeline.py
```

这将处理所有 46 个数据集，并生成对应的 BEV 和 Range 图像。

### 自定义参数

```bash
python full_pipeline.py \
    --base_dir /datasets/snail-radar \
    --save_folder datasets/snail/radar \
    --accum_win 7 \
    --gap_size 4 \
    --norm_type raw \
    --maximum_range 120.0 \
    --add_suffix 7frame_gap4
```

## 参数说明

### 🗂️ 数据路径参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_dir` | `/datasets/snail-radar` | 原始数据集根目录 |
| `--save_folder` | `datasets/snail/radar` | 预处理结果保存根目录 |

### ⚙️ 预处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--accum_win` | `7` | 多帧堆叠窗口大小 |
| `--target_points` | `-1` | 目标点数（-1=保留全部） |
| `--norm_type` | `raw` | 归一化类型：`raw`/`range`/`sphere` |
| `--maximum_range` | `120.0` | 最大距离（米） |
| `--add_suffix` | `""` | 输出文件夹自定义后缀 |
| `-o, --generate_original` | `False` | 同时生成未去噪的原始点云 |
| `-i, --generate_images` | `False` | 从原始数据复制相机图像 |

### 📏 采样参数（二选一）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gap_size` | `None` | 按固定帧间隔采样 |
| `--interval_dist` | `None` | 按空间距离采样（米） |

### 🔀 流程控制参数

| 参数 | 说明 |
|------|------|
| `--skip_preprocess` | 跳过预处理，仅生成图像（适用于 NPY 已存在的情况） |
| `--skip_bev` | 跳过 BEV 图像生成 |
| `--skip_range` | 跳过 Range 图像生成 |

## 使用场景

### 场景 1: 完整流程（从零开始）

```bash
python full_pipeline.py \
    --accum_win 7 \
    --gap_size 4 \
    --add_suffix 7f_gap4
```

**执行内容**：
- ✅ 预处理所有 46 个数据集（生成 NPY）
- ✅ 生成所有 BEV 图像
- ✅ 生成所有 Range 图像

**输出示例**：
```
datasets/snail/radar/
├── bc/
│   └── 20230920_1_preprocessed_accm7_7f_gap4/
│       ├── pointclouds/           # NPY 文件
│       ├── pointclouds_bev/       # BEV 图像
│       └── range_image/           # Range 图像
├── if/
│   └── 20231208_4_preprocessed_accm7_7f_gap4/
│       ├── pointclouds/
│       ├── pointclouds_bev/
│       └── range_image/
└── ...
```

### 场景 2: 仅为已有 NPY 生成图像

```bash
python full_pipeline.py \
    --skip_preprocess \
    --accum_win 7 \
    --add_suffix 7f_gap4
```

**执行内容**：
- ⏭️  跳过预处理
- ✅ 生成所有 BEV 图像
- ✅ 生成所有 Range 图像

### 场景 3: 仅生成 BEV 图像

```bash
python full_pipeline.py \
    --skip_preprocess \
    --skip_range \
    --accum_win 7
```

### 场景 4: 单帧模式（无运动拖尾）

```bash
python full_pipeline.py \
    --accum_win 1 \
    --add_suffix singleframe
```

**说明**: `accum_win=1` 不会有运动模糊/拖尾现象

### 场景 5: 距离采样模式

```bash
python full_pipeline.py \
    --interval_dist 5.0 \
    --add_suffix dist5m
```

**说明**: 每移动 5 米取一帧，空间均匀采样

## 数据集覆盖范围

脚本将处理以下 8 个地点共 46 个序列：

| 地点 | 序列数 | 示例 |
|------|--------|------|
| **bc** | 5 | 20230920_1, 20230921_2, ... |
| **sl** | 9 | 20230920_2, 20230921_3, ... |
| **ss** | 6 | 20230921_4, 20231019_2, ... |
| **if** | 7 | 20231208_4, 20240116_5, ... |
| **iaf** | 8 | 20231201_2, 20231208_5, ... |
| **iaef** | 3 | 20240113_5, 20240115_2, ... |
| **st** | 3 | 20231208_1, 20231213_1, ... |
| **81r** | 3 | 20240116_2, 20240123_2, ... |

## 日志输出

脚本会输出详细的进度信息：

```
2026-02-15 10:00:00 - INFO - ============================================================
2026-02-15 10:00:00 - INFO - STEP 1: Running batch_preprocess.py to generate NPY files
2026-02-15 10:00:00 - INFO - ============================================================
2026-02-15 10:05:30 - INFO - ✅ Batch preprocessing completed successfully
2026-02-15 10:05:30 - INFO - ============================================================
2026-02-15 10:05:30 - INFO - STEP 2&3: Generating BEV and Range images
2026-02-15 10:05:30 - INFO - ============================================================
2026-02-15 10:05:31 - INFO - [1/46] bc/20230920_1
2026-02-15 10:05:31 - INFO - 📁 Processing: bc/20230920_1
2026-02-15 10:05:32 - INFO -   🗺️  Generating BEV images -> .../pointclouds_bev
2026-02-15 10:05:35 - INFO -   ✅ BEV images generated
2026-02-15 10:05:35 - INFO -   📊 Generating Range images -> .../range_image
2026-02-15 10:05:38 - INFO -   ✅ Range images generated
...
2026-02-15 10:30:00 - INFO - ✅ All image generation completed!
2026-02-15 10:30:00 - INFO - 🎉 Full pipeline completed successfully!
```

## 注意事项

1. **磁盘空间**: 确保有足够的存储空间（每个数据集约 1-2GB）
2. **处理时间**: 完整流程可能需要数小时，建议使用 `screen` 或 `tmux`
3. **中断恢复**: 如果中途中断，可以使用 `--skip_preprocess` 从图像生成步骤继续
4. **路径一致性**: `--save_folder` 和 `--add_suffix` 必须与之前的预处理保持一致才能找到 NPY 文件

## 故障排除

### 问题 1: "Folder not found (skipping)"

**原因**: NPY 文件夹不存在  
**解决**: 移除 `--skip_preprocess` 参数，重新运行预处理

### 问题 2: 中断后如何继续

```bash
# 仅生成图像，跳过预处理
python full_pipeline.py --skip_preprocess --accum_win 7
```

### 问题 3: 只想处理特定地点

**解决**: 修改 `full_pipeline.py` 中的 `data_dict`，注释掉不需要的地点

## 性能优化建议

- 使用 SSD 存储可大幅提升速度
- 多核 CPU 可加速并行处理
- 对于大批量处理，建议分批次运行并监控磁盘空间
