# 多数据集训练系统使用指南

## 📋 概述

支持两种数据集加载模式：
1. **单序列模式**（向后兼容）- 使用命令行参数指定单个训练/验证序列
2. **多序列配置文件模式**（推荐）- 使用JSON配置文件加载多个序列并自动合并

---

## 🚀 快速开始

### 方式1: 单序列模式（原有方式）

```bash
python main_fusion_rangerem.py --mode train \
  --dataset_root /mnt/kaiyan/datasets/SNAIL \
  --train_seq if/20231213_4_preprocessed_accm7 \
  --val_db_seq if/20240115_3_preprocessed_accm7 \
  --val_q_seq if/20240116_5_preprocessed_accm7 \
  --runsPath ./runs \
  --cachePath ./cache/fusion_single \
  --nEpochs 20
```

### 方式2: 多序列配置文件模式（新）

```bash
python main_fusion_rangerem.py --mode train \
  --dataset_root /mnt/kaiyan/datasets/SNAIL \
  --dataset_config configs/dataset_splits.json \
  --runsPath ./runs \
  --cachePath ./cache/fusion_multi \
  --nEpochs 20 \
  --threads 8
```

---

## 📁 配置文件结构

配置文件路径：`configs/dataset_splits.json`

```json
{
  "train": {
    "query": {
      "sl": ["20231105_aft_4"],
      "ss": ["20231105_aft_5"]
    },
    "database": {
      "sl": ["20231105_2"],
      "ss": ["20231109_4"],
      "if": ["20240116_5"],
      "81r": ["20240123_2"]
    }
  },
  "val": {
    "query": {
      "iaf": ["20231201_3"],
      "if": ["20240115_3"]
    },
    "database": {
      "iaf": ["20231201_2"],
      "if": ["20231213_4"]
    }
  },
  "test": {
    "bc": {
      "query": ["20230921_2", "20231007_4", ...],
      "database": ["20230920_1"]
    },
    ...
  }
}
```

### 配置说明

- **train**: 训练集配置（query + database 会合并为一个训练集）
- **val**: 验证集配置（用于训练过程中的 recall 计算）
- **test**: 测试集配置（用于最终评估）

每个地点（place）可以包含多个序列，程序会自动：
1. 加载所有序列
2. 合并 poses 和索引
3. 在训练时统一采样

---

## 🔧 完整参数说明

### 数据集参数

| 参数 | 描述 | 默认值 | 示例 |
|------|------|--------|------|
| `--dataset_root` | 数据集根目录 | `./datasets/snail` | `/mnt/kaiyan/datasets/SNAIL` |
| `--dataset_config` | 多序列配置文件 | `''` (不使用) | `configs/dataset_splits.json` |
| `--train_seq` | 单序列训练集 | `''` | `if/20231213_4_preprocessed_accm7` |
| `--val_db_seq` | 单序列验证DB | `''` | `if/20240115_3_preprocessed_accm7` |
| `--val_q_seq` | 单序列验证Query | `''` | `if/20240116_5_preprocessed_accm7` |
| `--sample_interval` | 采样间隔 | `10` | `10` |

### 训练参数

| 参数 | 描述 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--batchSize` | 训练batch | `1` | `1-4` |
| `--cacheBatchSize` | 推理batch | `4` | `8-16` |
| `--threads` | 数据加载线程 | `0` | `8` |
| `--nEpochs` | 训练轮数 | `20` | `20-30` |
| `--lr` | 学习率 | `0.0001` | `0.0001` |

### 模型参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--stage` | 训练阶段 | `'B'` |
| `--bev_path` | BEV预训练权重 | `runs/fusion_Feb14_17-38-13!/model_best.pth.tar` |
| `--load_from` | 恢复训练checkpoint | `''` |
| `--cachePath` | 缓存目录 | `./cache/fusion_integrated5/` |

---

## 📊 数据合并逻辑

### 训练模式

```
train配置:
  query: {"sl": [seq1], "ss": [seq2]}
  database: {"sl": [seq3], "if": [seq4, seq5]}

合并后训练集:
  {"sl": [seq1, seq3], "ss": [seq2], "if": [seq4, seq5]}
  
总帧数 = sum(各序列帧数)
```

### 验证/测试模式

```
val配置:
  query: {"if": [seq1, seq2]}
  database: {"if": [seq3], "iaf": [seq4]}

Database数据集: if/seq3 + iaf/seq4
Query数据集: if/seq1 + if/seq2

Recall计算: Query中每帧 vs Database全局检索
```

---

## 💡 使用建议

### 1. 训练时使用多序列
- ✅ 增加数据多样性
- ✅ 提升泛化能力
- ✅ 避免过拟合特定环境

```bash
# 推荐：多地点多序列训练
--dataset_config configs/dataset_splits.json
```

### 2. 调试时使用单序列
- ✅ 快速验证代码
- ✅ 节省时间
- ✅ 便于定位问题

```bash
# 调试：单序列快速测试
--train_seq if/20231213_4_preprocessed_accm7
```

### 3. 性能优化
- 设置 `--threads 8` 启用多线程加载
- 增大 `--cacheBatchSize` 加速特征提取
- 使用SSD存储数据集提升IO速度

---

## 🔍 测试数据集加载

```bash
# 测试配置文件是否正确
cd /home/kaiyan/BEVPlace3
python datasets/multi_dataset.py
```

输出示例：
```
========== 测试 Train 数据集 ==========
✅ 加载序列: sl/20231105_aft_4 (728 帧)
✅ 加载序列: ss/20231105_aft_5 (645 帧)
✅ 加载序列: sl/20231105_2 (892 帧)
...
📊 MultiSeqDataset 统计:
   总序列数: 6
   总帧数: 4521
   
========== 测试 Val 数据集 ==========
Database 大小: 2145
Query 大小: 1087
```

---

## ⚙️ 高级功能

### 自定义数据集配置

创建新的配置文件 `configs/my_splits.json`:

```json
{
  "train": {
    "query": {"my_place": ["seq1", "seq2"]},
    "database": {"my_place": ["seq3"]}
  },
  ...
}
```

使用:
```bash
--dataset_config configs/my_splits.json
```

### 序列命名规则

程序会自动添加后缀 `_preprocessed_accm7`:
```
配置: "if": ["20231213_4"]
实际路径: {dataset_root}/if/20231213_4_preprocessed_accm7/
```

如需自定义后缀，修改 `datasets/multi_dataset.py` 中的 `suffix` 参数。

---

## 🐛 常见问题

### Q1: 配置文件不生效？
**A:** 确保使用了 `--dataset_config` 参数，优先级：
```
配置文件模式 > 单序列模式
```

### Q2: 序列路径找不到？
**A:** 检查：
1. `dataset_root` 是否正确
2. 序列名称是否匹配（区分大小写）
3. 后缀 `_preprocessed_accm7` 是否存在

### Q3: 内存不足？
**A:** 减少：
- `--cacheBatchSize`（推理batch）
- 配置文件中的序列数量
- 使用 `--sample_interval` 增大采样间隔

---

## 📝 示例命令汇总

```bash
# 1. 单序列训练（调试）
python main_fusion_rangerem.py --mode train \
  --train_seq if/20231213_4_preprocessed_accm7 \
  --val_db_seq if/20240115_3_preprocessed_accm7 \
  --val_q_seq if/20240116_5_preprocessed_accm7 \
  --threads 8 --nEpochs 5

# 2. 多序列训练（完整）
python main_fusion_rangerem.py --mode train \
  --dataset_config configs/dataset_splits.json \
  --threads 8 --nEpochs 20 \
  --cachePath ./cache/fusion_multi

# 3. 多序列测试
python main_fusion_rangerem.py --mode test \
  --dataset_config configs/dataset_splits.json \
  --load_from runs/fusion_Feb15_XX-XX-XX \
  --threads 8

# 4. 从checkpoint恢复训练
python main_fusion_rangerem.py --mode train \
  --dataset_config configs/dataset_splits.json \
  --load_from runs/fusion_Feb15_14-30-00 \
  --threads 8
```
