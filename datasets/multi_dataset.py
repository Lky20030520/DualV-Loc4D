"""
多数据集加载器 - 支持合并多个序列
"""
import os
import json
import numpy as np
from os.path import join, exists
import torch.utils.data as data
from datasets.fusion_dataset import FusionInferDataset, FusionTrainingDataset


class MultiSeqDataset(data.Dataset):
    """合并多个序列的数据集（用于推理/验证）"""
    
    def __init__(self, sequences, dataset_root, sample_interval=1, suffix='_preprocessed_accm7'):
        """
        Args:
            sequences: dict, 格式 {"place": ["seq1", "seq2"], ...}
                      例如 {"if": ["20231213_4", "20240115_3"]}
            dataset_root: str, 数据集根目录
            sample_interval: int, 采样间隔
            suffix: str, 序列后缀（如 '_preprocessed_accm7'）
        """
        super().__init__()
        self.sequences = sequences
        self.dataset_root = dataset_root
        self.sample_interval = sample_interval
        self.suffix = suffix
        
        # 存储各序列的子数据集
        self.subdatasets = []
        self.seq_names = []
        self.seq_lengths = []
        self.cumulative_lengths = [0]
        
        # 加载所有序列
        for place, seqs in sequences.items():
            for seq in seqs:
                seq_path = f"{place}/{seq.replace('/', '_')}{suffix}"
                seq_name = f"{place}/{seq}"
                
                try:
                    subdataset = FusionInferDataset(
                        seq=seq_path,
                        dataset_root=dataset_root,
                        sample_inteval=sample_interval
                    )
                    self.subdatasets.append(subdataset)
                    self.seq_names.append(seq_name)
                    self.seq_lengths.append(len(subdataset))
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(subdataset))
                    print(f"✅ 加载序列: {seq_name} ({len(subdataset)} 帧)")
                except Exception as e:
                    print(f"⚠️ 跳过序列 {seq_name}: {e}")
        
        if len(self.subdatasets) == 0:
            raise ValueError("没有成功加载任何序列！")
        
        # 合并所有 poses
        self.poses = np.concatenate([ds.poses for ds in self.subdatasets], axis=0)
        
        print(f"📊 MultiSeqDataset 统计:")
        print(f"   总序列数: {len(self.subdatasets)}")
        print(f"   总帧数: {len(self.poses)}")
        print(f"   序列分布: {dict(zip(self.seq_names, self.seq_lengths))}")
    
    def __getitem__(self, index):
        # 找到对应的子数据集
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= index < end:
                local_index = index - start
                return self.subdatasets[i][local_index]
        raise IndexError(f"Index {index} out of range")
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def get_sequence_info(self, index):
        """返回给定索引所属的序列信息"""
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= index < end:
                return {
                    'seq_name': self.seq_names[i],
                    'local_index': index - start,
                    'global_index': index
                }
        return None


class MultiSeqTrainingDataset(data.Dataset):
    """合并多个序列的训练数据集（支持 Hard Mining）"""
    
    def __init__(self, sequences, dataset_root, max_frames=None, 
                 cache_path=None, sample_interval=1, suffix='_preprocessed_accm7'):
        """
        Args:
            sequences: dict, 同 MultiSeqDataset
            max_frames: int, 最大帧数限制（用于调试）
            cache_path: str, hard mining 缓存路径
        """
        super().__init__()
        
        # 先创建推理数据集（用于构建 base_dataset）
        self.base_dataset = MultiSeqDataset(
            sequences=sequences,
            dataset_root=dataset_root,
            sample_interval=sample_interval,
            suffix=suffix
        )
        
        # 应用最大帧数限制
        if max_frames and max_frames < len(self.base_dataset):
            print(f"⚠️ 限制训练帧数: {len(self.base_dataset)} → {max_frames}")
            # 这里需要截断 poses 和其他属性
            # 简化实现：只截断 poses，实际使用时动态检查
            self.max_frames = max_frames
            self.poses = self.base_dataset.poses[:max_frames]
        else:
            self.max_frames = len(self.base_dataset)
            self.poses = self.base_dataset.poses
        
        # 训练相关参数
        self.pos_thres = 10
        self.neg_thres = 50
        self.num_neg = 5
        
        # 计算正负样本索引
        self.positives, self.negatives = self._compute_pos_neg_samples()
        
        # Hard Mining 相关
        self.mining = False
        self.cache = cache_path
        self.h5feat = None
    
    def _compute_pos_neg_samples(self):
        """计算正负样本索引（同原始 FusionTrainingDataset）"""
        positives = []
        negatives = []
        num_frames = len(self.poses)
        trans = self.poses[:, [3, 7, 11]]
        
        print("正在计算正负样本索引...")
        for i in range(num_frames):
            dist = np.linalg.norm(trans - trans[i], axis=1)
            pos = np.where((dist < self.pos_thres) & (dist > 0.01))[0]
            if len(pos) == 0: 
                pos = np.array([i])
            positives.append(pos)
            neg = np.where(dist > self.neg_thres)[0]
            negatives.append(neg)
        
        return positives, negatives
    
    def refreshCache(self):
        """加载 hard mining 缓存到内存"""
        if self.cache is None: 
            return
        if not exists(self.cache): 
            return
        import h5py
        with h5py.File(self.cache, 'r') as h5:
            self.h5feat = np.array(h5.get("features"))
    
    def __getitem__(self, index):
        """同 FusionTrainingDataset 的旋转增强版本"""
        import random
        import torchvision.transforms.functional as TF
        import torch
        
        # 1. 确定正负样本索引
        if self.mining and self.h5feat is not None:
            q_feat = self.h5feat[index]
            pos_idx = np.random.choice(self.positives[index])
            neg_candidates = self.negatives[index]
            if len(neg_candidates) > 0:
                subset = np.random.choice(neg_candidates, min(len(neg_candidates), 1000))
                neg_feats = self.h5feat[subset]
                feat_dists = np.linalg.norm(neg_feats - q_feat, axis=1)
                hardest_idx = np.argsort(feat_dists)[:self.num_neg]
                neg_indices = subset[hardest_idx]
            else:
                neg_indices = np.random.choice(range(len(self.poses)), self.num_neg)
        else:
            pos_idx = np.random.choice(self.positives[index])
            neg_indices = np.random.choice(self.negatives[index], self.num_neg)
        
        # 2. 读取并增强数据
        def load_and_augment(idx):
            bev, range_tensor, _ = self.base_dataset[idx]
            angle = random.uniform(-30, 30)
            bev_rotated = TF.rotate(bev, angle)
            c, h, w = range_tensor.shape
            shift_ratio = angle / 360.0
            range_shifted = torch.roll(range_tensor, shifts=int(w * shift_ratio), dims=-1)
            return bev_rotated, range_shifted
        
        q_bev, q_range = load_and_augment(index)
        p_bev, p_range = load_and_augment(pos_idx)
        
        n_bevs, n_ranges = [], []
        for ni in neg_indices:
            nb, nr = load_and_augment(ni)
            n_bevs.append(nb)
            n_ranges.append(nr)
        
        n_bevs = torch.stack(n_bevs)
        n_ranges = torch.stack(n_ranges)
        
        return (q_bev, q_range), (p_bev, p_range), (n_bevs, n_ranges), index
    
    def __len__(self):
        return len(self.poses)


def load_dataset_config(config_path='configs/dataset_splits.json'):
    """加载数据集配置文件"""
    if not exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_datasets_from_config(mode, config_path='configs/dataset_splits.json', 
                                dataset_root='/mnt/kaiyan/datasets/SNAIL',
                                sample_interval=10, suffix='_preprocessed_accm7'):
    """
    从配置文件创建数据集
    
    Args:
        mode: 'train', 'val', 'test'
        config_path: 配置文件路径
        dataset_root: 数据集根目录
        sample_interval: 采样间隔
        suffix: 序列后缀
    
    Returns:
        对于 train: (train_dataset, None)
        对于 val/test: (db_dataset, query_dataset)
    """
    config = load_dataset_config(config_path)
    
    if mode not in config:
        raise ValueError(f"配置中没有 '{mode}' 模式")
    
    mode_config = config[mode]
    
    if mode == 'train':
        # 训练模式：合并 query + database 作为训练集
        all_sequences = {}
        for place in set(list(mode_config.get('query', {}).keys()) + 
                        list(mode_config.get('database', {}).keys())):
            all_sequences[place] = (
                mode_config.get('query', {}).get(place, []) +
                mode_config.get('database', {}).get(place, [])
            )
        
        train_dataset = MultiSeqTrainingDataset(
            sequences=all_sequences,
            dataset_root=dataset_root,
            sample_interval=sample_interval,
            suffix=suffix
        )
        return train_dataset, None
    
    else:  # val or test
        # 验证/测试模式：分别创建 database 和 query
        db_dataset = MultiSeqDataset(
            sequences=mode_config['database'],
            dataset_root=dataset_root,
            sample_interval=sample_interval,
            suffix=suffix
        )
        
        query_dataset = MultiSeqDataset(
            sequences=mode_config['query'],
            dataset_root=dataset_root,
            sample_interval=sample_interval,
            suffix=suffix
        )
        
        return db_dataset, query_dataset


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 测试加载
    print("\n========== 测试 Train 数据集 ==========")
    train_ds, _ = create_datasets_from_config('train')
    print(f"训练集总大小: {len(train_ds)}")
    
    print("\n========== 测试 Val 数据集 ==========")
    db_ds, q_ds = create_datasets_from_config('val')
    print(f"Database 大小: {len(db_ds)}")
    print(f"Query 大小: {len(q_ds)}")
    
    print("\n========== 测试数据加载 ==========")
    sample = train_ds[0]
    print(f"训练样本结构: {type(sample)}")
    if isinstance(sample, tuple):
        print(f"  Query: {type(sample[0])}")
        print(f"  Positive: {type(sample[1])}")
        print(f"  Negatives: {type(sample[2])}")
