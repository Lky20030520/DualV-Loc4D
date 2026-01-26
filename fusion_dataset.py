import os
from os.path import join, exists, splitext
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import h5py
import faiss
from torchvision import transforms

# ================= 配置区域 =================

def get_transforms():
    range_tf = transforms.Compose([
        transforms.Resize((70, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    bev_tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return bev_tf, range_tf

BEV_TF, RANGE_TF = get_transforms()

def extract_timestamp(filename):
    """从文件名提取时间戳"""
    base_name = splitext(os.path.basename(filename))[0]
    try:
        if '_' in base_name:
            parts = base_name.split('_')
            candidate = parts[0]
            if '.' not in candidate and len(parts) > 1 and parts[1].isdigit():
                 candidate = f"{parts[0]}.{parts[1]}"
            try:
                return float(candidate)
            except:
                pass 
        return float(base_name)
    except Exception:
        import re
        nums = re.findall(r"\d+\.?\d*", base_name)
        if nums:
            return float(nums[0])
        raise ValueError(f"无法从文件名提取时间戳: {filename}")

# ================= 核心数据集类 =================

class FusionInferDataset(data.Dataset):
    def __init__(self, seq, dataset_root='datasets/snail', sample_inteval=1):
        super().__init__()
        self.sample_inteval = sample_inteval
        self.seq = seq
        
        # =========================================================
        # [修改点] 适配截图中的目录结构: root/radar/{seq}/...
        # =========================================================
        # 假设 dataset_root 是 './datasets/snail'
        # 这里的 seq 就是 'if_20231208_4' 或 'if_20240116_5'
        
        # 1. 尝试直接拼接序列名
        base_dir = join(dataset_root, 'radar', seq)
        
        # 兼容性检查：如果 dataset_root 里面没 radar 层，直接拼 seq
        if not exists(join(dataset_root, 'radar')):
            base_dir = join(dataset_root, seq)

        if not exists(base_dir):
            raise FileNotFoundError(f"找不到序列目录: {base_dir}")

        self.bev_dir = join(base_dir, 'bev_image') # BEV 文件夹名，看截图应该是这个
        self.range_dir = join(base_dir, 'range_image') # Range 文件夹名
        self.pose_dir = join(base_dir, 'poses') # Pose 文件夹名

        # 检查路径是否存在
        if not exists(self.bev_dir):
            # 截图里好像没有 enhanced_bev，只有 bev_image，这里做个兼容
            self.bev_dir = join(base_dir, 'bev_image') 
            
        if not exists(self.range_dir):
            # 截图里有 range_image
            pass 

        if not exists(self.bev_dir) or not exists(self.range_dir):
            raise FileNotFoundError(f"数据子目录不存在，请检查:\n{self.bev_dir}\n{self.range_dir}")

        # 2. 读取所有文件并按时间戳匹配
        self.bev_files = sorted([f for f in os.listdir(self.bev_dir) if f.endswith('.png')])
        self.range_files = sorted([f for f in os.listdir(self.range_dir) if f.endswith('.png')])
        
        self.range_map = {extract_timestamp(f): f for f in self.range_files}
        
        # 3. 过滤并同步数据 (BEV <-> Range)
        self.pairs = [] 
        
        for bev_f in self.bev_files:
            try:
                ts = extract_timestamp(bev_f)
                if ts in self.range_map:
                    range_f = self.range_map[ts]
                    self.pairs.append({
                        'bev': join(self.bev_dir, bev_f),
                        'range': join(self.range_dir, range_f),
                        'ts': ts
                    })
            except:
                continue
        
        if not self.pairs:
            # 尝试宽松匹配（如果文件名完全一致）
            common_names = set(self.bev_files) & set(self.range_files)
            for name in common_names:
                self.pairs.append({
                    'bev': join(self.bev_dir, name),
                    'range': join(self.range_dir, name),
                    'ts': extract_timestamp(name)
                })
        
        if not self.pairs:
            raise ValueError(f"未找到匹配的 BEV 和 Range 图像对！Root: {dataset_root}")
        
        self.pairs.sort(key=lambda x: x['ts'])
        self.pairs = self.pairs[::self.sample_inteval]
        
        # 4. 加载 Pose
        self.poses = self._load_poses()

    def _load_poses(self):
        poses = []
        # 如果找不到 Pose 目录或目录为空，使用 Dummy Pose
        if not exists(self.pose_dir) or len(os.listdir(self.pose_dir)) == 0:
            print("⚠️ 警告: 未找到 Pose 文件，使用全0 Dummy Pose。")
            return np.zeros((len(self.pairs), 12), dtype=np.float32)

        pose_files = [f for f in os.listdir(self.pose_dir) if 'txt' in f or 'csv' in f]
        pose_map = {extract_timestamp(f): join(self.pose_dir, f) for f in pose_files}
        sorted_pose_ts = sorted(list(pose_map.keys()))
        
        if not sorted_pose_ts:
             return np.zeros((len(self.pairs), 12), dtype=np.float32)

        sorted_pose_ts_np = np.array(sorted_pose_ts)
        
        for p in self.pairs:
            img_ts = p['ts']
            idx = np.argmin(np.abs(sorted_pose_ts_np - img_ts))
            closest_ts = sorted_pose_ts_np[idx]
            
            pose_file = pose_map[closest_ts]
            try:
                with open(pose_file, 'r') as f:
                    line = f.readline().replace(',', ' ').strip()
                    vals = [float(x) for x in line.split()]
                    if len(vals) > 12: vals = vals[-12:]
                    if len(vals) < 12: vals = vals + [0]*(12-len(vals))
                    poses.append(vals)
            except:
                poses.append(np.zeros(12))

        return np.array(poses, dtype=np.float32)

    def __getitem__(self, index):
        item = self.pairs[index]
        
        bev_img = Image.open(item['bev']).convert('RGB')
        bev_tensor = BEV_TF(bev_img)
        
        range_img = Image.open(item['range']).convert('RGB')
        range_tensor = RANGE_TF(range_img)
        
        return bev_tensor, range_tensor, index

    def __len__(self):
        return len(self.pairs)


class FusionTrainingDataset(data.Dataset):
    def __init__(self, dataset_root='datasets/snail', seq='radar', max_frames=10000, cache_path=None):
        super().__init__()
        # 复用 InferDataset
        self.base_dataset = FusionInferDataset(seq, dataset_root, sample_inteval=1)
        
        if len(self.base_dataset) > max_frames:
             self.base_dataset.pairs = self.base_dataset.pairs[:max_frames]
             self.base_dataset.poses = self.base_dataset.poses[:max_frames]

        self.poses = self.base_dataset.poses
        
        self.pos_thres = 10 
        self.neg_thres = 50 
        self.num_neg = 5 
        self.positives, self.negatives = self._compute_pos_neg_samples()
        
        self.mining = False
        self.cache = None
        self.h5feat = None 

    def _compute_pos_neg_samples(self):
        positives = []
        negatives = []
        num_frames = len(self.poses)
        trans = self.poses[:, [3, 7, 11]]
        
        print("正在计算正负样本索引...")
        for i in range(num_frames):
            dist = np.linalg.norm(trans - trans[i], axis=1)
            pos = np.where((dist < self.pos_thres) & (dist > 0.01))[0]
            if len(pos) == 0: pos = np.array([i])
            positives.append(pos)
            neg = np.where(dist > self.neg_thres)[0]
            negatives.append(neg)
            
        return positives, negatives

    def refreshCache(self):
        if self.cache is None: return
        if not exists(self.cache): return
        with h5py.File(self.cache, 'r') as h5:
            self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
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

        q_bev, q_range, _ = self.base_dataset[index]
        p_bev, p_range, _ = self.base_dataset[pos_idx]
        
        n_bevs, n_ranges = [], []
        for ni in neg_indices:
            nb, nr, _ = self.base_dataset[ni]
            n_bevs.append(nb)
            n_ranges.append(nr)
            
        n_bevs = torch.stack(n_bevs)
        n_ranges = torch.stack(n_ranges)
        
        return (q_bev, q_range), (p_bev, p_range), (n_bevs, n_ranges), index

    def __len__(self):
        return len(self.poses)

# ================= 工具函数 =================

def fusion_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    
    queries, positives, negatives, indices = zip(*batch)
    
    q_bevs = torch.stack([x[0] for x in queries])
    q_ranges = torch.stack([x[1] for x in queries])
    
    p_bevs = torch.stack([x[0] for x in positives])
    p_ranges = torch.stack([x[1] for x in positives])
    
    n_bevs = torch.cat([x[0] for x in negatives])
    n_ranges = torch.cat([x[1] for x in negatives])
    
    all_bevs = torch.cat([q_bevs, p_bevs, n_bevs], dim=0)
    all_ranges = torch.cat([q_ranges, p_ranges, n_ranges], dim=0)
    
    return all_bevs, all_ranges, indices

# ================= 评估函数适配 =================

def evaluateFusionResults(dataset, model, device):
    # 这个函数暂时用不到了，因为 main.py 里已经手写了简单的评估逻辑
    # 但为了完整性保留，防止 ImportError
    pass