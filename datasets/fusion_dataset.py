import os
from os.path import join, exists, splitext
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import h5py
import faiss
from torchvision import transforms
import cv2 
import torchvision.transforms.functional as TF
import random

# ================= 配置区域 =================

def get_transforms():
    bev_tf = transforms.Compose([
        transforms.Resize((256, 256)),  # <-- 核心修改：强制缩放到 256x256
        # 注意：这里不需要 ToTensor 或 Normalize，因为我们在 __getitem__ 里已经手动做了
    ])
    
    
    # Range 图保持不变
    range_tf = transforms.Compose([
        transforms.Resize((70, 518)),# interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return bev_tf, range_tf 


# 在 get_transforms 中添加 normalize 定义
# def get_transforms():
#     # 定义通用的 normalize
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                      std=[0.229, 0.224, 0.225])

#     bev_tf = transforms.Compose([
#         transforms.Resize((256, 256)),
#         # 注意：transforms.Normalize 期望输入是 Tensor，所以放在 Resize 后
#         normalize 
#     ])

#     range_tf = transforms.Compose([
#         transforms.Resize((70, 518)),
#         transforms.ToTensor(),
#         normalize
#     ])
#     return bev_tf, range_tf

BEV_TF, RANGE_TF = get_transforms()

def extract_timestamp(filename):
    """从文件名提取时间戳 (兼容新旧格式)
    
    支持格式：
    - 新格式: 1702473515_979433 -> 1702473515.979433
    - 旧格式: 1702473515979433 -> 1702473515.979433 (自动拆分)
    - 其他: 直接转换为浮点数
    """
    base_name = splitext(os.path.basename(filename))[0]
    try:
        # 处理 seconds_microseconds 格式（如 1702473515_979433）
        if '_' in base_name:
            parts = base_name.split('_')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                # 转换为浮点数：秒.微秒
                return float(f"{parts[0]}.{parts[1]}")
        
        # 处理纯数字格式（如 1702473515979433，16位微秒时间戳）
        if base_name.isdigit() and len(base_name) >= 15:
            # 自动拆分：前面是秒，后6位是微秒
            seconds = base_name[:-6]
            microseconds = base_name[-6:]
            return float(f"{seconds}.{microseconds}")
        
        # 尝试直接转换
        return float(base_name)
    except Exception:
        import re
        nums = re.findall(r"\d+\.?\d*", base_name)
        if nums:
            return float(nums[0])
        return 0.0  # 失败返回0

# ================= 核心数据集类 =================

class FusionInferDataset(data.Dataset):
    def __init__(self, seq, dataset_root='datasets/snail', sample_inteval=1):
        super().__init__()
        self.sample_inteval = sample_inteval
        self.seq = seq
        
        # [修改 1] 适配你的截图路径结构
        # 优先直接找 dataset_root/seq
        # dataset_root = "./datasets/snail_old"
        # seq = "if_20231208_4"
        base_dir = join(dataset_root, seq)
        
        # 兼容性：如果找不到，再试试 radar/seq
        if not exists(base_dir):
            base_dir = join(dataset_root, 'radar', seq)

        if not exists(base_dir):
            raise FileNotFoundError(f"找不到序列目录: {base_dir} (请检查 dataset_root 是否指向 snail_old)")

        self.bev_dir = join(base_dir, 'bev_image') 
        self.range_dir = join(base_dir, 'range_image') 
        self.pose_dir = join(base_dir, 'poses')

        if not exists(self.bev_dir) or not exists(self.range_dir):
            raise FileNotFoundError(f"数据子目录缺失:\nBEV: {self.bev_dir}\nRange: {self.range_dir}")

        # 读取文件
        self.bev_files = sorted([f for f in os.listdir(self.bev_dir) if f.endswith('.png')])
        self.range_files = sorted([f for f in os.listdir(self.range_dir) if f.endswith('.png')])
        
        print(f"🔍 [DEBUG] 原始文件统计: BEV={len(self.bev_files)}, Range={len(self.range_files)}")

        # 构建 Range 的时间戳索引
        range_db = []
        for f in self.range_files:
            range_db.append({'ts': extract_timestamp(f), 'name': f})
        # 按时间戳排序，方便二分查找
        range_db.sort(key=lambda x: x['ts'])
        range_timestamps = np.array([x['ts'] for x in range_db])
        
        # [修改 2] 使用最近邻匹配 (Nearest Neighbor Matching)
        self.pairs = []
        match_count = 0
        
        for bev_f in self.bev_files:
            bev_ts = extract_timestamp(bev_f)
            
            # 在 Range 时间戳中找到插入位置
            idx = np.searchsorted(range_timestamps, bev_ts)
            
            # 检查前后两个哪个更近
            candidates = []
            if idx < len(range_timestamps): candidates.append(idx)
            if idx > 0: candidates.append(idx - 1)
            
            if not candidates: continue
            
            # 找到时间差最小的那个
            best_idx = min(candidates, key=lambda i: abs(range_timestamps[i] - bev_ts))
            closest_ts = range_timestamps[best_idx]
            time_diff = abs(closest_ts - bev_ts)
            
            # 容忍阈值：0.1秒 (100ms)
            if time_diff < 0.1:
                range_f = range_db[best_idx]['name']
                self.pairs.append({
                    'bev': join(self.bev_dir, bev_f),
                    'range': join(self.range_dir, range_f),
                    'ts': bev_ts
                })
                match_count += 1
        
        print(f"✅ [DEBUG] 模糊匹配成功: {match_count} 对 (阈值=0.1s)")
        
        if not self.pairs:
            raise ValueError(f"匹配失败！即使使用了模糊匹配也没找到对应帧。请检查时间戳格式是否一致。")
        
        # 排序并采样
        self.pairs.sort(key=lambda x: x['ts'])
        self.pairs = self.pairs[::self.sample_inteval]
        
        # 加载 Pose (逻辑不变)
        self.poses = self._load_poses()

    def _load_poses(self):
        poses = []
        if not exists(self.pose_dir):
            return np.zeros((len(self.pairs), 12), dtype=np.float32)

        pose_files = [f for f in os.listdir(self.pose_dir) if 'txt' in f]
        pose_map = {extract_timestamp(f): join(self.pose_dir, f) for f in pose_files}
        sorted_pose_ts = sorted(list(pose_map.keys()))
        
        if not sorted_pose_ts:
             return np.zeros((len(self.pairs), 12), dtype=np.float32)

        sorted_pose_ts_np = np.array(sorted_pose_ts)
        
        for p in self.pairs:
            img_ts = p['ts']
            # 同样使用最近邻查找 Pose
            idx = np.argmin(np.abs(sorted_pose_ts_np - img_ts))
            closest_ts = sorted_pose_ts_np[idx]
            
            # 也就是如果 Pose 也没对齐，这里会找最近的
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
            # 1. 获取数据项 (注意：FusionDataset 使用 self.pairs 字典列表)
            item = self.pairs[index]
            
            # 2. 获取 BEV 图片路径
            bev_path = item['bev']
            
            # 3. 读取彩色 BEV 图片 (3通道)
            # 使用 cv2.IMREAD_COLOR 确保读入 R, G, B 信息
            img = cv2.imread(bev_path, cv2.IMREAD_COLOR) 
            #img = np.clip(img.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
            
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {bev_path}")

            # 4. BGR 转 RGB (OpenCV 默认是 BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 5. 归一化 (0-255 -> 0.0-1.0)
            img = img.astype(np.float32) / 255.0
            
            # 6. 调整维度: [H, W, 3] -> [3, H, W]
            # 这样就得到了 3 通道的 BEV Tensor (强度, 高度, 多普勒)
            bev_tensor = torch.from_numpy(img).permute(2, 0, 1)
            
            # # 确保使用了文件头部的全局变量 BEV_TF
            if BEV_TF is not None:
                bev_tensor = BEV_TF(bev_tensor)
            
            # =======================================================
            # 7. 处理 Range 图 (必须保留！否则 infer_fusion 会解包失败)
            # =======================================================
            range_img = Image.open(item['range']).convert('RGB')
            range_tensor = RANGE_TF(range_img)
            
            return bev_tensor, range_tensor, index

    def __len__(self):
        return len(self.pairs)


class FusionTrainingDataset(data.Dataset):
    def __init__(self, dataset_root='datasets/snail', seq='radar', max_frames=10000, cache_path=None, sample_inteval=1):
        super().__init__()
        # 复用上面的 InferDataset
        self.base_dataset = FusionInferDataset(seq, dataset_root, sample_inteval=sample_inteval)
        
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

    ## 无旋转版本
    # def __getitem__(self, index):
    #     if self.mining and self.h5feat is not None:
    #         q_feat = self.h5feat[index]
    #         pos_idx = np.random.choice(self.positives[index])
            
    #         neg_candidates = self.negatives[index]
    #         if len(neg_candidates) > 0:
    #             subset = np.random.choice(neg_candidates, min(len(neg_candidates), 1000))
    #             neg_feats = self.h5feat[subset]
    #             feat_dists = np.linalg.norm(neg_feats - q_feat, axis=1)
    #             hardest_idx = np.argsort(feat_dists)[:self.num_neg]
    #             neg_indices = subset[hardest_idx]
    #         else:
    #             neg_indices = np.random.choice(range(len(self.poses)), self.num_neg)
    #     else:
    #         pos_idx = np.random.choice(self.positives[index])
    #         neg_indices = np.random.choice(self.negatives[index], self.num_neg)

    #     q_bev, q_range, _ = self.base_dataset[index]
    #     p_bev, p_range, _ = self.base_dataset[pos_idx]
        
    #     n_bevs, n_ranges = [], []
    #     for ni in neg_indices:
    #         nb, nr, _ = self.base_dataset[ni]
    #         n_bevs.append(nb)
    #         n_ranges.append(nr)
            
    #     n_bevs = torch.stack(n_bevs)
    #     n_ranges = torch.stack(n_ranges)
        
    #     return (q_bev, q_range), (p_bev, p_range), (n_bevs, n_ranges), index
    
    
    
    ## 旋转版本
    def __getitem__(self, index):
        # 1. 确定正负样本索引 (保持不变)
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

        # =========================================================
        # 2. 读取数据并加入 [随机旋转增强]
        # =========================================================
        
        # 定义一个辅助函数：同步处理 BEV 和 Range
        def load_and_augment(idx):
            # A. 读取基础数据
            bev, range_tensor, _ = self.base_dataset[idx]
            
            # B. 生成随机角度 (0 ~ 360)
            angle = random.uniform(-90, 90)
            
            # C. 旋转 BEV (几何旋转)
            # BEV 是 [3, H, W]
            bev_rotated = TF.rotate(bev, angle)
            
            # D. 平移 Range (循环滚动)
            # Range 是 [C, H, W]，宽 W 对应 360度
            # 计算需要移动多少个像素
            c, h, w = range_tensor.shape
            # shift_ratio = angle / 360.0
            # # 注意方向：通常 BEV 逆时针转，全景图需要向某一侧滚动
            # # 这里假设顺时针滚动，具体正负可能需要根据雷达厂商定义微调，通常负号对齐
            # pixel_shift = int(w * shift_ratio)
            
            # # 使用 torch.roll 实现循环移位
            # # dims=-1 表示在宽度方向 (W) 滚动
            # range_shifted = torch.roll(range_tensor, shifts=int(w * shift_ratio), dims=-1)
            
            return bev_rotated, range_tensor

        # --- 处理 Query ---
        q_bev, q_range = load_and_augment(index)

        # --- 处理 Positive ---
        p_bev, p_range = load_and_augment(pos_idx)

        # --- 处理 Negatives ---
        n_bevs, n_ranges = [], []
        for ni in neg_indices:
            nb, nr = load_and_augment(ni)
            n_bevs.append(nb)
            n_ranges.append(nr)
            
        n_bevs = torch.stack(n_bevs)
        n_ranges = torch.stack(n_ranges)
        
        # 返回打包好的数据
        return (q_bev, q_range), (p_bev, p_range), (n_bevs, n_ranges), index
    
    

    def __len__(self):
        return len(self.poses)

# ================= 工具函数 =================

# def fusion_collate_fn(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     if len(batch) == 0: return None
    
#     queries, positives, negatives, indices = zip(*batch)
    
#     q_bevs = torch.stack([x[0] for x in queries])
#     q_ranges = torch.stack([x[1] for x in queries])
    
#     p_bevs = torch.stack([x[0] for x in positives])
#     p_ranges = torch.stack([x[1] for x in positives])
    
#     n_bevs = torch.cat([x[0] for x in negatives])
#     n_ranges = torch.cat([x[1] for x in negatives])
    
#     all_bevs = torch.cat([q_bevs, p_bevs, n_bevs], dim=0)
#     all_ranges = torch.cat([q_ranges, p_ranges, n_ranges], dim=0)
    
#     return all_bevs, all_ranges, indices

def collate_fn_inference(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None
    
    data_tuple = list(zip(*batch))
    
    if len(data_tuple) == 3:
        bevs, ranges, indices = data_tuple
        bevs = torch.stack(bevs, 0)
        ranges = torch.stack(ranges, 0)
        return (bevs, ranges), list(indices)
        
    elif len(data_tuple) == 2:
        bevs, indices = data_tuple
        bevs = torch.stack(bevs, 0)
        return bevs, list(indices)
    else:
        raise ValueError(f"collate_fn_inference 收到异常数据长度: {len(data_tuple)}")
    
# ==========================================
#  请将此函数添加到 bev_dataset.py 的最末尾
#  (注意缩进，它应该是一个顶层函数，不属于任何类)
# ==========================================

def collate_fn(batch):
    """
    处理 FusionTrainingDataset 的 batch 数据
    将 [(q, p, n, idx), ...] 这种列表结构
    打包成 ((q_bev, q_range), (p_bev, p_range), (n_bevs, n_ranges), indices)
    """
    # 1. 过滤掉读取失败的数据 (None)
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None
    
    # 2. 解包 batch 中的四项 (Query, Positive, Negatives, Index)
    # dataset.__getitem__ 返回的是: (q_bev, q_range), (p_bev, p_range), (n_bevs, n_ranges), index
    queries, positives, negatives, indices = zip(*batch)
    
    # 3. 堆叠 Query (BEV, Range)
    q_bevs = torch.stack([x[0] for x in queries])
    q_ranges = torch.stack([x[1] for x in queries])
    
    # 4. 堆叠 Positive (BEV, Range)
    p_bevs = torch.stack([x[0] for x in positives])
    p_ranges = torch.stack([x[1] for x in positives])
    
    # 5. 堆叠 Negatives
    # 注意：每个样本有多个负样本，所以这里用 cat 把它们串起来
    n_bevs = torch.cat([x[0] for x in negatives])
    n_ranges = torch.cat([x[1] for x in negatives])
    
    # 6. 返回打包好的 4 个元素，对应 train_epoch 中的解包顺序
    return (q_bevs, q_ranges), (p_bevs, p_ranges), (n_bevs, n_ranges), list(indices)

# ================= 评估函数 =================
def evaluateResults(seq, global_descs, local_feats, dataset, match_results_save_path=None):
    is_list_style = isinstance(global_descs, (list, tuple))
    if is_list_style:
        if len(global_descs) != 2:
            raise ValueError("当以 list/tuple 形式传入 global_descs 时，期望格式为 [db_descs, q_descs]")
        db_descs = np.asarray(global_descs[0]).astype(np.float32)
        q_descs = np.asarray(global_descs[1]).astype(np.float32)
        desc_dim = db_descs.shape[1]
    else:
        global_descs = np.asarray(global_descs).astype(np.float32)
        desc_dim = global_descs.shape[1]
        db_idx = dataset.db_split_index
        db_descs = global_descs[:db_idx].astype(np.float32)
        q_descs = global_descs[db_idx:].astype(np.float32)

    gt_thres = 25
    faiss_index = faiss.IndexFlatL2(desc_dim)
    all_positives = 0
    tp = 0
    all_errs = []

    if is_list_style:
            faiss_index.add(db_descs)
            _, predictions = faiss_index.search(q_descs, 1)
            
            print("\n" + "="*40)
            print("===== 原始匹配深度分析 (前 10 帧) =====")
            
            # 提取 DB 的位姿子集供参考
            db_poses_subset = dataset.poses[:dataset.db_split_index]
            
            for i in range(min(10, len(predictions))):
                q_idx = i
                pred_db_idx = predictions[q_idx][0]
                
                # 【关键修正】从 dataset.poses 中正确索引 Query 位姿
                # Query 的真实索引 = 当前索引 + 数据库长度
                q_global_idx = q_idx + dataset.db_split_index
                q_pose = dataset.poses[q_global_idx, [3, 7, 11]]
                
                # 从数据库位姿子集中提取匹配到的位姿
                db_pose = db_poses_subset[pred_db_idx, [3, 7, 11]]
                
                # 计算物理距离
                dist = np.sqrt(np.sum((q_pose - db_pose)**2))
                
                print(f"Query {q_idx} (全局Idx:{q_global_idx}) -> 匹配到 DB {pred_db_idx}")
                print(f"  > 实际物理距离: {dist:.4f} 米")
            print("="*40 + "\n")
    else:
        raise NotImplementedError("非list/tuple模式")

    print("\n===== 检索详细信息 =====")
    target_q_idx = 20

    for q_idx, pred in enumerate(predictions):
        query_idx = q_idx + dataset.db_split_index
        query_pose = dataset.poses[query_idx]
        if not hasattr(dataset, 'db_split_index'):
            raise AttributeError("数据集对象需包含 db_split_index 属性")
        db_poses = dataset.poses[:dataset.db_split_index]

        gt_dis = (query_pose - db_poses) ** 2
        positives = np.where(np.sum(gt_dis[:, [3, 7, 11]], axis=1) < gt_thres ** 2)[0]

        if len(positives) > 0:
            all_positives += 1
            if pred[0] in positives:
                tp += 1

        if q_idx == target_q_idx:
            q_x = query_pose[3]
            q_y = query_pose[7]
            q_z = query_pose[11]
            print(f"\nQuery {q_idx} - 全局索引: {query_idx} | 位姿(x,y,z): ({q_x:.2f}, {q_y:.2f}, {q_z:.2f})")
            print(f"前5个DB样本与Query的距离：")
            for db_i in range(min(5, len(db_poses))):
                db_pose = db_poses[db_i]
                dx = q_x - db_pose[3]
                dy = q_y - db_pose[7]
                dz = q_z - db_pose[11]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                print(f"DB {db_i} - 位姿(x,y,z): ({db_pose[3]:.2f}, {db_pose[7]:.2f}, {db_pose[11]:.2f}) | 距离: {dist:.2f}m")
            print(f"Query {q_idx} - 正样本数量（距离<{gt_thres}m）: {len(positives)}")
            print(f"Query {q_idx} - FAISS检索到的DB索引: {pred[0]}")
            is_tp = pred[0] in positives if len(positives) > 0 else False
            print(f"Query {q_idx} - 检索结果是否为正样本: {is_tp}")

    recall_top1 = tp / all_positives if all_positives > 0 else 0.0
    # print(f"前 10 个 Query 匹配到的 DB 索引分别是: {predictions[:500].flatten()}")
    print(f"\n===== 评估结果 =====")
    print(f"Recall@1: {recall_top1:.4f} ({recall_top1*100:.2f}%)")

    return recall_top1, 0.0, 0.0, 0.0