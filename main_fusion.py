import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import faiss 
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

# --- 导入模型 ---
from fusion_model import FusionPlaceModel
# --- 导入数据集 ---
import fusion_dataset as ds_module 

def get_args():
    parser = argparse.ArgumentParser(description='FusionPlace Dual-Sequence Test')
    
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    
    # 数据集路径
    parser.add_argument('--dataset_root', type=str, 
                        default='./datasets/snail', 
                        help='Snail 数据集根目录')
    
    # 序列设置 (根据你的截图填写的默认值)
    parser.add_argument('--val_db_seq', type=str, default="if_20231208_4", help='数据库序列 (旧路线)')
    parser.add_argument('--val_q_seq', type=str, default="if_20240116_5", help='查询序列 (新路线)')

    # 训练参数 (测试模式下主要用 batchSize)
    parser.add_argument('--batchSize', type=int, default=8, help='训练批量')
    parser.add_argument('--cacheBatchSize', type=int, default=8, help='推理批量')
    parser.add_argument('--nEpochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--threads', type=int, default=4, help='数据加载线程数 (报错Bus Error请设为0)')
    parser.add_argument('--seed', type=int, default=1024, help='随机种子')
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrStep', type=float, default=5)
    parser.add_argument('--lrGamma', type=float, default=0.5)
    parser.add_argument('--weightDecay', type=float, default=0.001)
    
    # 路径
    parser.add_argument('--runsPath', type=str, default='./runs/')
    parser.add_argument('--cachePath', type=str, default='./cache/fusion_v6')
    
    # 预训练权重
    parser.add_argument('--vggt_path', type=str, default='runs/vggt_model/vggtmodel.pt')
    parser.add_argument('--bev_path', type=str, default='runs/Aug08_10-17-29/model_best.pth.tar')
    
    parser.add_argument('--range_dim', type=int, default=2048)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--match_save_path', type=str, default='./runs/test_matches/')

    opt = parser.parse_args()
    return opt

def collate_fn_inference(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None 
    bevs, ranges, indices = zip(*batch)
    bevs = torch.stack(bevs, 0)
    ranges = torch.stack(ranges, 0)
    indices = list(indices)
    return (bevs, ranges), indices

def infer_fusion_data(eval_set, model, opt, device, desc_name="特征"):
    model.eval() 
    test_loader = DataLoader(
        dataset=eval_set, 
        num_workers=opt.threads,
        batch_size=opt.cacheBatchSize, 
        shuffle=False, 
        collate_fn=collate_fn_inference
    )
    all_global_descs = []
    with torch.no_grad():
        for data, indices in tqdm(test_loader, desc=f"提取{desc_name}"):
            if data is None: continue
            (bevs, ranges) = data
            bevs = bevs.to(device)
            ranges = ranges.to(device)
            res = model(bevs, ranges)
            if isinstance(res, tuple): res = res[-1]
            all_global_descs.append(res.detach().cpu().numpy())
    if not all_global_descs: return np.array([])
    return np.concatenate(all_global_descs, axis=0)

def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    if not exists(model_out_path): makedirs(model_out_path)
    filename = join(model_out_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(model_out_path, 'model_best.pth.tar'))

# ==============================================================================
#  主程序
# ==============================================================================
if __name__ == "__main__":
    opt = get_args()
    print('====> FusionPlace (Dual-Sequence Test Mode) ====')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===> 使用设备: {device}")
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    # 1. 加载模型
    print('===> 加载 FusionPlaceModel')
    model = FusionPlaceModel(
        vggt_path=opt.vggt_path, 
        bev_path=opt.bev_path, 
        range_dim=opt.range_dim
    )
    model = model.to(device)

    # 2. 测试模式 (双序列对比)
    if opt.mode.lower() == 'test':
        print(f'===> 进入双序列测试模式')
        print(f"    - Database (库): {opt.val_db_seq}")
        print(f"    - Query (查):    {opt.val_q_seq}")
        
        if not exists(opt.match_save_path): makedirs(opt.match_save_path)

        # A. 加载数据集
        print("\n[1/4] 加载数据集...")
        try:
            db_set = ds_module.FusionInferDataset(seq=opt.val_db_seq, dataset_root=opt.dataset_root)
            q_set = ds_module.FusionInferDataset(seq=opt.val_q_seq, dataset_root=opt.dataset_root)
        except FileNotFoundError as e:
            print(f"❌ 数据集加载失败: {e}")
            print("请检查 dataset_root 和 seq 名称是否与截图中的文件夹一致。")
            exit()

        print(f"    - DB 帧数: {len(db_set)}")
        print(f"    - Query 帧数: {len(q_set)}")
        
        # B. 提取特征
        print("\n[2/4] 提取特征...")
        db_feats = infer_fusion_data(db_set, model, opt, device, "DB特征")
        q_feats = infer_fusion_data(q_set, model, opt, device, "Query特征")
        
        # C. 建立索引
        print("\n[3/4] 建立 FAISS 索引 (DB)...")
        index = faiss.IndexFlatL2(db_feats.shape[1])
        index.add(db_feats)
        
        # D. 检索与评估
        print("\n[4/4] 检索与评估 (Top-1)...")
        D, I = index.search(q_feats, 1) # 只取最近的1个
        
        # 评估参数
        recall_radius = 25.0   # 判定成功的距离阈值 (米)
        
        correct = 0
        valid_queries = 0
        
        # 获取 Pose (x, y, z) -> indices [3, 7, 11]
        # 注意：这里假设 Pose 文件格式是 3x4 矩阵展平或 12个float
        # 如果 Pose 不对齐，计算出的距离会很大
        db_poses = db_set.poses[:, [3, 7, 11]]
        q_poses = q_set.poses[:, [3, 7, 11]]
        
        # 使用 KDTree 快速计算 Query 是否有对应的真值 (Ground Truth)
        # 也就是：Query 这一帧的地方，DB 到底有没有走过？没走过就不算数。
        from scipy.spatial import KDTree
        db_tree = KDTree(db_poses)
        
        for i in range(len(q_feats)):
            query_pose = q_poses[i]
            
            # --- 步骤 1: 检查是否存在真值 (Ground Truth) ---
            # 在 DB 中查找距离 Query 最近的点
            dist_to_nearest_db, _ = db_tree.query(query_pose, k=1)
            
            # 如果 Query 所在的位置，DB 里压根没去过 (距离 > 25m)，那这一帧没法评测，跳过
            if dist_to_nearest_db > recall_radius:
                continue
                
            valid_queries += 1
            
            # --- 步骤 2: 检查模型预测是否正确 ---
            pred_db_idx = I[i][0]
            pred_pose = db_poses[pred_db_idx]
            
            error_dist = np.linalg.norm(query_pose - pred_pose)
            
            if error_dist < recall_radius:
                correct += 1
                
        # E. 输出结果
        print("\n" + "="*40)
        if valid_queries > 0:
            recall = correct / valid_queries
            print(f"✅ Recall@1: {recall:.4f} ({(recall*100):.2f}%)")
            print(f"📊 统计详情:")
            print(f"   - 总 Query 帧数: {len(q_feats)}")
            print(f"   - 有效重叠帧数 (分母): {valid_queries}")
            print(f"   - 预测正确次数 (分子): {correct}")
        else:
            print("❌ 两个序列的物理路径似乎没有重叠！(最小距离 > 25m)")
            print("请检查两个序列的 GPS/Pose 是否在同一个坐标系下。")
        print("="*40)

    elif opt.mode.lower() == 'train':
        print("请使用上一版代码进行训练，此脚本专用于双序列测试。")