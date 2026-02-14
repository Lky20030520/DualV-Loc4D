import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py
import faiss 
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

# --- 导入模型 ---
from model.fusion_model_salad import FusionPlaceModel

# --- 导入数据集 ---
import datasets.fusion_dataset as ds_module 

def get_args():
    parser = argparse.ArgumentParser(description='FusionPlace Integrated (Training & Test)')
    
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    parser.add_argument('--stage', type=str, default='B', help='Training stage', choices=['A', 'B'])
    parser.add_argument('--dataset_root', type=str, default='./datasets/snail', help='Snail 数据集根目录')
    
    # 序列设置
    parser.add_argument('--train_seq', type=str, default='if_20231208_4', help='训练序列')
    parser.add_argument('--val_db_seq', type=str, default='if_20231208_4', help='验证数据库')
    parser.add_argument('--val_q_seq', type=str, default='if_20240116_5', help='验证查询')

    # 模型参数
    parser.add_argument('--vggt_path', type=str, default='')
    parser.add_argument('--bev_path', type=str, default='')
    parser.add_argument('--load_from', type=str, default='', help='恢复训练或测试的模型路径')
    parser.add_argument('--cachePath', type=str, default='./cache/fusion_integrated/')
    parser.add_argument('--match_save_path', type=str, default='./fusion_match_results/')
    parser.add_argument('--runsPath', type=str, default='./runs/')
    parser.add_argument('--sample_interval', type=int, default=30)
    parser.add_argument('--range_dim', type=int, default=2048)
    
    # 训练参数
    parser.add_argument('--batchSize', type=int, default=1, help='训练批量') 
    parser.add_argument('--cacheBatchSize', type=int, default=4, help='缓存/推理批量')
    parser.add_argument('--nEpochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lrStep', type=float, default=5)
    parser.add_argument('--lrGamma', type=float, default=0.5)
    parser.add_argument('--weightDecay', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=0) 
    parser.add_argument('--seed', type=int, default=1024)

    opt = parser.parse_args()
    return opt

class TripletLoss(nn.Module):
    """三元组损失"""
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss

def train_epoch(epoch, model, train_set, opt, device, writer, optimizer):
    epoch_loss = 0
    if not exists(opt.cachePath): makedirs(opt.cachePath)

    # === Hard Mining Cache 构建 ===
    if epoch >= 5: 
        print(f'====> Epoch {epoch}: 构建硬样本挖掘特征缓存')
        train_set.mining = False 
        train_set.cache = join(opt.cachePath, 'desc_cen_hardmining.hdf5')
        
        with h5py.File(train_set.cache, mode='w') as h5:
            h5feat = None 
            
            # DataLoader
            train_loader = DataLoader(
                dataset=train_set.base_dataset, 
                num_workers=opt.threads, 
                batch_size=opt.cacheBatchSize, 
                shuffle=False, 
                collate_fn=ds_module.collate_fn_inference
            )
            
            with torch.no_grad():
                model.eval()
                for iteration, (data, indices) in enumerate(tqdm(train_loader, desc="Caching"), 1):
                    if isinstance(data, (tuple, list)):
                        bevs, ranges = data 

                    bevs = bevs.to(device)
                    ranges = ranges.to(device)

                    # Stage A: 仅使用 BEV；Stage B: 使用 BEV + Range
                    if opt.stage == 'A':
                        res = model.forward_bev_only(bevs)
                    else:
                        res = model(bevs, ranges)
                    
                    # 兼容性处理
                    if isinstance(res, tuple): res = res[-1]
                    
                    res_np = res.detach().cpu().numpy()

                    # 延迟初始化 HDF5
                    if h5feat is None:
                        real_dim = res_np.shape[1] 
                        print(f"\n🔍 [Auto-Detect] 检测到特征维度: {real_dim} (以此创建HDF5)")
                        h5feat = h5.create_dataset("features", [len(train_set), real_dim], dtype=np.float32)
                    
                    indices_list = list(indices)
                    h5feat[indices_list, :] = res_np

        train_set.mining = True
        train_set.refreshCache()

    # === 正式训练 ===
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                              batch_size=opt.batchSize, shuffle=True, 
                              collate_fn=ds_module.collate_fn)
    model.train()
    criterion = TripletLoss().to(device)

    for iteration, (query, positives, negatives, indices) in enumerate(train_loader):
        if query is None: continue 
        
        q_bev, q_range = query
        p_bev, p_range = positives
        n_bev, n_range = negatives 

        B = q_bev.shape[0]
        
        input_bevs = torch.cat([q_bev, p_bev, n_bev]).to(device)
        input_ranges = torch.cat([q_range, p_range, n_range]).to(device)

        # Stage A: 仅使用 BEV；Stage B: 使用 BEV + Range
        if opt.stage == 'A':
            global_descs = model.forward_bev_only(input_bevs)
        else:
            global_descs = model(input_bevs, input_ranges)
        # 如果模型返回了 tuple，取最后一个 (final_desc)
        if isinstance(global_descs, tuple): global_descs = global_descs[-1]

        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, n_bev.shape[0]])

        optimizer.zero_grad()
        loss = 0.0
        
        num_negs_per_q = n_bev.shape[0] // B
        for i in range(B):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], 
                                           global_descs_N[num_negs_per_q*i : num_negs_per_q*(i+1)]))
            loss += max_loss
        loss /= B
        loss.backward()
        
        optimizer.step()

        epoch_loss += loss.item()
        if iteration % 10 == 0:
            print(f"Epoch[{epoch}]({iteration}): Loss: {loss.item():.4f}")
            writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(train_loader) + iteration)

def infer_fusion(eval_set, model, opt, device):
    """ 推理函数 """
    model.eval()
    loader = DataLoader(eval_set, batch_size=opt.cacheBatchSize, shuffle=False, 
                        num_workers=opt.threads, collate_fn=ds_module.collate_fn_inference)
    
    all_global_descs = []
    with torch.no_grad():
        for data, indices in tqdm(loader, desc="Inference"):
            (bevs, ranges) = data 
            bevs = bevs.to(device)
            
            # Stage A: 仅使用 BEV；Stage B: 使用 BEV + Range
            if opt.stage == 'A':
                res = model.forward_bev_only(bevs)
            else:
                ranges = ranges.to(device)
                res = model(bevs, ranges)
            if isinstance(res, tuple): res = res[-1]
            
            all_global_descs.append(res.detach().cpu().numpy())
            
    return np.concatenate(all_global_descs, axis=0)

def saveCheckpoint(state, is_best, path):
    if not exists(path): makedirs(path)
    filename = join(path, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(path, 'model_best.pth.tar'))

if __name__ == "__main__":
    opt = get_args()
    print('====> FusionPlace Integrated (SALAD Version)')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not exists(opt.cachePath): 
        makedirs(opt.cachePath)
    
    print(f"====> Cache Directory: {opt.cachePath}")
    
    # 1. 初始化模型
    print(f'===> 加载 FusionPlaceModel (Stage={opt.stage}, with SALAD)')
    model = FusionPlaceModel(
        vggt_path=opt.vggt_path, 
        bev_path=opt.bev_path, 
        range_dim=opt.range_dim,
        freeze_backbones=True,
        stage=opt.stage  # 传入 stage 参数
    )
    model = model.to(device)
    
    # ==============================================================================
    # 2. 训练/测试流程 (已移除 NetVLAD 聚类初始化)
    # ==============================================================================
    if opt.mode == 'train':
        log_dir = join(opt.runsPath, f"fusion_{datetime.now().strftime('%b%d_%H-%M-%S')}")
        writer = SummaryWriter(log_dir=log_dir)

        # 如果指定了 load_from，加载权重；否则 SALAD 使用默认随机初始化
        if opt.load_from and isfile(opt.load_from):
             print(f"Loading checkpoint: {opt.load_from}")
             checkpoint = torch.load(opt.load_from, map_location=device, weights_only=False)
             model.load_state_dict(checkpoint['state_dict'])
        else:
             print("🚀 模型使用默认初始化 (SALAD 不需要 K-Means 聚类)")

        # 加载训练集
        train_set = ds_module.FusionTrainingDataset(
            dataset_root=opt.dataset_root, 
            seq=opt.train_seq,
            sample_inteval=opt.sample_interval
        )
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        
        best_recall = 0.0
        
        for epoch in range(opt.nEpochs):
            # 训练
            train_epoch(epoch, model, train_set, opt, device, writer, optimizer)
            scheduler.step()
            
            # 验证
            db_set = ds_module.FusionInferDataset(seq=opt.val_db_seq, dataset_root=opt.dataset_root, sample_inteval=opt.sample_interval)
            q_set = ds_module.FusionInferDataset(seq=opt.val_q_seq, dataset_root=opt.dataset_root, sample_inteval=opt.sample_interval)
            
            db_feats = infer_fusion(db_set, model, opt, device)
            q_feats = infer_fusion(q_set, model, opt, device)
            
            # 构造 Wrapper 进行评估
            class _Wrapper: pass
            wrapper = _Wrapper()
            wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
            wrapper.db_split_index = len(db_set.poses)
            wrapper.sample_inteval = opt.sample_interval
            
            recall, _, _, _ = ds_module.evaluateResults(
                seq=f"Epoch_{epoch}",
                global_descs=[db_feats, q_feats],
                local_feats=None,
                dataset=wrapper
            )
            
            print(f"===> Epoch {epoch} Val Recall@1: {recall:.4f}")
            writer.add_scalar('Val/Recall', recall, epoch)
            
            is_best = recall > best_recall
            if is_best: best_recall = recall
            
            saveCheckpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_recall': best_recall,
                'optimizer': optimizer.state_dict()
            }, is_best, log_dir)

    elif opt.mode == 'test':
        print('===> 进入测试模式')
        if opt.load_from and isfile(opt.load_from):
            print(f"Loading checkpoint for testing: {opt.load_from}")
            checkpoint = torch.load(opt.load_from, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("⚠️ 警告：测试模式下没有指定 --load_from，模型将使用初始权重！")
        
        db_set = ds_module.FusionInferDataset(seq=opt.val_db_seq, dataset_root=opt.dataset_root, sample_inteval=opt.sample_interval)
        q_set = ds_module.FusionInferDataset(seq=opt.val_q_seq, dataset_root=opt.dataset_root, sample_inteval=opt.sample_interval)
        
        print("提取特征...")
        db_feats = infer_fusion(db_set, model, opt, device)
        q_feats = infer_fusion(q_set, model, opt, device)
        
        print("开始评估...")
        class _Wrapper: pass
        wrapper = _Wrapper()
        wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
        wrapper.db_split_index = len(db_set.poses)
        wrapper.sample_inteval = opt.sample_interval
        
        recall, _, _, _ = ds_module.evaluateResults(
            seq=f"{opt.val_db_seq}+{opt.val_q_seq}",
            global_descs=[db_feats, q_feats],
            local_feats=None,
            dataset=wrapper,
            match_results_save_path=opt.match_save_path
        )
        print(f"✅ Final Recall@1: {recall:.4f}")