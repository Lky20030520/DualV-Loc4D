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

# --- 导入模型 (保留代码1的模型) ---
from bev_model import FusionPlaceModel

# --- 导入数据集 (使用代码1的数据集模块，但可能需要适配代码2的接口) ---
import bev_dataset as ds_module 
# 注意：确保 fusion_dataset 中包含 evaluateResults 函数。
# 如果 fusion_dataset 没有 evaluateResults，你需要从 bevdata_dataset 复制该函数过去，
# 或者在这里 import bevdata_dataset 并借用它的评估函数。

def get_args():
    parser = argparse.ArgumentParser(description='FusionPlace Integrated (Training & Test)')
    
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    parser.add_argument('--dataset_root', type=str, default='./datasets/snail', help='Snail 数据集根目录')
    
    # 序列设置
    # 注意：这里假设 fusion_dataset 内部定义了 train_seq 列表，或者你可以手动指定
    parser.add_argument('--train_seq', type=str, default='if_20231208_4', help='训练序列')
    parser.add_argument('--val_db_seq', type=str, default='if_20231208_4', help='验证数据库')
    parser.add_argument('--val_q_seq', type=str, default='if_20240116_5', help='验证查询')

    # 模型参数 (保留代码1的设置)
    parser.add_argument('--vggt_path', type=str, default='runs/vggt_model/vggtmodel.pt')
    parser.add_argument('--bev_path', type=str, default='runs/bevdata_Nov06_16-29-10/model_best.pth.tar')
    parser.add_argument('--range_dim', type=int, default=2048)
    
    # 训练参数 (来自代码2)
    parser.add_argument('--batchSize', type=int, default=4, help='训练批量') # Fusion模型显存占用大，建议改小
    parser.add_argument('--cacheBatchSize', type=int, default=8, help='缓存/推理批量')
    parser.add_argument('--nEpochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrStep', type=float, default=5)
    parser.add_argument('--lrGamma', type=float, default=0.5)
    parser.add_argument('--weightDecay', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)
    
    # 路径参数
    parser.add_argument('--runsPath', type=str, default='./runs/')
    parser.add_argument('--cachePath', type=str, default='./cache/fusion_integrated/')
    parser.add_argument('--match_save_path', type=str, default='./fusion_match_results/')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--load_from', type=str, default='', help='恢复训练或测试的模型路径')

    opt = parser.parse_args()
    return opt

class TripletLoss(nn.Module):
    """三元组损失 (来自代码2)"""
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss

def collate_fn_wrapper(batch):
    """
    为了兼容 FusionDataset 的输出格式 (bevs, ranges, indices) 
    我们需要确保它能适配代码2风格的训练循环
    """
    # 假设 fusion_dataset.collate_fn 已经存在，如果不存在，使用代码1的逻辑
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None
    
    # 训练模式下，Dataset通常返回 (query, pos, neg, indices)
    # 如果 FusionDataset 尚未实现 Hard Mining 的输出格式，这里可能需要调整
    # 这里假设我们正在使用类似 bevdata_dataset 的 TrainingDataset 结构
    return ds_module.collate_fn(batch) 

def train_epoch(epoch, model, train_set, opt, device, writer, optimizer):
    epoch_loss = 0
    # 确保 cachePath 存在
    if not exists(opt.cachePath): makedirs(opt.cachePath)

    # === Hard Mining Cache 构建 (来自代码2) ===
    if epoch >= 5: # 可以设置晚一点开启 Hard Mining
        print(f'====> Epoch {epoch}: 构建硬样本挖掘特征缓存')
        train_set.mining = False 
        train_set.cache = join(opt.cachePath, 'desc_cen_hardmining.hdf5')
        
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = 256 * 2 # 假设 Fusion 输出维度，需要根据模型调整
            # 临时获取维度
            dummy_dim = model.bev_model.global_feat_dim if hasattr(model, 'bev_model') else 256
            
            h5feat = h5.create_dataset("features", [len(train_set), dummy_dim], dtype=np.float32)
            
            train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                      batch_size=opt.cacheBatchSize, shuffle=False, 
                                      collate_fn=ds_module.collate_fn) # 需确保 dataset 实现了 collate_fn
            
            with torch.no_grad():
                model.eval()
                for iteration, (data, indices) in enumerate(tqdm(train_loader, desc="Caching"), 1):
                    # 数据解包：需要适配 FusionDataset 的返回
                    # 假设返回的是 ((bevs, ranges), indices) 或者单纯 input
                    if isinstance(data, (tuple, list)):
                        bevs, ranges = data
                        bevs = bevs.to(device)
                        # ranges = ranges.to(device) # 如果只是 BEV-only ablation
                    else:
                        bevs = data.to(device)

                    # 强制使用 BEV Only (根据你的需求)
                    res = model.forward_bev_only(bevs)
                    if isinstance(res, tuple): res = res[-1]
                    # res = F.normalize(res, p=2, dim=1)
                    h5feat[indices, :] = res.detach().cpu().numpy()
        
        train_set.mining = True
        train_set.refreshCache()

    # === 正式训练 ===
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                              batch_size=opt.batchSize, shuffle=True, 
                              collate_fn=ds_module.collate_fn)
    model.train()
    criterion = TripletLoss().to(device)

    # 这里的循环假设 Dataset 返回 (Query, Pos, Neg, Indices)
    # 这是一个关键点：你的 fusion_dataset.TrainingDataset 需要支持 Triplet 返回
    for iteration, (query, positives, negatives, indices) in enumerate(train_loader):
        # 注意：这里需要处理 Fusion 的数据结构 (可能包含 Range 图)
        # 为简化，假设 query 包含 (bev_q, range_q)
        
        # 简化处理：假设只用 BEV (对应代码1的逻辑)
        # 如果 query 是 tuple (bev, range)，需要拆包
        if isinstance(query, (list, tuple)): 
            q_bev, _ = query
            p_bev, _ = positives
            n_bev, _ = negatives
        else:
            q_bev, p_bev, n_bev = query, positives, negatives

        B = q_bev.shape[0]
        input_bevs = torch.cat([q_bev, p_bev, n_bev]).to(device)

        # 前向传播 (BEV Only)
        _, _, global_descs = model.forward_bev_only(input_bevs, return_all=True) # 需要修改 model 支持 return_all

        # 分割特征
        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, n_bev.shape[0]])

        optimizer.zero_grad()
        loss = 0.0
        # 计算 Loss (与代码2一致)
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
    """ 推理函数 (适配代码2的 infer_bevdata 逻辑) """
    model.eval()
    loader = DataLoader(eval_set, batch_size=opt.cacheBatchSize, shuffle=False, 
                        num_workers=opt.threads, collate_fn=ds_module.collate_fn_inference)
    
    all_global_descs = []
    with torch.no_grad():
        for data, indices in tqdm(loader, desc="Inference"):
            if data is None: continue
            (bevs, ranges) = data
            bevs = bevs.to(device)
            
            # 使用 BEV Only
            res = model.forward_bev_only(bevs)
            if isinstance(res, tuple): res = res[-1]
            # res = F.normalize(res, p=2, dim=1)
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
    print('====> FusionPlace Integrated (Code 1 Model + Code 2 Workflow)')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模型 (Code 1)
    print('===> 加载 FusionPlaceModel')
    model = FusionPlaceModel(
        vggt_path=opt.vggt_path, 
        bev_path=opt.bev_path, 
        range_dim=opt.range_dim
    )
    model = model.to(device)
    
    # 如果指定了加载路径 (恢复训练或测试)
    if opt.load_from and isfile(opt.load_from):
        print(f"===> Loading checkpoint from {opt.load_from}")
        checkpoint = torch.load(opt.load_from, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if opt.mode == 'train':
        log_dir = join(opt.runsPath, f"fusion_{datetime.now().strftime('%b%d_%H-%M-%S')}")
        writer = SummaryWriter(log_dir=log_dir)
        
        # 加载训练集 (需确保 fusion_dataset 实现了 TrainingDataset)
        train_set = ds_module.TrainingDataset(
            dataset_root=opt.dataset_root, 
            seq=opt.train_seq
        )
        
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        
        best_recall = 0.0
        
        for epoch in range(opt.nEpochs):
            # 训练一轮
            train_epoch(epoch, model, train_set, opt, device, writer, optimizer)
            scheduler.step()
            
            # 验证 (使用代码2的封装评估)
            # 需要创建临时的 InferDataset
            db_set = ds_module.FusionInferDataset(seq=opt.val_db_seq, dataset_root=opt.dataset_root)
            q_set = ds_module.FusionInferDataset(seq=opt.val_q_seq, dataset_root=opt.dataset_root)
            
            db_feats = infer_fusion(db_set, model, opt, device)
            q_feats = infer_fusion(q_set, model, opt, device)
            
            # 构造 Wrapper 进行评估
            class _Wrapper: pass
            wrapper = _Wrapper()
            wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
            wrapper.db_split_index = len(db_set.poses)
            wrapper.sample_inteval = opt.sample_interval
            
            # 借用 ds_module 或 bevdata_dataset 的评估函数
            # 注意：如果 fusion_dataset 没有 evaluateResults，请确保导入了 bevdata_dataset
            # import bevdata_dataset # 确保能用到 evaluateResults
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
        print('===> 进入测试模式 (使用代码2的封装评估)')
        
        db_set = ds_module.FusionInferDataset(seq=opt.val_db_seq, dataset_root=opt.dataset_root)
        q_set = ds_module.FusionInferDataset(seq=opt.val_q_seq, dataset_root=opt.dataset_root)
        
        print("提取特征...")
        db_feats = infer_fusion(db_set, model, opt, device)
        q_feats = infer_fusion(q_set, model, opt, device)
        
        print("开始评估...")
        class _Wrapper: pass
        wrapper = _Wrapper()
        wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
        wrapper.db_split_index = len(db_set.poses)
        wrapper.sample_inteval = opt.sample_interval
        
        # 借用 bevdata_dataset 的评估函数
        recall, _, _, _ = ds_module.evaluateResults(
            seq=f"{opt.val_db_seq}+{opt.val_q_seq}",
            global_descs=[db_feats, q_feats],
            local_feats=None,
            dataset=wrapper,
            match_results_save_path=None
        )
        print(f"✅ Final Recall@1: {recall:.4f}")