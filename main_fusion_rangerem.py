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
from model.fusion_model_rangerem import FusionPlaceModel

# --- 导入数据集 (使用代码1的数据集模块，但可能需要适配代码2的接口) ---
import datasets.fusion_dataset as ds_module 
# 注意：确保 fusion_dataset 中包含 evaluateResults 函数。

def get_args():
    parser = argparse.ArgumentParser(description='FusionPlace (Stage A/B Training)')
    
    # ===== 新增：阶段选择 =====
    parser.add_argument('--stage', type=str, default='B', choices=['A', 'B'],
                        help='Training stage: A=BEV only, B=BEV+Range fusion')
    
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    parser.add_argument('--dataset_root', type=str, default='./datasets/snail', help='Snail 数据集根目录')
    
    # 序列设置
    parser.add_argument('--train_seq', type=str, default='if_20231208_4', help='训练序列')
    parser.add_argument('--val_db_seq', type=str, default='if_20231208_4', help='验证数据库')
    parser.add_argument('--val_q_seq', type=str, default='if_20240116_5', help='验证查询')

    # 模型参数 (保留代码1的设置)
    parser.add_argument('--bev_path', type=str, default='runs/fusion_Feb14_17-38-13/model_best.pth.tar')
    parser.add_argument('--load_from', type=str, default='', help='恢复训练或测试的模型路径')
    parser.add_argument('--cachePath', type=str, default='./cache/fusion_integrated3/')
    parser.add_argument('--match_save_path', type=str, default='./fusion_match_results/')
    parser.add_argument('--runsPath', type=str, default='./runs/')
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--range_dim', type=int, default=2048)
    
    # 训练参数 (来自代码2)
    parser.add_argument('--batchSize', type=int, default=1, help='训练批量') 
    parser.add_argument('--cacheBatchSize', type=int, default=4, help='缓存/推理批量')
    parser.add_argument('--nEpochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrStep', type=float, default=2)
    parser.add_argument('--lrGamma', type=float, default=0.8)
    parser.add_argument('--weightDecay', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)

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
        
    # 1. 打开 HDF5 文件
        with h5py.File(train_set.cache, mode='w') as h5:
            h5feat = None  # <--- [关键] 先不创建，等数据来了再说
            
            # 2. 定义 DataLoader (注意使用 collate_fn_inference)
            train_loader = DataLoader(
                dataset=train_set.base_dataset, 
                num_workers=opt.threads, 
                batch_size=opt.cacheBatchSize, 
                shuffle=False, 
                collate_fn=ds_module.collate_fn_inference
            )
            
            # 3. 开始循环
            with torch.no_grad():
                model.eval()
                for iteration, (data, indices) in enumerate(tqdm(train_loader, desc="Caching"), 1):
                    # --- 数据解包 ---
                    if isinstance(data, (tuple, list)):
                        bevs, ranges = data  # 取 BEV 和 Range
                    else:
                        bevs = data

                    bevs = bevs.to(device)

                    # --- 提取特征 (根据 Stage) ---
                    if opt.stage == 'A':
                        # Stage A: 仅 BEV，使用专用方法（不加载 Range）
                        res = model.forward_bev_only(bevs)
                    else:
                        # Stage B: 融合
                        ranges = ranges.to(device)
                        res = model(bevs, ranges)
                    
                    # 兼容性处理：如果返回的是 tuple (out1, local, global)，取最后一个
                    if isinstance(res, tuple): res = res[-1]
                    
                    # 转为 numpy
                    res_np = res.detach().cpu().numpy()

                    # --- [核心修复] 延迟初始化 ---
                    # 只有在第一次循环，h5feat 为 None 时，才创建数据集
                    if h5feat is None:
                        real_dim = res_np.shape[1] # 获取真实维度 (比如 8192)
                        print(f"\n🔍 [Auto-Detect] 检测到特征维度: {real_dim} (以此创建HDF5)")
                        h5feat = h5.create_dataset("features", [len(train_set), real_dim], dtype=np.float32)
                    
                    # --- 写入数据 ---
                    # 此时 h5feat 的宽度是 real_dim，res_np 也是 real_dim，完美匹配
                    
                    # 额外的安全性检查：确保 indices 是排序的（h5py 有时对乱序支持不好）
                    # 由于 shuffle=False，通常不需要，但加个 list() 保险
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

    # 这里的循环假设 Dataset 返回 (Query, Pos, Neg, Indices)
    for iteration, (query, positives, negatives, indices) in enumerate(train_loader):
        # 解包双模态数据（支持单模或双模）
        if isinstance(query, (tuple, list)):
            q_bev, q_range = query
            p_bev, p_range = positives
            n_bev, n_range = negatives
        else:
            q_bev = query
            p_bev = positives
            n_bev = negatives
            q_range = p_range = n_range = None

        B = q_bev.shape[0]
        # 拼接数据以减少 Forward 次数
        input_bevs = torch.cat([q_bev, p_bev, n_bev]).to(device)
        
        # 根据 stage 选择 forward 方式
        if opt.stage == 'A':
            # Stage A: 仅 BEV，使用专用方法
            global_descs = model.forward_bev_only(input_bevs)
        else:
            # Stage B: 融合
            input_ranges = torch.cat([q_range, p_range, n_range]).to(device)
            global_descs = model(input_bevs, input_ranges)

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
    """ 推理函数 (支持 Stage A/B) """
    model.eval()
    loader = DataLoader(eval_set, batch_size=opt.cacheBatchSize, shuffle=False, 
                        num_workers=opt.threads, collate_fn=ds_module.collate_fn_inference)
    
    all_global_descs = []
    with torch.no_grad():
        for data, indices in tqdm(loader, desc="Inference"):
            # 支持 Stage A（只有 BEV）和 Stage B（BEV + Range）
            if isinstance(data, tuple) and len(data) == 2:
                bevs, ranges = data
                bevs = bevs.to(device)
            else:
                bevs = data.to(device) if not isinstance(data, torch.Tensor) else data.to(device)
            
            # 根据 stage 调用 forward
            if opt.stage == 'A':
                # Stage A: 仅 BEV，使用专用方法
                res = model.forward_bev_only(bevs)
            else:
                # Stage B: 融合
                ranges = ranges.to(device) if 'ranges' in locals() else None
                res = model(bevs, ranges)
            
            # 兼容性：处理 tuple 返回
            if isinstance(res, tuple): 
                res = res[-1]
            
            all_global_descs.append(res.detach().cpu().numpy())
            
    return np.concatenate(all_global_descs, axis=0)


# ==============================================================================
# [新增] 聚类初始化函数 (完全复刻旧代码逻辑，但适配 Fusion 架构)
# ==============================================================================
def getClusters(cluster_set, opt, model, device):
    """
    使用 K-Means 初始化 NetVLAD 的聚类中心。
    逻辑源自源代码，已适配 FusionPlaceModel 和 FusionDataset。
    """
    n_descriptors = 10000  # 目标：凑够 10,000 个特征点
    n_per_image = 25       # 每张图只取 25 个点 (和旧代码一致)
    n_im = ceil(n_descriptors / n_per_image) # 需要抽多少张图

    print(f'====> [Init] 正在从 {len(cluster_set)} 张图中随机抽取 {n_im} 张用于聚类...')

    # 1. 随机采样器 (SubsetRandomSampler)
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    
    # 2. DataLoader (使用 inference collate，只读图不读 triplet)
    data_loader = DataLoader(
        dataset=cluster_set, 
        num_workers=opt.threads,
        batch_size=opt.cacheBatchSize, 
        shuffle=False, 
        sampler=sampler,
        collate_fn=ds_module.collate_fn_inference # 确保使用了正确的打包函数
    )

    if not exists(opt.cachePath): makedirs(opt.cachePath)
    initcache = join(opt.cachePath, 'centroids_init.hdf5')

    # 3. 提取局部特征 (Local Features)
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            
            # REIN 的局部特征维度通常是 128
            feat_dim = 128 
            all_feats = h5.create_dataset("descriptors", [n_descriptors, feat_dim], dtype=np.float32)

            count = 0
            for iteration, (data, indices) in enumerate(tqdm(data_loader, desc="Extracting Local Feats")):
                if data is None: continue
                
                # 解包数据（可能是 BEV only 或 BEV+Range）
                if isinstance(data, (tuple, list)):
                    bevs = data[0] 
                else:
                    bevs = data
                
                bevs = bevs.to(device)
                
                # 直接调用 BEV Backbone（跳过融合层）
                out1, _ = model.bev_backbone.rem(bevs) 
                
                # 变形: [B, C, H, W] -> [B, C, N_pixels] -> [B, N_pixels, C]
                # 这样就把一张图变成了 N 个 128维 的向量
                local_map = out1.view(bevs.size(0), feat_dim, -1).permute(0, 2, 1)
                
                # [核心逻辑] 随机像素采样
                for i in range(bevs.size(0)):
                    if count >= n_descriptors: break
                    
                    num_pixels = local_map.size(1)
                    # 防止像素不够
                    if num_pixels < n_per_image:
                        chosen_idx = np.random.choice(num_pixels, num_pixels, replace=False)
                    else:
                        chosen_idx = np.random.choice(num_pixels, n_per_image, replace=False)
                    
                    feats = local_map[i, chosen_idx, :].cpu().numpy()
                    
                    # 存入 HDF5
                    save_count = min(feats.shape[0], n_descriptors - count)
                    all_feats[count : count+save_count] = feats[:save_count]
                    count += save_count

        # 4. 执行 K-Means (FAISS)
        print('====> [Init] 开始 K-Means 聚类 (这可能需要几分钟)...')
        # 使用 GPU 加速 (如果显存不够报错，把 gpu=True 改为 gpu=False)
        kmeans = faiss.Kmeans(feat_dim, 64, niter=100, verbose=True, gpu=True) 
        descriptors = h5.get("descriptors")[...]
        kmeans.train(descriptors)
        
        # 5. 保存结果
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> [Init] 聚类完成。中心已保存。')
        
    return kmeans.centroids, descriptors

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
    
    if not exists(opt.cachePath): 
        makedirs(opt.cachePath)
    
    # 2. 强制将聚类中心文件指定在 cachePath 下
    # 我们手动给 opt 加上这个属性，这样后面的代码都不用改
    opt.centroids_path = join(opt.cachePath, 'centroids_init.hdf5')
    
    print(f"====> Cache Directory: {opt.cachePath}")
    print(f"====> Centroids File:  {opt.centroids_path}")
    
    # 1. 初始化模型 (根据 stage 选择)
    print(f'===> 加载 FusionPlaceModel (Stage {opt.stage})')
    model = FusionPlaceModel(
        bev_path=opt.bev_path,
        stage=opt.stage,  # 关键：传入 stage 参数
        freeze_backbones=False
    )
    model = model.to(device)
    
    # ==============================================================================
    # [修改] 智能初始化逻辑
    # ==============================================================================
    if opt.mode == 'train':
            log_dir = join(opt.runsPath, f"fusion_{datetime.now().strftime('%b%d_%H-%M-%S')}")
            writer = SummaryWriter(log_dir=log_dir)

            if opt.load_from == '' : 
                # 优先尝试从本地读取已有的聚类中心
                if exists(opt.centroids_path):
                    print(f"✅ 检测到现有的聚类中心文件: {opt.centroids_path}，正在直接加载...")
                    with h5py.File(opt.centroids_path, mode='r') as h5:
                        centroids = h5.get("centroids")[...]
                        # 注意：NetVLAD 初始化通常还需要特征描述符来计算 scale (b)
                        # 如果你的 h5 文件里没存 descriptors，可以在 getClusters 存一下
                        descriptors = h5.get("descriptors")[...] 
                    print("🚀 聚类中心加载完毕。")
                else:
                    print(f"⚠️ 未找到聚类中心文件，正在从序列 {opt.train_seq} 重新聚类...")
                    cluster_dataset = ds_module.FusionInferDataset(
                        dataset_root=opt.dataset_root, 
                        seq=opt.train_seq,
                        sample_inteval=opt.sample_interval
                    )
                    centroids, descriptors = getClusters(cluster_dataset, opt, model, device)
                    
                    # [新增] 将聚类结果持久化保存，下次直接用
                    if not exists(os.path.dirname(opt.centroids_path)): 
                        makedirs(os.path.dirname(opt.centroids_path))
                    with h5py.File(opt.centroids_path, mode='w') as h5:
                        h5.create_dataset('centroids', data=centroids)
                        h5.create_dataset('descriptors', data=descriptors)
                    print(f"💾 聚类中心已保存至: {opt.centroids_path}")

                # 统一应用聚类中心到模型
                print("====> 正在更新 NetVLAD 权重...")
                model.bev_backbone.pooling.init_params(centroids, descriptors)
                model = model.to(device) 
            else:
                print("✅ 已从 checkpoint 加载权重，跳过 NetVLAD 初始化。")

            # 3. 加载正式训练集 (FusionTrainingDataset)
            # 这才是后面 train_epoch 用到的
            train_set = ds_module.FusionTrainingDataset(
                dataset_root=opt.dataset_root, 
                seq=opt.train_seq,
                sample_inteval=opt.sample_interval
            )
            
            optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightDecay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
            
            best_recall = 0.0
            
            for epoch in range(opt.nEpochs):
                # 训练一轮
                train_epoch(epoch, model, train_set, opt, device, writer, optimizer)
                scheduler.step()
                
                # ... (验证逻辑保持不变) ...
                # 验证 (使用代码2的封装评估)
                # 需要创建临时的 InferDataset
                db_set = ds_module.FusionInferDataset(seq=opt.val_db_seq, dataset_root=opt.dataset_root,sample_inteval=opt.sample_interval)
                q_set = ds_module.FusionInferDataset(seq=opt.val_q_seq, dataset_root=opt.dataset_root,sample_inteval=opt.sample_interval)
                
                db_feats = infer_fusion(db_set, model, opt, device)
                q_feats = infer_fusion(q_set, model, opt, device)
                
                # 构造 Wrapper 进行评估
                class _Wrapper: pass
                wrapper = _Wrapper()
                wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
                wrapper.db_split_index = len(db_set.poses)
                wrapper.sample_inteval = opt.sample_interval
                
                # 借用 ds_module 或 bevdata_dataset 的评估函数
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
        print(f'===> 进入测试模式 (Stage {opt.stage})')
        # 加载训练好的权重
        if opt.load_from and isfile(opt.load_from):
            print(f"Loading checkpoint: {opt.load_from}")
            checkpoint = torch.load(opt.load_from)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("⚠️ 警告：测试模式下没有指定 --load_from")
        
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
        
        # 评估
        recall, _, _, _ = ds_module.evaluateResults(
            seq=f"{opt.val_db_seq}+{opt.val_q_seq} (Stage {opt.stage})",
            global_descs=[db_feats, q_feats],
            local_feats=None,
            dataset=wrapper,
            match_results_save_path=None
        )
        print(f"✅ Final Recall@1: {recall:.4f}")