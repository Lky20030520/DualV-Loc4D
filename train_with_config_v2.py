#!/usr/bin/env python3
"""
基于 dataset_splits.json 配置文件的融合模型训练脚本
支持 Stage A (BEV-only) 和 Stage B (BEV+Range fusion)
"""

import sys
import os
import argparse
import json
from os.path import join, exists, isfile
from os import makedirs
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import h5py
import shutil
import faiss
from math import ceil

# 导入模型和数据集模块
from model.fusion_model_rangerem_v2 import FusionPlaceModel
from datasets.multi_dataset_v2 import create_datasets_from_config, MultiSeqTrainingDataset
from datasets import fusion_dataset_v2 as ds_module


class TripletLoss(nn.Module):
    """三元组损失"""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss


def get_args():
    parser = argparse.ArgumentParser(description='FusionPlace 配置文件训练器')
    
    # === 基础参数 ===
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'],
                        help='运行模式')
    parser.add_argument('--stage', type=str, default='B', choices=['A', 'B'],
                        help='训练阶段: A=BEV only, B=BEV+Range fusion')
    
    # === 数据集参数 ===
    parser.add_argument('--dataset_root', type=str, default='/mnt/kaiyan/datasets/SNAIL',
                        help='数据集根目录')
    parser.add_argument('--dataset_config', type=str, default='configs/dataset_splits2.json',
                        help='数据集配置文件路径')
    parser.add_argument('--sample_interval', type=int, default=30,
                        help='采样间隔')
    
    # === 模型参数 ===
    parser.add_argument('--bev_path', type=str, default='runs/fusion_Feb14_17-38-13!/model_best.pth.tar',
                        help='使用预训练的 BEV 模型路径')
    parser.add_argument('--load_from', type=str, default='',
                        help='恢复训练的 checkpoint 路径或目录')
    parser.add_argument('--enable_semantic_fusion', action='store_true',
                        help='启用 SCA/DINOv2 语义通道调制')
    parser.add_argument('--dino_feature_path', type=str, default='',
                        help='预提取 DINO 全局特征文件(.pt)，需包含 global_vec 与 timestamps 或 image_paths')
    parser.add_argument('--dino_dim', type=int, default=2048,
                        help='DINO 全局语义向量维度')
    parser.add_argument('--disable_gated_fusion', action='store_true',
                        help='关闭 G-CAF 门控融合，回退固定 0.1 残差')
    
    # === 缓存和输出 ===
    parser.add_argument('--cache_dir', type=str, default='./cache/fusion_config5',
                        help='缓存目录（存放聚类中心和特征）')
    parser.add_argument('--runs_dir', type=str, default='./runs',
                        help='运行结果目录（存放 checkpoints 和 logs）')
    
    # === 训练参数 ===
    parser.add_argument('--batch_size', type=int, default=1,
                        help='训练批量大小')
    parser.add_argument('--cache_batch_size', type=int, default=4,
                        help='缓存/推理批量大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--lr_step', type=int, default=2,
                        help='学习率衰减步长（epochs）')
    parser.add_argument('--lr_gamma', type=float, default=0.8,
                        help='学习率衰减系数')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='权重衰减')
    
    # === 系统参数 ===
    parser.add_argument('--threads', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=1024,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')
    
    return parser.parse_args()


def setup_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def getClusters(cluster_set, opt, model, device):
    """
    使用 K-Means 初始化 NetVLAD 的聚类中心。
    逻辑源自 main_fusion_rangerem.py，已适配 FusionPlaceModel 和 FusionDataset。
    """
    n_descriptors = 10000  # 目标：凑够 10,000 个特征点
    n_per_image = 25       # 每张图只取 25 个点
    n_im = ceil(n_descriptors / n_per_image) # 需要抽多少张图

    print(f'====> [Init] 正在从 {len(cluster_set)} 张图中随机抽取 {n_im} 张用于聚类...')

    # 1. 随机采样器
    from torch.utils.data import SubsetRandomSampler
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    
    # 2. DataLoader
    data_loader = DataLoader(
        dataset=cluster_set, 
        num_workers=opt.threads,
        batch_size=opt.cache_batch_size, 
        shuffle=False, 
        sampler=sampler,
        collate_fn=ds_module.collate_fn_inference
    )

    if not exists(opt.cache_dir): makedirs(opt.cache_dir)
    initcache = join(opt.cache_dir, 'centroids_init.hdf5')

    # 3. 提取局部特征
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            
            # REIN 的局部特征维度通常是 128
            feat_dim = 128 
            all_feats = h5.create_dataset("descriptors", [n_descriptors, feat_dim], dtype=np.float32)

            count = 0
            for iteration, (data, indices) in enumerate(tqdm(data_loader, desc="提取局部特征用于聚类")):
                if data is None: continue
                
                # 解包数据
                if isinstance(data, (tuple, list)):
                    bevs = data[0] 
                else:
                    bevs = data
                
                bevs = bevs.to(device)
                
                # 直接调用 BEV Backbone（跳过融合层）
                out1, _ = model.bev_backbone.rem(bevs) 
                
                # 变形: [B, C, H, W] -> [B, C, N_pixels] -> [B, N_pixels, C]
                local_map = out1.view(bevs.size(0), feat_dim, -1).permute(0, 2, 1)
                
                # 随机像素采样
                for i in range(bevs.size(0)):
                    if count >= n_descriptors: break
                    
                    num_pixels = local_map.size(1)
                    if num_pixels < n_per_image:
                        chosen_idx = np.random.choice(num_pixels, num_pixels, replace=False)
                    else:
                        chosen_idx = np.random.choice(num_pixels, n_per_image, replace=False)
                    
                    feats = local_map[i, chosen_idx, :].cpu().numpy()
                    
                    # 存入 HDF5
                    save_count = min(feats.shape[0], n_descriptors - count)
                    all_feats[count : count+save_count] = feats[:save_count]
                    count += save_count

        # 4. 执行 K-Means
        print('====> [Init] 开始 K-Means 聚类 (这可能需要几分钟)...')
        kmeans = faiss.Kmeans(feat_dim, 64, niter=100, verbose=True, gpu=True) 
        descriptors = h5.get("descriptors")[...]
        kmeans.train(descriptors)
        
        # 5. 保存结果
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> [Init] 聚类完成。中心已保存。')
        
    return kmeans.centroids, descriptors


def infer_fusion(eval_set, model, opt, device):
    """推理函数 - 提取全局特征"""
    model.eval()
    loader = DataLoader(eval_set, batch_size=opt.cache_batch_size, shuffle=False, 
                        num_workers=opt.threads, collate_fn=ds_module.collate_fn_inference)
    
    all_descs = []
    with torch.no_grad():
        for data, indices in tqdm(loader, desc="特征提取"):
            # 处理 BEV+Range 或单模数据
            if isinstance(data, tuple) and len(data) >= 2:
                bevs = data[0].to(device)
                ranges = data[1].to(device)
                dino_global = data[2].to(device) if len(data) > 2 else None
            else:
                bevs = data.to(device)
                ranges = None
                dino_global = None
            
            # 前向传播
            if opt.stage == 'A':
                desc = model.forward_bev_only(bevs)
            else:
                desc = model(bevs, ranges, dino_global=dino_global)
            
            # 处理返回值 (可能是 tuple)
            if isinstance(desc, tuple):
                desc = desc[-1]
            
            all_descs.append(desc.detach().cpu().numpy())
    
    return np.concatenate(all_descs, axis=0)


def train_epoch(epoch, model, train_set, opt, device, writer, optimizer):
    """训练一个 epoch"""
    if not exists(opt.cache_dir): 
        makedirs(opt.cache_dir)
    
    # === Hard Mining Cache 构建 ===
    if epoch >= 5:  # 从 epoch 5 开始启用 Hard Mining
        print(f'====> Epoch {epoch}: 构建硬样本挖掘特征缓存')
        train_set.mining = False 
        train_set.cache = join(opt.cache_dir, 'desc_cen_hardmining.hdf5')
        
        # 1. 打开 HDF5 文件
        with h5py.File(train_set.cache, mode='w') as h5:
            h5feat = None  # 延迟初始化
            
            # 2. DataLoader
            cache_loader = DataLoader(
                dataset=train_set.base_dataset, 
                num_workers=opt.threads, 
                batch_size=opt.cache_batch_size, 
                shuffle=False, 
                collate_fn=ds_module.collate_fn_inference
            )
            
            # 3. 提取特征
            with torch.no_grad():
                model.eval()
                for iteration, (data, indices) in enumerate(tqdm(cache_loader, desc="缓存特征用于 Hard Mining"), 1):
                    # 数据解包
                    if isinstance(data, (tuple, list)):
                        bevs = data[0]
                        ranges = data[1] if len(data) > 1 else None
                        dino_global = data[2] if len(data) > 2 else None
                    else:
                        bevs = data
                        ranges = None
                        dino_global = None

                    bevs = bevs.to(device)

                    # 提取特征（根据 Stage）
                    if opt.stage == 'A':
                        res = model.forward_bev_only(bevs)
                    else:
                        ranges = ranges.to(device)
                        dino_global = dino_global.to(device) if dino_global is not None else None
                        res = model(bevs, ranges, dino_global=dino_global)
                    
                    # 兼容性处理
                    if isinstance(res, tuple): 
                        res = res[-1]
                    
                    # 转为 numpy
                    res_np = res.detach().cpu().numpy()

                    # 延迟初始化 HDF5 dataset
                    if h5feat is None:
                        real_dim = res_np.shape[1]
                        print(f"\n🔍 [Auto-Detect] 检测到特征维度: {real_dim} (以此创建HDF5)")
                        h5feat = h5.create_dataset("features", [len(train_set), real_dim], dtype=np.float32)
                    
                    # 写入数据（HDF5 需要递增索引）
                    indices_list = list(indices)
                    if len(indices_list) > 1:
                        order = np.argsort(indices_list)
                        indices_list = [indices_list[i] for i in order]
                        res_np = res_np[order]
                    h5feat[indices_list, :] = res_np
        
        train_set.mining = True
        train_set.refreshCache()
    
    # === 正式训练 ===
    model.train()
    criterion = TripletLoss().to(device)
    
    train_loader = DataLoader(
        dataset=train_set, 
        num_workers=opt.threads,
        batch_size=opt.batch_size, 
        shuffle=True, 
        collate_fn=ds_module.collate_fn
    )
    
    epoch_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for iteration, batch_data in enumerate(pbar):
        # 处理 None batch
        if batch_data[0] is None:
            continue
        
        query, positives, negatives, indices = batch_data
        
        # 解包 BEV 和 Range
        q_bev, q_range = query[0], query[1]
        p_bev, p_range = positives[0], positives[1]
        n_bev, n_range = negatives[0], negatives[1]
        q_dino = query[2] if len(query) > 2 else None
        p_dino = positives[2] if len(positives) > 2 else None
        n_dino = negatives[2] if len(negatives) > 2 else None
        
        B = q_bev.shape[0]
        
        # 拼接数据以减少前向次数
        input_bevs = torch.cat([q_bev, p_bev, n_bev]).to(device)
        
        if opt.stage == 'A':
            # Stage A: 仅 BEV
            descs = model.forward_bev_only(input_bevs)
        else:
            # Stage B: BEV + Range 融合
            input_ranges = torch.cat([q_range, p_range, n_range]).to(device)
            input_dino = None
            if q_dino is not None and p_dino is not None and n_dino is not None:
                input_dino = torch.cat([q_dino, p_dino, n_dino]).to(device)
            descs = model(input_bevs, input_ranges, dino_global=input_dino)
        
        # 分割特征
        desc_q, desc_p, desc_n = torch.split(descs, [B, B, n_bev.shape[0]])
        
        # 计算损失
        optimizer.zero_grad()
        loss = 0.0
        num_negs_per_q = n_bev.shape[0] // B
        
        for i in range(B):
            neg_batch = desc_n[num_negs_per_q*i : num_negs_per_q*(i+1)]
            batch_loss = torch.max(criterion(desc_q[i], desc_p[i], neg_batch))
            loss += batch_loss
        
        loss = loss / B
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * B
        total_samples += B
        
        # 更新进度条 - 显示即时loss和累积平均loss
        avg_loss = epoch_loss / total_samples
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{avg_loss:.4f}'
        })
        
        # 记录到 tensorboard
        if iteration % 10 == 0:
            writer.add_scalar('train/batch_loss', loss.item(), 
                            epoch * len(train_loader) + iteration)
    
    # 输出 epoch 统计
    avg_epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
    print(f"✅ Epoch {epoch} 完成 | 平均 Loss: {avg_epoch_loss:.6f}")
    writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
    
    return avg_epoch_loss


def validate(model, opt, device, writer, epoch):
    """验证一个 epoch"""
    print(f"\n📊 验证中... (Epoch {epoch})")
    
    # 加载验证数据集
    db_set, q_set = create_datasets_from_config(
        mode='val',
        config_path=opt.dataset_config,
        dataset_root=opt.dataset_root,
        sample_interval=opt.sample_interval,
        suffix='_preprocessed_accm7',
        enable_dino=opt.enable_semantic_fusion,
        dino_feature_path=opt.dino_feature_path,
        dino_dim=opt.dino_dim
    )
    
    # 提取特征
    db_feats = infer_fusion(db_set, model, opt, device)
    q_feats = infer_fusion(q_set, model, opt, device)
    
    # 构造评估数据
    class Wrapper:
        pass
    wrapper = Wrapper()
    wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
    wrapper.db_split_index = len(db_set.poses)
    wrapper.sample_inteval = opt.sample_interval
    
    # 评估
    recall, _, _, _ = ds_module.evaluateResults(
        seq=f"验证-Epoch{epoch}",
        global_descs=[db_feats, q_feats],
        local_feats=None,
        dataset=wrapper,
        match_results_save_path=None
    )
    
    print(f"✨ Epoch {epoch} Recall@1: {recall:.4f}")
    writer.add_scalar('val/recall@1', recall, epoch)
    
    return recall


def save_checkpoint(state, is_best, save_dir):
    """保存 checkpoint"""
    if not exists(save_dir):
        makedirs(save_dir)
    
    ckpt_path = join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    
    if is_best:
        best_path = join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(ckpt_path, best_path)
        print(f"💾 最佳模型已保存: {best_path}")


def main():
    opt = get_args()
    setup_seed(opt.seed)
    
    # 设置设备
    device = torch.device(opt.device)
    print(f"🔧 使用设备: {device}")
    
    # 创建必要的目录
    makedirs(opt.cache_dir, exist_ok=True)
    makedirs(opt.runs_dir, exist_ok=True)
    
    # 加载模型
    print(f"📦 加载 FusionPlaceModel (Stage {opt.stage})")
    model = FusionPlaceModel(
        bev_path=opt.bev_path,
        stage=opt.stage,
        freeze_backbones=False,
        vision_dim=opt.dino_dim,
        enable_semantic_fusion=opt.enable_semantic_fusion,
        enable_gated_fusion=not opt.disable_gated_fusion
    )
    model = model.to(device)
    
    if opt.mode == 'train':
        print("\n" + "="*60)
        print("🚀 开始训练 (配置文件模式)")
        print("="*60)
        
        # 创建 log 目录
        log_dir = join(opt.runs_dir, 
                      f"fusion_{datetime.now().strftime('%b%d_%H-%M-%S')}")
        makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        # 保存配置
        config_save = join(log_dir, 'train_config.json')
        with open(config_save, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        print(f"📝 配置已保存: {config_save}")
        
        # === NetVLAD 聚类初始化 ===
        centroids_path = join(opt.cache_dir, 'centroids_init.hdf5')
        
        if opt.load_from and exists(opt.load_from):
            # 如果从 checkpoint 恢复，跳过聚类初始化
            if isfile(opt.load_from):
                ckpt_path = opt.load_from
            else:
                ckpt_path = join(opt.load_from, 'model_best.pth.tar')
                if not exists(ckpt_path):
                    ckpt_path = join(opt.load_from, 'checkpoint.pth.tar')
            
            if exists(ckpt_path):
                print(f"✅ 加载 checkpoint: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=device)
                if 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    model.load_state_dict(ckpt)
                print("✅ 已从 checkpoint 加载权重，跳过 NetVLAD 初始化。")
            else:
                print(f"⚠️ Checkpoint 未找到: {ckpt_path}")
        else:
            # 全新训练：需要 NetVLAD 聚类初始化
            if exists(centroids_path):
                print(f"✅ 检测到现有的聚类中心文件: {centroids_path}，正在直接加载...")
                with h5py.File(centroids_path, mode='r') as h5:
                    centroids = h5.get("centroids")[...]
                    descriptors = h5.get("descriptors")[...] 
                print("🚀 聚类中心加载完毕。")
            else:
                print(f"⚠️ 未找到聚类中心文件，正在从训练集重新聚类...")
                # 创建用于聚类的推理数据集（无 triplet mining）
                from datasets.multi_dataset import MultiSeqDataset
                cluster_dataset, _ = create_datasets_from_config(
                    mode='train',
                    config_path=opt.dataset_config,
                    dataset_root=opt.dataset_root,
                    sample_interval=opt.sample_interval,
                    suffix='_preprocessed_accm7',
                    enable_dino=False
                )
                # 使用 base_dataset（推理模式）
                centroids, descriptors = getClusters(cluster_dataset.base_dataset, opt, model, device)
                
                # 保存聚类结果
                with h5py.File(centroids_path, mode='w') as h5:
                    h5.create_dataset('centroids', data=centroids)
                    h5.create_dataset('descriptors', data=descriptors)
                print(f"💾 聚类中心已保存至: {centroids_path}")

            # 应用聚类中心到模型
            print("====> 正在更新 NetVLAD 权重...")
            model.bev_backbone.pooling.init_params(centroids, descriptors)
            model = model.to(device)
        
        # 加载训练数据集
        print("\n📂 加载训练数据集...")
        train_set, _ = create_datasets_from_config(
            mode='train',
            config_path=opt.dataset_config,
            dataset_root=opt.dataset_root,
            sample_interval=opt.sample_interval,
            suffix='_preprocessed_accm7',
            enable_dino=opt.enable_semantic_fusion,
            dino_feature_path=opt.dino_feature_path,
            dino_dim=opt.dino_dim
        )
        print(f"✅ 训练集加载完毕: {len(train_set)} 帧")
        
        # 优化器和调度器
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
        
        best_recall = 0.0
        
        # 训练循环
        for epoch in range(opt.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{opt.epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_epoch(epoch, model, train_set, opt, device, writer, optimizer)
            scheduler.step()
            
            # 验证
            recall = validate(model, opt, device, writer, epoch)
            
            # 保存最佳模型
            is_best = recall > best_recall
            if is_best:
                best_recall = recall
                print(f"🏆 新的最佳 Recall: {best_recall:.4f}")
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_recall': best_recall,
                'optimizer': optimizer.state_dict(),
                'args': vars(opt)
            }, is_best, log_dir)
        
        writer.close()
        print(f"\n✅ 训练完成 | 最佳 Recall: {best_recall:.4f}")
        print(f"📁 结果保存在: {log_dir}")
    
    elif opt.mode in ['val', 'test']:
        print(f"\n🧪 开始 {opt.mode.upper()} 评估")
        
        # 加载模型
        if opt.load_from and exists(opt.load_from):
            if isfile(opt.load_from):
                ckpt_path = opt.load_from
            else:
                ckpt_path = join(opt.load_from, 'model_best.pth.tar')
                if not exists(ckpt_path):
                    ckpt_path = join(opt.load_from, 'checkpoint.pth.tar')
            
            if exists(ckpt_path):
                print(f"✅ 加载 checkpoint: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=device)
                if 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    model.load_state_dict(ckpt)
            else:
                raise FileNotFoundError(f"Checkpoint 未找到: {ckpt_path}")
        else:
            raise ValueError(f"必须指定 --load_from 来进行 {opt.mode} 模式")
        
        # 加载数据集
        db_set, q_set = create_datasets_from_config(
            mode=opt.mode,
            config_path=opt.dataset_config,
            dataset_root=opt.dataset_root,
            sample_interval=opt.sample_interval,
            suffix='_preprocessed_accm7',
            enable_dino=opt.enable_semantic_fusion,
            dino_feature_path=opt.dino_feature_path,
            dino_dim=opt.dino_dim
        )
        
        # 提取特征
        print("\n📊 提取特征...")
        db_feats = infer_fusion(db_set, model, opt, device)
        q_feats = infer_fusion(q_set, model, opt, device)
        
        # 评估
        print("\n📈 开始评估...")
        class Wrapper:
            pass
        wrapper = Wrapper()
        wrapper.poses = np.concatenate([db_set.poses, q_set.poses], axis=0)
        wrapper.db_split_index = len(db_set.poses)
        wrapper.sample_inteval = opt.sample_interval
        
        recall, _, _, _ = ds_module.evaluateResults(
            seq=opt.mode,
            global_descs=[db_feats, q_feats],
            local_feats=None,
            dataset=wrapper,
            match_results_save_path=None
        )
        
        print(f"\n{'='*60}")
        print(f"✨ 最终 Recall@1: {recall:.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
