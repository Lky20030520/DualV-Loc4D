#!/usr/bin/env python3
"""
基于 dataset_splits.json 配置文件的融合模型训练脚本
支持 Stage A (BEV-only) 和 Stage B (BEV+Range fusion)
"""

import sys
import os
import argparse
import json
import csv
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
import cv2
from math import ceil

# 导入模型和数据集模块
from model.fusion_model_rangerem import FusionPlaceModel
from datasets.multi_dataset import create_datasets_from_config, MultiSeqTrainingDataset
from datasets import fusion_dataset as ds_module


class TripletLoss(nn.Module):
    """三元组损失"""
    def __init__(self, margin=0.5, eps=1e-8):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum()+ self.eps)
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1)+ self.eps)
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
    parser.add_argument('--dataset_root', type=str, default='/workspace/DualV-Loc4D/datasets/npy_accum',
                        help='数据集根目录')
    parser.add_argument('--dataset_config', type=str, default='configs/dataset_splits2.json',
                        help='数据集配置文件路径')
    parser.add_argument('--sample_interval', type=int, default=1,
                        help='采样间隔')
    
    # === 模型参数 ===
    parser.add_argument('--bev_path', type=str, default='runs/fusion_Feb25_12-51-36/model_best.pth.tar',
                        help='使用预训练的 BEV 模型路径')
    parser.add_argument('--load_from', type=str, default='runs/fusion_Feb26_05-08-33/model_best.pth.tar',
                        help='恢复训练的 checkpoint 路径或目录')
    
    # === 缓存和输出 ===
    parser.add_argument('--cache_dir', type=str, default='./cache/fusion_config2',
                        help='缓存目录（存放聚类中心和特征）')
    parser.add_argument('--runs_dir', type=str, default='./runs',
                        help='运行结果目录（存放 checkpoints 和 logs）')
    
    # === 训练参数 ===
    parser.add_argument('--batch_size', type=int, default=4,
                        help='训练批量大小')
    parser.add_argument('--cache_batch_size', type=int, default=2,
                        help='缓存/推理批量大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--lr_step', type=int, default=3,
                        help='学习率衰减步长（epochs）')
    parser.add_argument('--lr_gamma', type=float, default=0.9,
                        help='学习率衰减系数')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='权重衰减')

    # === 测试可视化参数 ===
    parser.add_argument('--visualize_test', action='store_true',
                        help='在 val/test 结束后保存检索可视化结果（test 模式默认开启）')
    parser.add_argument('--vis_topk', type=int, default=1,
                        help='可视化检索 Top-K（建议 1）')
    parser.add_argument('--vis_max_cases', type=int, default=80,
                        help='最多保存多少个 query 的可视化样例')
    parser.add_argument('--vis_output_dir', type=str, default='',
                        help='可视化输出目录；为空时自动创建')
    
    # === 系统参数 ===
    parser.add_argument('--threads', type=int, default=16,
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
            if isinstance(data, tuple) and len(data) == 2:
                bevs, ranges = data
                bevs = bevs.to(device)
                ranges = ranges.to(device)
            else:
                bevs = data.to(device)
                ranges = None
            
            # 前向传播
            if opt.stage == 'A':
                desc = model.forward_bev_only(bevs)
            else:
                desc = model(bevs, ranges)
            
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
                        bevs, ranges = data
                    else:
                        bevs = data

                    bevs = bevs.to(device)

                    # 提取特征（根据 Stage）
                    if opt.stage == 'A':
                        res = model.forward_bev_only(bevs)
                    else:
                        ranges = ranges.to(device)
                        res = model(bevs, ranges)
                    
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
        q_bev, q_range = query
        p_bev, p_range = positives
        n_bev, n_range = negatives
        
        B = q_bev.shape[0]
        
        # 拼接数据以减少前向次数
        input_bevs = torch.cat([q_bev, p_bev, n_bev]).to(device)
        
        if opt.stage == 'A':
            # Stage A: 仅 BEV
            descs = model.forward_bev_only(input_bevs)
        else:
            # Stage B: BEV + Range 融合
            input_ranges = torch.cat([q_range, p_range, n_range]).to(device)
            descs = model(input_bevs, input_ranges)
        
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
        suffix='_accum_7'
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


def _resolve_bev_path(dataset, global_index):
    """从数据集中解析某个全局索引对应的样本字典。"""
    if hasattr(dataset, 'pairs'):
        if 0 <= global_index < len(dataset.pairs):
            item = dataset.pairs[global_index]
            if isinstance(item, dict):
                seq_name = getattr(dataset, 'seq', None)
                return {
                    'bev': item.get('bev'),
                    'range': item.get('range'),
                    'ts': item.get('ts', None),
                    'seq': seq_name
                }
        return None

    if hasattr(dataset, 'subdatasets') and hasattr(dataset, 'cumulative_lengths'):
        for i, (start, end) in enumerate(zip(dataset.cumulative_lengths[:-1], dataset.cumulative_lengths[1:])):
            if start <= global_index < end:
                local_idx = global_index - start
                subdataset = dataset.subdatasets[i]
                if hasattr(subdataset, 'pairs') and 0 <= local_idx < len(subdataset.pairs):
                    item = subdataset.pairs[local_idx]
                    if isinstance(item, dict):
                        seq_name = getattr(subdataset, 'seq', None)
                        return {
                            'bev': item.get('bev'),
                            'range': item.get('range'),
                            'ts': item.get('ts', None),
                            'seq': seq_name
                        }
                break
    return None


def _infer_visual_root(dataset_root):
    parent_dir = os.path.dirname(os.path.abspath(dataset_root))
    candidates = [
        os.path.join(parent_dir, 'snail_radar'),
        os.path.join(parent_dir, 'snail'),
    ]
    for cand in candidates:
        if exists(cand):
            return cand
    return None


def _raw_seq_from_processed(seq_name):
    if not seq_name:
        return None
    if '_accum_' in seq_name:
        return seq_name.split('_accum_')[0]
    return seq_name


def _build_camera_index(cam_dir):
    if not cam_dir or not exists(cam_dir):
        return None, None
    image_names = sorted([f for f in os.listdir(cam_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if len(image_names) == 0:
        return None, None
    timestamps = []
    valid_names = []
    for name in image_names:
        stem = os.path.splitext(name)[0]
        try:
            ts = float(stem)
            timestamps.append(ts)
            valid_names.append(name)
        except Exception:
            continue
    if len(valid_names) == 0:
        return None, None
    return np.asarray(timestamps, dtype=np.float64), valid_names


def _match_camera_gray(info, visual_root, cam_cache, tolerance=0.2):
    if info is None:
        return None
    seq = _raw_seq_from_processed(info.get('seq', None))
    ts = info.get('ts', None)
    if (visual_root is None) or (seq is None) or (ts is None):
        return None

    if seq not in cam_cache:
        cam_dir = os.path.join(visual_root, seq, 'zed2i', 'left')
        ts_arr, names = _build_camera_index(cam_dir)
        cam_cache[seq] = {
            'dir': cam_dir,
            'ts': ts_arr,
            'names': names
        }

    entry = cam_cache[seq]
    if entry['ts'] is None:
        return None

    idx = int(np.searchsorted(entry['ts'], ts))
    candidates = []
    if idx < len(entry['ts']):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if len(candidates) == 0:
        return None

    best = min(candidates, key=lambda i: abs(entry['ts'][i] - ts))
    if abs(entry['ts'][best] - ts) > tolerance:
        return None

    img_path = os.path.join(entry['dir'], entry['names'][best])
    if not exists(img_path):
        return None

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _read_or_placeholder_image(img_path, height=320, width=320):
    if img_path and exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return np.full((height, width, 3), 180, dtype=np.uint8)


def _read_or_placeholder_from_array(img, height=320, width=320):
    if img is not None:
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return resized
    placeholder = np.full((height, width, 3), 242, dtype=np.uint8)
    cv2.putText(placeholder, 'N/A', (width // 2 - 20, height // 2 + 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (130, 130, 130), 1, cv2.LINE_AA)
    return placeholder


def _draw_text_clean(img, text, org, font_scale=0.55, color=(40, 40, 40), thickness=1):
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def _add_card_border(img, border_color=(210, 210, 210)):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border_color, 1)
    return img


def _render_triplet_row(bev_img, gray_img, range_img, labels, section_title,
                        section_color=(80, 120, 220), cell_h=240, cell_w=300,
                        pad=10, title_h=36):
    cells = [
        _read_or_placeholder_from_array(bev_img, cell_h, cell_w),
        _read_or_placeholder_from_array(gray_img, cell_h, cell_w),
        _read_or_placeholder_from_array(range_img, cell_h, cell_w),
    ]
    cells = [_add_card_border(c.copy()) for c in cells]

    row_w = pad * 4 + cell_w * 3
    row_h = title_h + cell_h + pad * 2
    row = np.full((row_h, row_w, 3), 252, dtype=np.uint8)

    cv2.line(row, (0, title_h), (row_w - 1, title_h), (220, 220, 220), 1)
    _draw_text_clean(row, section_title, (10, 24), font_scale=0.58, color=(45, 45, 45), thickness=1)

    # 三个模态卡片
    for i, cell in enumerate(cells):
        x0 = pad + i * (cell_w + pad)
        y0 = title_h + pad
        row[y0:y0 + cell_h, x0:x0 + cell_w] = cell

        # 轻量标签条
        label_text = labels[i]
        cv2.rectangle(row, (x0 + 6, y0 + 6), (x0 + 124, y0 + 30), (255, 255, 255), -1)
        cv2.rectangle(row, (x0 + 6, y0 + 6), (x0 + 124, y0 + 30), (210, 210, 210), 1)
        _draw_text_clean(row, label_text, (x0 + 12, y0 + 24), font_scale=0.48, color=(60, 60, 60), thickness=1)

    return row


def visualize_test_results(db_set, q_set, db_feats, q_feats, opt, save_dir):
    """保存 test/val 检索可视化：CSV明细、Top-1配对图、轨迹连线图。"""
    makedirs(save_dir, exist_ok=True)
    cases_dir = join(save_dir, 'top1_cases')
    makedirs(cases_dir, exist_ok=True)

    db_feats = np.asarray(db_feats, dtype=np.float32)
    q_feats = np.asarray(q_feats, dtype=np.float32)
    topk = max(1, min(opt.vis_topk, len(db_feats)))

    index = faiss.IndexFlatL2(db_feats.shape[1])
    index.add(db_feats)
    distances, predictions = index.search(q_feats, topk)

    db_poses = np.asarray(db_set.poses)
    q_poses = np.asarray(q_set.poses)
    gt_thres = 5.0

    details_path = join(save_dir, 'retrieval_details.csv')
    visual_root = _infer_visual_root(opt.dataset_root)
    cam_cache = {}
    recall_count = 0
    all_positives = 0

    with open(details_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_idx', 'rank', 'pred_db_idx', 'feature_l2',
            'xyz_distance', 'is_tp',
            'query_bev_path', 'query_range_path', 'query_visual_gray_source',
            'pred_bev_path', 'pred_range_path', 'pred_visual_gray_source'
        ])

        for q_idx in range(len(q_feats)):
            query_pose = q_poses[q_idx]
            gt_dis = (query_pose - db_poses) ** 2
            positives = np.where(np.sum(gt_dis[:, [3, 7, 11]], axis=1) < gt_thres ** 2)[0]

            if len(positives) > 0:
                all_positives += 1

            for rank in range(topk):
                pred_idx = int(predictions[q_idx, rank])
                feat_l2 = float(distances[q_idx, rank])
                xyz_dist = float(np.linalg.norm(query_pose[[3, 7, 11]] - db_poses[pred_idx, [3, 7, 11]]))
                is_tp = bool(pred_idx in positives) if len(positives) > 0 else False

                if rank == 0 and is_tp:
                    recall_count += 1

                q_info = _resolve_bev_path(q_set, q_idx)
                db_info = _resolve_bev_path(db_set, pred_idx)
                q_vis_src = None
                d_vis_src = None
                if q_info is not None and q_info.get('seq', None) is not None:
                    q_vis_src = os.path.join(
                        visual_root if visual_root else '',
                        _raw_seq_from_processed(q_info.get('seq', '')),
                        'zed2i', 'left'
                    ) if visual_root else None
                if db_info is not None and db_info.get('seq', None) is not None:
                    d_vis_src = os.path.join(
                        visual_root if visual_root else '',
                        _raw_seq_from_processed(db_info.get('seq', '')),
                        'zed2i', 'left'
                    ) if visual_root else None

                writer.writerow([
                    q_idx, rank + 1, pred_idx, f"{feat_l2:.6f}",
                    f"{xyz_dist:.6f}", int(is_tp),
                    q_info.get('bev') if q_info else None,
                    q_info.get('range') if q_info else None,
                    q_vis_src,
                    db_info.get('bev') if db_info else None,
                    db_info.get('range') if db_info else None,
                    d_vis_src
                ])

    # Top-1 图像配对可视化
    vis_cases = min(opt.vis_max_cases, len(q_feats))
    for q_idx in range(vis_cases):
        pred_idx = int(predictions[q_idx, 0])
        query_pose = q_poses[q_idx]
        xyz_dist = float(np.linalg.norm(query_pose[[3, 7, 11]] - db_poses[pred_idx, [3, 7, 11]]))
        is_tp = xyz_dist < gt_thres

        q_info = _resolve_bev_path(q_set, q_idx)
        d_info = _resolve_bev_path(db_set, pred_idx)

        q_bev = cv2.imread(q_info['bev'], cv2.IMREAD_COLOR) if q_info and q_info.get('bev') and exists(q_info['bev']) else None
        d_bev = cv2.imread(d_info['bev'], cv2.IMREAD_COLOR) if d_info and d_info.get('bev') and exists(d_info['bev']) else None

        q_range = cv2.imread(q_info['range'], cv2.IMREAD_COLOR) if q_info and q_info.get('range') and exists(q_info['range']) else None
        d_range = cv2.imread(d_info['range'], cv2.IMREAD_COLOR) if d_info and d_info.get('range') and exists(d_info['range']) else None

        q_gray = _match_camera_gray(q_info, visual_root, cam_cache)
        d_gray = _match_camera_gray(d_info, visual_root, cam_cache)

        q_row = _render_triplet_row(
            q_bev, q_gray, q_range,
            ['BEV', 'Visual Gray', 'Range'],
            section_title='Query Modalities',
            section_color=(235, 235, 235)
        )
        d_row = _render_triplet_row(
            d_bev, d_gray, d_range,
            ['BEV', 'Visual Gray', 'Range'],
            section_title='Retrieved DB Modalities',
            section_color=(235, 235, 235)
        )

        gap = 10
        info_h = 52
        panel_w = max(q_row.shape[1], d_row.shape[1])
        panel_h = q_row.shape[0] + d_row.shape[0] + gap * 3 + info_h
        panel = np.full((panel_h, panel_w + 24, 3), 255, dtype=np.uint8)

        # 外层边框
        cv2.rectangle(panel, (6, 6), (panel.shape[1] - 7, panel.shape[0] - 7), (210, 210, 210), 1)

        # 顶部标题
        title = f"Retrieval Case #{q_idx:05d}  |  Pred DB #{pred_idx:05d}"
        _draw_text_clean(panel, title, (16, 26), font_scale=0.62, color=(30, 30, 30), thickness=1)

        y = 34
        panel[y:y + q_row.shape[0], 12:12 + q_row.shape[1]] = q_row
        y += q_row.shape[0] + gap
        panel[y:y + d_row.shape[0], 12:12 + d_row.shape[1]] = d_row

        # 底部信息条
        y += d_row.shape[0] + gap
        status_text = 'TRUE POSITIVE' if is_tp else 'FALSE POSITIVE'
        cv2.rectangle(panel, (12, y), (panel.shape[1] - 12, y + info_h), (255, 255, 255), -1)
        cv2.rectangle(panel, (12, y), (panel.shape[1] - 12, y + info_h), (225, 225, 225), 1)

        dot_color = (85, 150, 85) if is_tp else (95, 95, 200)
        cv2.circle(panel, (28, y + 25), 6, dot_color, -1)
        _draw_text_clean(panel, status_text, (42, y + 30), font_scale=0.50, color=(45, 45, 45), thickness=1)

        metric_text = f"XYZ Distance: {xyz_dist:.2f} m   (Threshold: {gt_thres:.1f} m)"
        _draw_text_clean(panel, metric_text, (265, y + 30), font_scale=0.50, color=(45, 45, 45), thickness=1)

        save_name = f"case_{q_idx:05d}_pred_{pred_idx:05d}_{'TP' if is_tp else 'FP'}.jpg"
        cv2.imwrite(join(cases_dir, save_name), panel)

    # 轨迹连线图（x-y）
    canvas_h, canvas_w = 1200, 1200
    pad = 40
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    all_x = np.concatenate([db_poses[:, 3], q_poses[:, 3]])
    all_y = np.concatenate([db_poses[:, 7], q_poses[:, 7]])
    min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
    min_y, max_y = float(np.min(all_y)), float(np.max(all_y))

    def to_canvas_xy(x, y):
        nx = 0.5 if max_x == min_x else (x - min_x) / (max_x - min_x)
        ny = 0.5 if max_y == min_y else (y - min_y) / (max_y - min_y)
        px = int(pad + nx * (canvas_w - 2 * pad))
        py = int(canvas_h - (pad + ny * (canvas_h - 2 * pad)))
        return px, py

    for i in range(len(db_poses)):
        p = to_canvas_xy(db_poses[i, 3], db_poses[i, 7])
        cv2.circle(canvas, p, 2, (255, 120, 0), -1)
    for i in range(len(q_poses)):
        p = to_canvas_xy(q_poses[i, 3], q_poses[i, 7])
        cv2.circle(canvas, p, 2, (0, 180, 0), -1)

    sampled = np.linspace(0, len(q_poses) - 1, num=max(1, min(vis_cases, len(q_poses))), dtype=int)
    for q_idx in sampled:
        pred_idx = int(predictions[q_idx, 0])
        q_pt = to_canvas_xy(q_poses[q_idx, 3], q_poses[q_idx, 7])
        d_pt = to_canvas_xy(db_poses[pred_idx, 3], db_poses[pred_idx, 7])
        xyz_dist = float(np.linalg.norm(q_poses[q_idx, [3, 7, 11]] - db_poses[pred_idx, [3, 7, 11]]))
        color = (0, 200, 0) if xyz_dist < gt_thres else (0, 0, 255)
        cv2.line(canvas, q_pt, d_pt, color, 1)

    cv2.putText(canvas, 'DB points: orange | Query points: green | Match lines: TP=green FP=red',
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.imwrite(join(save_dir, 'trajectory_matches_top1.jpg'), canvas)

    recall_top1 = recall_count / all_positives if all_positives > 0 else 0.0
    summary_path = join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'num_db': int(len(db_feats)),
            'num_query': int(len(q_feats)),
            'topk': int(topk),
            'gt_threshold_m': gt_thres,
            'recall_top1_from_visualizer': float(recall_top1),
            'details_csv': details_path,
            'cases_dir': cases_dir
        }, f, indent=2)

    print(f"🖼️ 可视化已保存到: {save_dir}")
    print(f"   - 明细: {details_path}")
    print(f"   - 配对图: {cases_dir}")
    print(f"   - 轨迹图: {join(save_dir, 'trajectory_matches_top1.jpg')}")


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
        freeze_backbones=False
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
                ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)
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
                    suffix='_accum_7'
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
            suffix='_accum_7'
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
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
            suffix='_accum_7'
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

        should_visualize = opt.visualize_test or (opt.mode == 'test')
        if should_visualize:
            if opt.vis_output_dir:
                vis_dir = opt.vis_output_dir
            else:
                vis_dir = join(
                    opt.runs_dir,
                    f"{opt.mode}_vis_{datetime.now().strftime('%b%d_%H-%M-%S')}"
                )
            visualize_test_results(
                db_set=db_set,
                q_set=q_set,
                db_feats=db_feats,
                q_feats=q_feats,
                opt=opt,
                save_dir=vis_dir
            )
        
        print(f"\n{'='*60}")
        print(f"✨ 最终 Recall@1: {recall:.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
