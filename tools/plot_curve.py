import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 确保 fusion_model.py 里的类定义已经更新为支持 bev_path 的版本
from fusion_model import FusionPlaceModel

# ================= 配置区域 =================

# 1. 模型权重路径
VGGT_CKPT_PATH = r'runs/vggt_model/vggtmodel.pt'
# 🟢 [新增] BEV 权重路径
BEV_CKPT_PATH  = r'runs/Aug08_10-17-29/model_best.pth.tar'

# 2. 【Anchor (锚点)】
ANCHOR_BEV_PATH   = r'datasets/snail/radar/if_20231208_4/bev_image/1702024841.910055429.png'
ANCHOR_RANGE_PATH = r'datasets/snail/radar/if_20231208_4/range_image/1702024841.910055429.png'

# 3. 【Sequence (序列文件夹)】
SEQUENCE_DIR_BEV   = r'datasets/snail/radar/if_20240116_5/bev_image/'
SEQUENCE_DIR_RANGE = r'datasets/snail/radar/if_20240116_5/range_image/'

# 4. 设置
OUTPUT_PLOT_PATH = 'similarity_curve_stepped.png'
STEP_SIZE = 5  # 🟢 [新增] 步长：每隔5帧计算一次

# ===========================================

def get_transforms():
    """定义预处理逻辑"""
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

def load_image_tensor(bev_path, range_path, bev_tf, range_tf, device):
    """读取并预处理单对图像"""
    try:
        img_bev = Image.open(bev_path).convert('RGB')
        img_range = Image.open(range_path).convert('RGB')
    except Exception as e:
        print(f"Error reading path: {e}")
        return None, None

    tensor_bev = bev_tf(img_bev).unsqueeze(0).to(device)
    tensor_range = range_tf(img_range).unsqueeze(0).to(device)
    return tensor_bev, tensor_range

def main():
    plt.switch_backend('Agg') 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动序列相似度分析 (Step={STEP_SIZE})... Device: {device}")

    # 1. 加载模型 (传入双路权重)
    print("🏗️  加载模型...")
    if not os.path.exists(VGGT_CKPT_PATH) or not os.path.exists(BEV_CKPT_PATH):
        print("❌ 警告：部分权重文件不存在，请检查路径！")
    
    # 🟢 [修改] 传入 bev_path
    model = FusionPlaceModel(
        vggt_path=VGGT_CKPT_PATH, 
        bev_path=BEV_CKPT_PATH, 
        range_dim=2048, 
        # verbose=False
    )
    model.eval()
    model = model.to(device)

    # 2. 准备变换
    bev_tf, range_tf = get_transforms()

    # 3. 提取 Anchor 特征
    print(f"⚓ 提取 Anchor 特征: {os.path.basename(ANCHOR_BEV_PATH)}")
    anchor_bev, anchor_range = load_image_tensor(ANCHOR_BEV_PATH, ANCHOR_RANGE_PATH, bev_tf, range_tf, device)
    
    if anchor_bev is None:
        print("❌ 无法加载 Anchor 图片")
        return

    with torch.no_grad():
        anchor_feat = model(anchor_bev, anchor_range)

    # 4. 准备序列文件列表
    if not os.path.exists(SEQUENCE_DIR_BEV):
        print("❌ 序列文件夹不存在")
        return

    all_files = sorted([f for f in os.listdir(SEQUENCE_DIR_BEV) if f.endswith('.png')])
    total_files = len(all_files)
    print(f"📂 发现总文件数: {total_files}, 预计处理: {total_files // STEP_SIZE}")

    similarities = []
    indices = []

    # 5. 遍历序列 (带步长)
    print("🌊 开始遍历序列...")
    
    # 🟢 [修改] 使用 range(start, stop, step) 实现步长
    # tqdm 显示进度条
    with torch.no_grad():
        for i in tqdm(range(0, len(all_files), STEP_SIZE)):
            filename = all_files[i]
            
            curr_bev_path = os.path.join(SEQUENCE_DIR_BEV, filename)
            curr_range_path = os.path.join(SEQUENCE_DIR_RANGE, filename)

            if not os.path.exists(curr_range_path):
                continue

            # 加载
            curr_bev, curr_range = load_image_tensor(curr_bev_path, curr_range_path, bev_tf, range_tf, device)
            if curr_bev is None: continue

            # 推理
            curr_feat = model(curr_bev, curr_range)

            # 计算相似度
            score = torch.mm(anchor_feat, curr_feat.t()).item()
            
            similarities.append(score)
            indices.append(i) # 记录真实的原始索引

    # 6. 绘制曲线图 
    print(f"🎨 正在绘制曲线图至: {OUTPUT_PLOT_PATH}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(indices, similarities, label='Similarity to Anchor', color='blue', linewidth=1.5)
    
    # 标记 Anchor
    anchor_filename = os.path.basename(ANCHOR_BEV_PATH)
    if anchor_filename in all_files:
        anchor_idx = all_files.index(anchor_filename)
        # 只有当 anchor 刚好在步长点上，或者我们强制画出来
        plt.scatter([anchor_idx], [1.0], color='red', s=100, label='Anchor Frame', zorder=5)
        plt.axvline(x=anchor_idx, color='red', linestyle='--', alpha=0.5)

    plt.title(f'Similarity Curve (Step={STEP_SIZE}, Loaded Weights)')
    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Similarity')
    plt.ylim(0, 1.05) 
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print("✅ 完成！")

if __name__ == "__main__":
    main()