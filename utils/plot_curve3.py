import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 确保 fusion_model.py 在同一目录下
from fusion_model import FusionPlaceModel

# ================= 配置区域 =================

# 1. 模型权重路径
# (请确认这些路径在你的服务器上是真实存在的)
VGGT_CKPT_PATH = r'runs/vggt_model/vggtmodel.pt'
BEV_CKPT_PATH  = r'runs/Aug08_10-17-29/model_best.pth.tar' # 即使不用也留着路径防报错

# 2. Anchor (锚点)
ANCHOR_BEV_PATH   = r'datasets/snail/radar/bev_image/enhanced_bev/1702024656.194923193.png'
ANCHOR_RANGE_PATH = r'datasets/snail/radar/range_image/range_projection_pngs/1702024656.194923193.png'

# 3. 序列文件夹
SEQUENCE_DIR_BEV   = r'datasets/snail/radar/bev_image/enhanced_bev/'
SEQUENCE_DIR_RANGE = r'datasets/snail/radar/range_image/range_projection_pngs/'

# 4. 设置
OUTPUT_PLOT_PATH = 'similarity_curve_range_only.png' # 🟠 改名了
STEP_SIZE = 5  # 步长

# ===========================================

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

def load_image_tensor(bev_path, range_path, bev_tf, range_tf, device):
    try:
        # Range 模式必须要有 Range 图
        if not os.path.exists(range_path):
            return None, None
        img_range = Image.open(range_path).convert('RGB')
        
        # BEV 图即使不用，为了接口完整也读一下（或者造个假的）
        if os.path.exists(bev_path):
            img_bev = Image.open(bev_path).convert('RGB')
        else:
            img_bev = Image.new('RGB', (512, 512), (0,0,0))
            
    except Exception as e:
        print(f"Error reading path: {e}")
        return None, None

    tensor_bev = bev_tf(img_bev).unsqueeze(0).to(device)
    tensor_range = range_tf(img_range).unsqueeze(0).to(device)
    return tensor_bev, tensor_range

def main():
    plt.switch_backend('Agg') 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动纯 Range 对比分析 (Step={STEP_SIZE})... Device: {device}")

    # 1. 加载模型 (Range Only Mode)
    print("🏗️  加载模型 (Range Only)...")
    model = FusionPlaceModel(
        vggt_path=VGGT_CKPT_PATH, 
        bev_path=BEV_CKPT_PATH, 
        range_dim=2048, 
        verbose=True
    )
    model.eval()
    model = model.to(device)

    # 2. 准备变换
    bev_tf, range_tf = get_transforms()

    # 3. 提取 Anchor 特征
    print(f"⚓ 提取 Anchor: {os.path.basename(ANCHOR_RANGE_PATH)}")
    anchor_bev, anchor_range = load_image_tensor(ANCHOR_BEV_PATH, ANCHOR_RANGE_PATH, bev_tf, range_tf, device)
    
    if anchor_range is None:
        print("❌ 无法加载 Anchor 图片")
        return

    with torch.no_grad():
        # 这里只依赖 Range 特征
        anchor_feat = model(anchor_bev, anchor_range)

    # 4. 准备序列
    if not os.path.exists(SEQUENCE_DIR_RANGE):
        print("❌ Range 序列文件夹不存在")
        return

    all_files = sorted([f for f in os.listdir(SEQUENCE_DIR_BEV) if f.endswith('.png')])
    print(f"📂 总文件: {len(all_files)}")

    similarities = []
    indices = []

    # 5. 遍历
    print("🌊 开始计算 (Range Only)...")
    with torch.no_grad():
        for i in tqdm(range(0, len(all_files), STEP_SIZE)):
            filename = all_files[i]
            curr_bev_path = os.path.join(SEQUENCE_DIR_BEV, filename)
            curr_range_path = os.path.join(SEQUENCE_DIR_RANGE, filename)

            curr_bev, curr_range = load_image_tensor(curr_bev_path, curr_range_path, bev_tf, range_tf, device)
            if curr_range is None: continue

            curr_feat = model(curr_bev, curr_range)
            score = torch.mm(anchor_feat, curr_feat.t()).item()
            
            similarities.append(score)
            indices.append(i)

    # 6. 绘图
    print(f"🎨 绘图至: {OUTPUT_PLOT_PATH}")
    plt.figure(figsize=(12, 6))
    
    # 使用橙色代表 Range/Vision 分支
    plt.plot(indices, similarities, label='Range Only Similarity', color='orange', linewidth=1.5)
    
    # 标记 Anchor
    anchor_filename = os.path.basename(ANCHOR_BEV_PATH)
    if anchor_filename in all_files:
        anchor_idx = all_files.index(anchor_filename)
        plt.scatter([anchor_idx], [1.0], color='red', s=100, label='Anchor', zorder=5)

    plt.title('Similarity Curve (Ablation: Range Only / VGGT)')
    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Similarity')
    plt.ylim(0, 1.05) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print("✅ 完成！")

if __name__ == "__main__":
    main()