import torch
import torch.nn.functional as F
import os
import sys
from PIL import Image
from torchvision import transforms

# 导入上面的模型
from fusion_model import FusionPlaceModel

# ================= 配置 =================
# 🔴 请修改为你实际的权重路径
VGGT_CKPT_PATH = r'runs/vggt_model/vggtmodel.pt' 

# 🔴 请修改为你实际的图片路径
PATH_TO_BEV = r'datasets/snail/radar/bev_image/enhanced_bev/1702024660.224901230.png'
PATH_TO_RANGE = r'datasets/snail/radar/range_image/range_projection_pngs/1702024660.224901230.png'
# =======================================

def load_and_preprocess_image(path, img_type='range'):
    """
    读取图片并转换为模型需要的 Tensor 格式
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 图片不存在: {path}")
    
    # 1. 读取图片 (RGB)
    img = Image.open(path).convert('RGB')
    
    # 2. 定义变换
    if img_type == 'range':
        # VGGT / UniPR-3D 标准预处理
        transform = transforms.Compose([
            transforms.Resize((70, 518)), # 强制调整到 VGGT 训练分辨率
            transforms.ToTensor(),
            # ImageNet 归一化 (VGGT 权重基于此)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    else: # bev
        # BEV 标准预处理
        transform = transforms.Compose([
            transforms.Resize((512, 512)), # 假设 BEV 模型输入为 512
            transforms.ToTensor(),
            # BEV 图通常不需要 ImageNet 归一化，或者使用简单的 0-1
        ])
    
    # 3. 转换并增加 Batch 维度 (C, H, W) -> (1, C, H, W)
    tensor = transform(img).unsqueeze(0)
    return tensor

def test_vggt_fusion():
    print("🚀 开始 VGGT Fusion 模型实图测试...")

    # 1. 准备数据
    try:
        print(f"📥 正在读取 BEV 图片: {os.path.basename(PATH_TO_BEV)}")
        bev_input = load_and_preprocess_image(PATH_TO_BEV, img_type='bev')
        
        print(f"📥 正在读取 Range 图片: {os.path.basename(PATH_TO_RANGE)}")
        range_input = load_and_preprocess_image(PATH_TO_RANGE, img_type='range')
        
        print(f"✅ 数据加载完成:")
        print(f"   -> BEV Input Shape:   {bev_input.shape}")
        print(f"   -> Range Input Shape: {range_input.shape}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 初始化模型
    print(f"\n🏗️  正在加载模型 (权重: {VGGT_CKPT_PATH})...")
    if not os.path.exists(VGGT_CKPT_PATH):
        print("❌ 错误：找不到 VGGT 权重文件。")
        # ckpt_arg = None # 实际运行建议直接报错退出，否则结果无意义
        return 
    else:
        ckpt_arg = VGGT_CKPT_PATH

    try:
        # 🔑 关键设置：range_dim=2048 匹配你的权重
        model = FusionPlaceModel(vggt_path=ckpt_arg, range_dim=2048)
        model.eval()
        
        # 移至 GPU (如果有)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        bev_input = bev_input.to(device)
        range_input = range_input.to(device)

        print(f"   Device: {device}")

        # 3. 前向传播
        print("\n🌊 运行 Forward Pass...")
        with torch.no_grad():
            output = model(bev_input, range_input)

        # 4. 结果验证与可视化打印
        print("\n✅ 测试成功！结果分析如下：")
        print("=" * 60)
        
        # [检查 1] 输出形状
        print(f"1. [维度检查] Output Shape: {output.shape}")
        if output.shape[1] == 8192:
            print("   -> ✅ 维度正确 (128 channels * 64 clusters = 8192)")
        else:
            print(f"   -> ⚠️ 维度异常 (预期 8192)")

        # [检查 2] 归一化检查
        norm_val = torch.norm(output, dim=1).item()
        print(f"2. [归一化检查] L2 Norm: {norm_val:.4f}")
        if 0.99 < norm_val < 1.01:
            print("   -> ✅ 输出已做 L2 归一化，可以直接用于计算余弦相似度")
        else:
            print("   -> ⚠️ 输出未归一化")

        # [检查 3] 数据分布 (证明不是死值)
        out_np = output.cpu().numpy().flatten()
        print(f"3. [数值分布] Min: {out_np.min():.4f}, Max: {out_np.max():.4f}, Mean: {out_np.mean():.4f}")
        if out_np.std() > 1e-6:
            print("   -> ✅ 特征具有区分度 (Standard Deviation > 0)")
            print("   -> 🎉 恭喜！BEV 特征与 Range 语义已成功融合。")
        else:
            print("   -> ❌ 警告：输出特征可能是全0或常数，请检查输入或模型权重。")
            
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 模型运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vggt_fusion()