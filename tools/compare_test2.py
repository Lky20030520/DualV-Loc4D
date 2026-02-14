import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys

# ==========================================
# 0. 自动路径修复 (确保能导入 fusion_model)
# ==========================================
# 获取当前脚本文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录 (父目录)
parent_dir = os.path.dirname(current_dir)
# 将上一级目录加入到 Python 的搜索路径中
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from fusion_model import FusionPlaceModel
    print("✅ 成功导入 FusionPlaceModel")
except ImportError:
    print(f"❌ 错误：在路径 {parent_dir} 下未找到 fusion_model.py")
    sys.exit(1)

# ================= 配置区域 =================

# 1. 权重路径
VGGT_CKPT_PATH = r'runs/vggt_model/vggtmodel.pt'

# 2. 待对比的两张 Range 图片路径
# 图片 A
IMAGE_A_PATH = r'datasets/snail/radar/if_20231208_4/range_image/1702024656.194923193.png'

# 图片 B (修改这里来测试不同图片)
IMAGE_B_PATH = r'datasets/snail/radar/if_20240116_5/range_image/1705398609.737142597.png' 

# ===========================================

def get_range_transforms():
    """Range 图像专用的预处理"""
    return transforms.Compose([
        transforms.Resize((70, 518)), # VGGT 的标准输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_image(path, tf, device):
    if not os.path.exists(path):
        print(f"❌ 图片不存在: {path}")
        return None, None
    
    try:
        img_pil = Image.open(path).convert('RGB')
        img_tensor = tf(img_pil).unsqueeze(0).to(device) # (1, C, H, W)
        return img_pil, img_tensor
    except Exception as e:
        print(f"❌ 读取图片出错: {e}")
        return None, None

def main():
    # 0. 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 Range 对比分析... Device: {device}")

    # 1. 初始化模型 (只加载 VGGT 权重)
    print("🏗️  初始化 FusionPlaceModel (Range 模式)...")
    
    # 🔴【修复】这里显式指定 range_dim=2048，解决 mat1/mat2 维度不匹配报错
    model = FusionPlaceModel(
        vggt_path=VGGT_CKPT_PATH, 
        bev_path=None, 
        range_dim=2048,  # <--- 关键修改
    )
    
    model.eval()
    model = model.to(device)

    # 2. 读取图片
    tf = get_range_transforms()
    
    print(f"\n🖼️  读取图片 A: {os.path.basename(IMAGE_A_PATH)}")
    pil_a, tensor_a = load_image(IMAGE_A_PATH, tf, device)

    print(f"🖼️  读取图片 B: {os.path.basename(IMAGE_B_PATH)}")
    pil_b, tensor_b = load_image(IMAGE_B_PATH, tf, device)

    if tensor_a is None or tensor_b is None:
        return

    # 3. 构造 Dummy BEV 输入 (因为 forward 接口需要两个参数)
    # 造一个全黑的 (1, 3, 512, 512)
    dummy_bev = torch.zeros(1, 3, 512, 512).to(device)

    # 4. 推理 (Inference)
    print("⚡ 正在提取 Range 特征...")
    with torch.no_grad():
        # 调用模型提取特征
        desc_a = model(dummy_bev, tensor_a)
        desc_b = model(dummy_bev, tensor_b)

        # 检查返回值类型 (防止返回 Tuple)
        if isinstance(desc_a, tuple):
            desc_a = desc_a[-1]
            desc_b = desc_b[-1]

    # 5. 计算相似度
    # 归一化特征向量
    desc_a = F.normalize(desc_a, p=2, dim=1)
    desc_b = F.normalize(desc_b, p=2, dim=1)
    
    # 计算余弦相似度 (向量点积)
    similarity = torch.mm(desc_a, desc_b.t()).item()
    
    print(f"\n🎯 ===========================")
    print(f"🎯 Range 相似度得分: {similarity:.6f}")
    print(f"🎯 ===========================\n")

    # 6. 画图对比
    try:
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Range Similarity: {similarity:.4f}", fontsize=14, color='orange', fontweight='bold')

        # 左图
        plt.subplot(1, 2, 1)
        plt.imshow(pil_a)
        plt.title(f"Image A\n{os.path.basename(IMAGE_A_PATH)}")
        plt.axis('off')

        # 右图
        plt.subplot(1, 2, 2)
        plt.imshow(pil_b)
        plt.title(f"Image B\n{os.path.basename(IMAGE_B_PATH)}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("compare_range_result.png")
        print("💾 对比结果已保存至: compare_range_result.png")
    except Exception as e:
        print(f"⚠️ 绘图失败 (可能是服务器没有X11支持): {e}")

if __name__ == "__main__":
    main()