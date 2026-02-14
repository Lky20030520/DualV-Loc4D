import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys

# 尝试导入 REIN
try:
    from REIN import REIN
except ImportError:
    print("❌ 错误：未找到 REIN.py，请确保它在当前目录下。")
    sys.exit(1)

# ================= 配置区域 =================

# 1. 权重路径 (请修改为你服务器上的实际路径)
MODEL_WEIGHTS = r'runs/Aug08_10-17-29/model_best.pth.tar'

# 2. 待对比的两张图片路径
IMAGE_A_PATH = r'datasets/snail/radar/bev_image/enhanced_bev/1702024656.194923193.png'
IMAGE_B_PATH = r'datasets/snail/radar/bev_image/enhanced_bev/1702024866.889755279.png' # 找一张附近的
# IMAGE_B_PATH = IMAGE_A_PATH # 测试自己和自己对比（应该是 1.0）

# ===========================================

def get_transforms():
    # REIN 这里的输入通常调整为 512x512 (根据之前的 bev_tf)
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

def load_model(weights_path):
    print("🏗️  正在初始化 REIN 模型...")
    model = REIN()
    
    # REIN 代码里硬编码了 .cuda()，所以必须用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("⚠️ 警告: REIN 源码包含 .cuda() 调用，使用 CPU 可能会报错！")
    
    model = model.to(device)

    if os.path.exists(weights_path):
        print(f"📥 加载权重: {weights_path}")
        # 加载权重逻辑 (处理 .tar 和 module. 前缀)
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 去除 module. 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # 加载 (strict=False 以防万一有些不匹配，通常 REIN 是匹配的)
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ 权重加载成功")
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("   -> 将使用随机初始化参数运行...")
    else:
        print(f"⚠️ 警告: 找不到权重文件 {weights_path}，将使用随机初始化！")
    
    model.eval()
    return model, device

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
    # 1. 准备模型
    model, device = load_model(MODEL_WEIGHTS)
    tf = get_transforms()

    # 2. 读取图片
    print(f"\n🖼️  读取图片 A: {os.path.basename(IMAGE_A_PATH)}")
    pil_a, tensor_a = load_image(IMAGE_A_PATH, tf, device)

    print(f"🖼️  读取图片 B: {os.path.basename(IMAGE_B_PATH)}")
    pil_b, tensor_b = load_image(IMAGE_B_PATH, tf, device)

    if tensor_a is None or tensor_b is None:
        print("❌ 无法读取图片，程序终止。")
        return

    # 3. 推理 (Extract)
    print("⚡ 正在提取特征...")
    with torch.no_grad():
        # REIN forward 返回: out1, local_feats, global_desc
        # 我们只需要第三个: global_desc
        _, _, desc_a = model(tensor_a)
        _, _, desc_b = model(tensor_b)

        # 归一化 (虽然 NetVLAD 内部做过，但为了保险再做一次 L2 Norm)
        desc_a = F.normalize(desc_a, p=2, dim=1)
        desc_b = F.normalize(desc_b, p=2, dim=1)

    # 4. 计算相似度 (Cosine Similarity)
    # 两个归一化向量的点积就是余弦相似度
    similarity = torch.mm(desc_a, desc_b.t()).item()
    print(f"\n🎯 ===========================")
    print(f"🎯 相似度得分: {similarity:.6f}")
    print(f"🎯 ===========================\n")

    # 5. 可视化绘图
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Similarity Score: {similarity:.4f}", fontsize=16, color='blue', fontweight='bold')

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

    # 保存结果
    save_path = "pair_comparison_result.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"💾 对比结果图已保存至: {save_path}")

if __name__ == "__main__":
    main()