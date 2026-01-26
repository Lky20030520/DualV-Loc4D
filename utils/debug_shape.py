import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ================= 配置区域 =================
# 这里填你刚才发的真实路径
BEV_DIR = 'datasets/snail/radar/bev_image/enhanced_bev'
RANGE_DIR = 'datasets/snail/radar/range_image/range_projection_pngs'
BATCH_SIZE = 2
# ===========================================

class SimplePairDataset(Dataset):
    def __init__(self, bev_dir, range_dir):
        self.bev_dir = bev_dir
        self.range_dir = range_dir
        
        # 1. 获取所有图片文件名
        # 假设文件名是对应的（比如 00001.png 对应 00001.png）
        if not os.path.exists(bev_dir):
            raise FileNotFoundError(f"找不到 BEV 目录: {bev_dir}")
        if not os.path.exists(range_dir):
            raise FileNotFoundError(f"找不到 Range 目录: {range_dir}")
            
        self.bev_files = sorted([f for f in os.listdir(bev_dir) if f.endswith('.png')])
        self.range_files = sorted([f for f in os.listdir(range_dir) if f.endswith('.png')])
        
        # 简单对齐：取交集，或者直接假设一一对应
        # 这里为了保险，只取两个目录里都有的文件名
        self.common_files = sorted(list(set(self.bev_files) & set(self.range_files)))
        
        print(f"📂 目录扫描完成:")
        print(f"   - BEV 目录文件数: {len(self.bev_files)}")
        print(f"   - Range 目录文件数: {len(self.range_files)}")
        print(f"   - 匹配成功文件数: {len(self.common_files)}")
        
        if len(self.common_files) == 0:
            raise ValueError("两个目录没有同名文件，无法匹配！请检查文件名是否一致。")

        # 定义预处理 (模拟进入模型前的变换)
        # DINO 通常需要 224x224 或 518x518，这里我们先保持原图大小或缩放
        # 为了 DINO，我们需要 Normalize 到 ImageNet 的均值方差
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # <--- 这里先强制缩放到 DINO 常用大小，方便 debug
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        filename = self.common_files[idx]
        
        # 加载 BEV
        bev_path = os.path.join(self.bev_dir, filename)
        bev_img = Image.open(bev_path).convert('RGB') # 即使是灰度图，也转成 RGB 方便统一处理
        bev_tensor = self.transform(bev_img)
        
        # 加载 Range
        range_path = os.path.join(self.range_dir, filename)
        range_img = Image.open(range_path).convert('RGB')
        range_tensor = self.transform(range_img)
        
        return bev_tensor, range_tensor, filename

def debug_flow():
    print("🚀 开始极简数据流诊断...")
    
    try:
        dataset = SimplePairDataset(BEV_DIR, RANGE_DIR)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print("\n正在读取第一个 Batch...")
        for batch in loader:
            bev_tensors, range_tensors, filenames = batch
            
            print("\n" + "="*40)
            print("🔍 DATA SHAPE REPORT (基于 PNG 读取)")
            print("="*40)
            
            # --- 1. BEV 信息 ---
            print(f"1. BEV Tensor (Query Input):")
            print(f"   Shape: {bev_tensors.shape}  (Batch, Channel, H, W)")
            print(f"   Range: [{bev_tensors.min():.2f}, {bev_tensors.max():.2f}]")
            
            # --- 2. Range 信息 ---
            print(f"\n2. Range Tensor (Key/Value Input):")
            print(f"   Shape: {range_tensors.shape} (Batch, Channel, H, W)")
            print(f"   Range: [{range_tensors.min():.2f}, {range_tensors.max():.2f}]")
            
            # --- 3. 维度分析 ---
            B, C_r, H_r, W_r = range_tensors.shape
            print("\n💡 关键检查点:")
            if C_r == 3:
                print("   ✅ Range 图是 3 通道 (RGB)，可以直接喂给 DINO！")
            else:
                print(f"   ⚠️ Range 图是 {C_r} 通道，DINO 需要适配！")
                
            print(f"   ℹ️ 当前文件名示例: {filenames[0]}")
            print("="*40 + "\n")
            
            break # 只看一个就行
            
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    debug_flow()