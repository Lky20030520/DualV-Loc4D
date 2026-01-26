import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ==============================================================================
# 0. 智能路径修复 (Auto-Path Fix)
# ==============================================================================
# 防止找不到 vggt 或 REIN 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
possible_names = ["vggt-main", "vggt", "VGGT", "VGGT-main"]
vggt_root = None

for name in possible_names:
    path_candidate = os.path.join(os.getcwd(), name)
    if os.path.exists(path_candidate):
        vggt_root = path_candidate
        break
    path_candidate = os.path.join(script_dir, name)
    if os.path.exists(path_candidate):
        vggt_root = path_candidate
        break

if vggt_root and vggt_root not in sys.path:
    sys.path.append(vggt_root)

# ==============================================================================
# 1. 导入依赖
# ==============================================================================
try:
    from REIN import REIN
except ImportError:
    print("⚠️  严重警告: 未找到 'REIN' 模块。请确认 REIN.py 文件是否存在。")
    REIN = None

try:
    from vggt.models.vggt import VGGT
except ImportError:
    VGGT = None

class FusionPlaceModel(nn.Module):
    def __init__(self, 
                 vggt_path=None,   
                 bev_path=None,    
                 range_dim=2048,    
                 bev_dim=128,       
                 fusion_heads=4,
                 verbose=True): 
        super().__init__()
        self.verbose = verbose
        
        # -----------------------------------------------------------
        # 虽然我们这次只用 BEV，但为了不报错，还是把 VGGT 初始化留着
        # -----------------------------------------------------------
        if VGGT is None: raise ValueError("缺失 VGGT 模块")
        self.range_backbone = VGGT()
        
        # 加载 Range 权重 (这次实验虽然不用，但为了防止代码崩溃还是加载一下)
        if vggt_path and os.path.exists(vggt_path):
            try:
                state_dict = torch.load(vggt_path, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                state_dict = self._remove_prefix(state_dict)
                self.range_backbone.load_state_dict(state_dict, strict=False)
                if self.verbose: print(f"🦕 [Init] Range weights loaded.")
            except Exception as e:
                print(f"⚠️ Range 权重加载失败: {e}")
        
        self.range_dim = range_dim

        # -----------------------------------------------------------
        # 2. BEV 分支 (REIN) - 这次的主角
        # -----------------------------------------------------------
        if REIN is None: raise ValueError("缺失 REIN 模块")
        self.bev_backbone = REIN()
        
        # 🟢 加载 BEV 权重 (关键步骤)
        if bev_path and os.path.exists(bev_path):
            if self.verbose: print(f"🏗️  [Init] Loading BEV weights: {bev_path}")
            try:
                # 🟢 修复 weights_only 报错
                checkpoint = torch.load(bev_path, map_location="cpu", weights_only=False)
                
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                state_dict = self._remove_prefix(state_dict)
                msg = self.bev_backbone.load_state_dict(state_dict, strict=False)
                if self.verbose: print(f"   -> BEV Load Info: {msg}")
            except Exception as e:
                print(f"❌ BEV 权重加载严重错误: {e}")
        else:
            print(f"⚠️ 警告: 未找到 BEV 权重文件: {bev_path}，将使用随机初始化！")

        self.bev_dim = bev_dim

        # 冻结所有骨干
        for p in self.range_backbone.parameters(): p.requires_grad = False
        for p in self.bev_backbone.parameters():   p.requires_grad = False

        # 融合层 (这次实验不会用到，但为了类结构完整保留定义)
        self.range_proj = nn.Linear(self.range_dim, self.bev_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.bev_dim, num_heads=fusion_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.bev_dim)

    def _remove_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    def forward(self, bev_img, range_img):
        """
        🔴 [Ablation Mode] 纯 BEV 模式
        这里我们故意忽略 range_img 和融合层，直接输出 BEV 特征。
        """
        # ==========================
        # 1. 仅运行 BEV 分支
        # ==========================
        with torch.no_grad():
            # 提取 BEV 特征图 (B, 128, H, W)
            bev_map, _ = self.bev_backbone.rem(bev_img)
        
        # ==========================
        # 2. 直接聚合输出 (跳过融合！)
        # ==========================
        # 使用 REIN 自带的 NetVLAD/Pooling 层
        global_desc = self.bev_backbone.pooling(bev_map)
        
        # L2 归一化
        final_desc = F.normalize(global_desc, p=2, dim=1)
        
        return final_desc