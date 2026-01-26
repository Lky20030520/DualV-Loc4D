import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ==============================================================================
# 0. 智能路径修复 (Auto-Path Fix)
# ==============================================================================
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
    # 即使不用 REIN，为了类初始化不报错，最好还是留着检查
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
        # 1. Range 分支 (VGGT) - 这次的主角
        # -----------------------------------------------------------
        if VGGT is None: raise ValueError("缺失 VGGT 模块")
        self.range_backbone = VGGT()
        
        # 加载 Range 权重
        if vggt_path and os.path.exists(vggt_path):
            try:
                state_dict = torch.load(vggt_path, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                state_dict = self._remove_prefix(state_dict)
                self.range_backbone.load_state_dict(state_dict, strict=False)
                if self.verbose: print(f"🦕 [Init] Range weights loaded: {vggt_path}")
            except Exception as e:
                print(f"⚠️ Range 权重加载失败: {e}")
        else:
             print(f"⚠️ 警告: 未找到 VGGT 权重: {vggt_path}")
        
        self.range_dim = range_dim

        # -----------------------------------------------------------
        # 2. BEV 分支 (REIN) - 仅占位，防止报错
        # -----------------------------------------------------------
        # 为了保证代码通用性，我们还是初始化它，但不跑它
        if REIN is not None:
            self.bev_backbone = REIN()
            if bev_path and os.path.exists(bev_path):
                 # 加载是为了防止 forward 里如果不小心用到会报错，虽然下面 forward 并不用
                try:
                    checkpoint = torch.load(bev_path, map_location="cpu", weights_only=False)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    state_dict = self._remove_prefix(state_dict)
                    self.bev_backbone.load_state_dict(state_dict, strict=False)
                except: pass 
        else:
            self.bev_backbone = None

        self.bev_dim = bev_dim

        # 冻结所有骨干
        for p in self.range_backbone.parameters(): p.requires_grad = False
        if self.bev_backbone:
            for p in self.bev_backbone.parameters(): p.requires_grad = False

        # 融合层 (保留定义，不使用)
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
        🔴 [Ablation Mode] 纯 Range 模式 (VGGT Only)
        忽略 BEV，使用 GeM Pooling 聚合 Range 特征。
        """
        # ==========================
        # 1. 仅运行 Range 分支 (VGGT)
        # ==========================
        
        # 预处理：确保尺寸是 (70, 518)
        target_h, target_w = 70, 518
        if range_img.shape[-2:] != (target_h, target_w):
            range_img = F.interpolate(range_img, size=(target_h, target_w), 
                                      mode='bilinear', align_corners=False)
        range_input = range_img.unsqueeze(1) # (B, 1, 3, 70, 518)

        with torch.no_grad():
            # 提取 patch tokens
            agg_list, start_idx = self.range_backbone.aggregator(range_input)
            target_tokens = agg_list[-8] # 取深层特征
            patch_tokens = target_tokens[:, :, start_idx:, :]
            range_features = patch_tokens.flatten(1, 2) # (B, N, 2048)

        # ==========================
        # 2. 模拟聚合 (GeM Pooling)
        # ==========================
        # VGGT 输出的是序列，我们需要一个向量。GeM (Generalized Mean) 是常用方法。
        # GeM: f = (mean(x^p))^(1/p)
        p = 3.0
        # Permute to (B, 2048, N) for pooling
        x = range_features.permute(0, 2, 1)
        
        # 加上 clamp(min=1e-6) 防止0的幂运算产生NaN
        x = F.avg_pool1d(x.clamp(min=1e-6).pow(p), kernel_size=x.shape[-1]).squeeze(-1).pow(1./p)
        
        # L2 归一化 (得到最终描述子，维度 2048)
        final_desc = F.normalize(x, p=2, dim=1)
        
        return final_desc