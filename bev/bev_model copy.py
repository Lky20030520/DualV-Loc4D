import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ==============================================================================
# 0. 智能路径修复 (保持不变)
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
possible_names = ["vggt-main", "vggt", "VGGT", "VGGT-main"]
vggt_root = None
for name in possible_names:
    path_candidate = os.path.join(os.getcwd(), name)
    if os.path.exists(path_candidate): vggt_root = path_candidate; break
    path_candidate = os.path.join(script_dir, name)
    if os.path.exists(path_candidate): vggt_root = path_candidate; break
if vggt_root and vggt_root not in sys.path: sys.path.append(vggt_root)

# ==============================================================================
# 1. 导入依赖
# ==============================================================================
try:
    from REIN import REIN
except ImportError:
    print("⚠️ 严重警告: 未找到 'REIN' 模块。")
    REIN = None

try:
    from vggt.models.vggt import VGGT
except ImportError:
    print("⚠️ 严重警告: 未找到 'VGGT' 模块。")
    VGGT = None

# ==============================================================================
# 2. 主模型
# ==============================================================================
class FusionPlaceModel(nn.Module):
    def __init__(self, 
                 vggt_path=None,   
                 bev_path=None,    
                 range_dim=768,      
                 bev_dim=128,        
                 embed_dim=128,      
                 num_heads=4,
                 freeze_backbones=True): 
        super().__init__()
        
        # -------------------------------------------------------
        # A. Range 分支
        # -------------------------------------------------------
        if VGGT is None: raise ValueError("缺失 VGGT 模块")
        print(f"🦕 [Init] Loading VGGT (Spatial-DINO)...")
        self.range_backbone = VGGT()
        
        if vggt_path and os.path.exists(vggt_path):
            state_dict = torch.load(vggt_path, map_location="cpu", weights_only=False)
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            self.range_backbone.load_state_dict(self._remove_prefix(state_dict), strict=False)
            print(f"✅ VGGT Weights Loaded: {vggt_path}")
        else:
            print("⚠️ 未加载 VGGT 权重，将使用随机初始化！")

        self.range_dim = range_dim

        # -------------------------------------------------------
        # B. BEV 分支 (REIN)
        # -------------------------------------------------------
        if REIN is None: raise ValueError("缺失 REIN 模块")
        print(f"🏗️  [Init] Loading REIN (BEV)...")
        self.bev_backbone = REIN()
        
        if bev_path and os.path.exists(bev_path):
            ckpt = torch.load(bev_path, map_location="cpu", weights_only=False)
            if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
            self.bev_backbone.load_state_dict(self._remove_prefix(ckpt), strict=False)
            print(f"✅ BEV Weights Loaded: {bev_path}")
            
        self.bev_dim = bev_dim

        # -------------------------------------------------------
        # C. 冻结策略
        # -------------------------------------------------------
        for p in self.range_backbone.parameters(): p.requires_grad = not freeze_backbones
        for p in self.bev_backbone.parameters():   p.requires_grad = not freeze_backbones
        
        status = "FROZEN" if freeze_backbones else "UNFROZEN"
        print(f"🧊 Backbones are {status}. Training Fusion Layer only.")

        # -------------------------------------------------------
        # D. 融合层
        # -------------------------------------------------------
        print(f"🔗 [Init] Cross-Attention: Q=BEV({bev_dim}), K=VGGT_Patch({range_dim})")
        self.proj_range = nn.Linear(self.range_dim, self.bev_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.bev_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.bev_dim)

    def _remove_prefix(self, state_dict):
        return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    def forward(self, bev_img, range_img):
        # 原有的融合前向传播 (本次测试不使用，但保留以防报错)
        B = bev_img.shape[0]
        target_h, target_w = 70, 518
        if range_img.shape[-2:] != (target_h, target_w):
            range_img = F.interpolate(range_img, size=(target_h, target_w), mode='bilinear')
        range_input = range_img.unsqueeze(1)
        
        with torch.no_grad():
            agg_list, start_idx = self.range_backbone.aggregator(range_input)
            raw_tokens = agg_list[-1] 
            patch_tokens = raw_tokens[:, :, start_idx:, :] 
            range_feats = patch_tokens.flatten(1, 2) 
            
        kv = self.proj_range(range_feats)

        with torch.no_grad():
            bev_map, _ = self.bev_backbone.rem(bev_img)
            
        b, c, h, w = bev_map.shape
        query = bev_map.flatten(2).permute(0, 2, 1)

        attn_out, _ = self.cross_attn(query=query, key=kv, value=kv)
        fused_seq = self.norm(query + attn_out)

        fused_map = fused_seq.permute(0, 2, 1).view(b, c, h, w)
        global_desc = self.bev_backbone.pooling(fused_map)
        final_desc = F.normalize(global_desc, p=2, dim=1)
        
        return final_desc

    # ==========================================================================
    # [新增] 仅用于 BEV-Only 消融实验
    # ==========================================================================
    def forward_bev_only(self, bev_img):
        """
        跳过 VGGT 和 Fusion 层，仅使用 REIN (BEV Backbone) 提取特征。
        """
        with torch.no_grad():
            # 1. 提取特征图
            bev_map, _ = self.bev_backbone.rem(bev_img)
            
            # 2. Pooling (NetVLAD / GeM)
            global_desc = self.bev_backbone.pooling(bev_map)
            
            # 3. 归一化
            final_desc = F.normalize(global_desc, p=2, dim=1)
            
        return final_desc