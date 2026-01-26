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
# 2. 主模型 (UniPR-3D 风格: Patch Token Cross-Attention)
# ==============================================================================
class FusionPlaceModel(nn.Module):
    def __init__(self, 
                 vggt_path=None,   
                 bev_path=None,    
                 range_dim=768,      # VGGT(Base) 的 Patch Token 维度通常是 768
                 bev_dim=128,        # REIN 的 BEV 特征维度
                 embed_dim=128,      # 融合时的对齐维度
                 num_heads=4,
                 freeze_backbones=True): 
        super().__init__()
        
        # -------------------------------------------------------
        # A. Range 分支 (VGGT - 空间增强版 DINO)
        # -------------------------------------------------------
        if VGGT is None: raise ValueError("缺失 VGGT 模块")
        print(f"🦕 [Init] Loading VGGT (Spatial-DINO)...")
        # 初始化 VGGT
        self.range_backbone = VGGT()
        
        # 加载 VGGT 预训练权重 (这包含了 DINO 权重 + 空间微调)
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
        # C. 冻结策略 (Freeze Strategy)
        # -------------------------------------------------------
        # 按照师兄建议：使用预训练好的参数提取特征，重点训练融合层
        for p in self.range_backbone.parameters(): p.requires_grad = not freeze_backbones
        for p in self.bev_backbone.parameters():   p.requires_grad = not freeze_backbones
        
        status = "FROZEN" if freeze_backbones else "UNFROZEN"
        print(f"🧊 Backbones are {status}. Training Fusion Layer only.")

        # -------------------------------------------------------
        # D. 融合层 (Cross Attention)
        # -------------------------------------------------------
        # 师兄原话：正向图做KV (1)，BEV做Q (2)，维度与特征2相同
        print(f"🔗 [Init] Cross-Attention: Q=BEV({bev_dim}), K=VGGT_Patch({range_dim})")
        
        # 投影层：把 VGGT Patch 维度映射到 BEV 维度
        self.proj_range = nn.Linear(self.range_dim, self.bev_dim)
        
        # Cross Attention
        # embed_dim 设为 bev_dim (128)，保证输出也是 128，方便后面接 BEV Pooling
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.bev_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(self.bev_dim)

    def _remove_prefix(self, state_dict):
        return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    def forward(self, bev_img, range_img):
        B = bev_img.shape[0]
        
        # ==========================================
        # 1. 提取 VGGT Patch Tokens -> 做 Key/Value
        #    (参考 UniPR-3D: 提取 Patch Tokens)
        # ==========================================
        target_h, target_w = 70, 518
        if range_img.shape[-2:] != (target_h, target_w):
            range_img = F.interpolate(range_img, size=(target_h, target_w), mode='bilinear')
        
        # range_input: (B, 1, 3, H, W) -> VGGT 需要5维输入
        range_input = range_img.unsqueeze(1)
        
        with torch.no_grad():
            # 调用 VGGT 的 aggregator
            # 它会返回多层特征，UniPR-3D 论文提到使用 "3D patch tokens" [cite: 436]
            # 在 VGGT 代码中，agg_list[-1] 或 [-8] 通常包含深层 Patch 信息
            agg_list, start_idx = self.range_backbone.aggregator(range_input)
            
            # 选取倒数第X层的 Patch Tokens (这里选 -1 代表最深层，包含最丰富的语义/空间信息)
            # 形状通常是 (B, 1, N_patches, Dim)
            raw_tokens = agg_list[-1] 
            
            # 剥离 CLS/Register token，只取 Patch 部分
            # start_idx 通常指示 Patch Token 的起始位置
            patch_tokens = raw_tokens[:, :, start_idx:, :] 
            
            # 展平为序列: (B, N, Dim)
            range_feats = patch_tokens.flatten(1, 2) 
            
        # 投影到对齐维度 (B, N, 768) -> (B, N, 128)
        kv = self.proj_range(range_feats)

        # ==========================================
        # 2. 提取 BEV Feature Map -> 做 Query
        # ==========================================
        with torch.no_grad():
            # 提取 BEV 中间层特征图 (B, 128, H, W)
            bev_map, _ = self.bev_backbone.rem(bev_img)
            
        b, c, h, w = bev_map.shape
        # 展平为序列作为 Query: (B, HW, 128)
        query = bev_map.flatten(2).permute(0, 2, 1)

        # ==========================================
        # 3. Cross Attention 融合
        #    (师兄要求: 纬度和特征2相同, BEV做Q, Range做KV)
        # ==========================================
        # Q = BEV, K = Range, V = Range
        # 输出形状与 Q 一致: (B, HW, 128)
        attn_out, _ = self.cross_attn(query=query, key=kv, value=kv)
        
        # 残差连接 + 归一化 (保留 BEV 原始结构)
        fused_seq = self.norm(query + attn_out)

        # ==========================================
        # 4. 后续编码步骤一致 (恢复 BEV 结构 -> Pooling)
        # ==========================================
        # 恢复成 (B, 128, H, W)
        fused_map = fused_seq.permute(0, 2, 1).view(b, c, h, w)
        
        # 这里的 pooling 是 REIN 自带的 NetVLAD/GeM
        # 它期望输入是 2D 特征图，输出是全局描述子
        global_desc = self.bev_backbone.pooling(fused_map)
        
        # 最终归一化
        final_desc = F.normalize(global_desc, p=2, dim=1)
        
        return final_desc