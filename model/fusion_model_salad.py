import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.REM_SALAD import REINS
from model.RangeRes import RangeREM 

class FusionPlaceModel(nn.Module):
    def __init__(self, 
                 bev_path=None,    
                 embed_dim=128,      
                 num_heads=4,
                 freeze_backbones=False,
                 stage='B',  # 新增：'A'=BEV only, 'B'=BEV+Range fusion
                 **kwargs): 
        super().__init__()
        
        self.stage = stage
        self.feature_dim = 128
        
        # =======================================================
        # 1. BEV 分支 (必须) - REINS 已内置 REM + SALAD
        # =======================================================
        print(f"🏗️  [Init] BEV Branch: REINS+SALAD (Dim={self.feature_dim})")
        self.bev_backbone = REINS()
        
        # 加载 BEV 权重 (如果有)
        if bev_path and os.path.exists(bev_path):
            ckpt = torch.load(bev_path, map_location="cpu", weights_only=False)
            if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
            new_state_dict = {k.replace('module.', '').replace('bev_backbone.', ''): v for k, v in ckpt.items()}
            self.bev_backbone.load_state_dict(new_state_dict, strict=False)
            print(f"✅ BEV Weights Loaded")

        # =======================================================
        # 2. Stage B：融合层（Range + Attention）
        # =======================================================
        if stage == 'B':
            print(f"🦕 [Init] Stage B: Range Branch + Cross-Attention")
            print(f"🔗 [Init] Range Branch: RangeREM (Dim={self.feature_dim})")
            self.range_backbone = RangeREM(rotations=8)
            
            print(f"⚙️  [Init] Cross-Attention: Dim={self.feature_dim}")
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.feature_dim, 
                num_heads=num_heads, 
                batch_first=True
            )
            self.norm_bev = nn.LayerNorm(self.feature_dim)
            self.norm_range = nn.LayerNorm(self.feature_dim)
        else:
            print(f"📍 [Init] Stage A: BEV-Only Mode (Range disabled)")
            self.range_backbone = None

        # =======================================================
        # 3. 训练策略
        # =======================================================
        # BEV: 根据配置冻结
        for p in self.bev_backbone.parameters():   p.requires_grad = True
        # Range: 仅在 Stage B 时训练
        if stage == 'B' and self.range_backbone is not None:
            for p in self.range_backbone.parameters(): p.requires_grad = True

    def forward(self, bev_img, range_img=None):
        """
        Args:
            bev_img: [B, 3, H, W]
            range_img: [B, 3, H_r, W_r] (仅 Stage B 需要)
        
        Returns:
            final_desc: [B, embedding_dim] (全局地点描述符)
        """
        
        B = bev_img.shape[0]
        
        # ===== Stage A: BEV-Only =====
        if self.stage == 'A':
            # 直接使用REINS的前向传播（包含REM + SALAD聚合）
            bev_map, local_feats, global_desc = self.bev_backbone(bev_img)
            
            # 归一化
            final_desc = F.normalize(global_desc, p=2, dim=1)
            
            return final_desc
        
        # ===== Stage B: BEV + Range 融合 =====
        else:  # self.stage == 'B'
            
            # 1. BEV 特征 (使用REM，跳过pooling)
            bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
            bev_tokens = bev_map.flatten(2).permute(0, 2, 1)  # [B, N_bev, 128]
            
            # 2. Range 特征
            range_map, _ = self.range_backbone(range_img)  # [B, 128, H/8, W/8]
            range_tokens = range_map.flatten(2).permute(0, 2, 1) # [B, N_range, 128]
            
            # 3. Cross-Attention 融合
            q = self.norm_bev(bev_tokens)
            k = self.norm_range(range_tokens)
            v = k
            
            attn_out, _ = self.cross_attn(query=q, key=k, value=v)
            
            # 4. 残差连接
            fused_tokens = bev_tokens + 1.0 * attn_out  # [B, N, 128]
            
            # 5. 还原特征图形状
            C, H, W = bev_map.shape[1], bev_map.shape[2], bev_map.shape[3]
            fused_map = fused_tokens.permute(0, 2, 1).view(B, C, H, W)
            
            # 6. 计算Global Token用于SALAD
            global_token = torch.mean(fused_map, dim=[2, 3])  # [B, 128]
            
            # 7. 使用REINS的pooling（SALAD）进行聚合
            global_desc = self.bev_backbone.pooling((fused_map, global_token))
            
            # 8. 归一化
            final_desc = F.normalize(global_desc, p=2, dim=1)
            
            return final_desc
    
    
    # =======================================================
    # [新增] BEV-Only 前向传播（Stage A 专用）
    # =======================================================
    def forward_bev_only(self, bev_img, return_all=False):
        """
        仅使用 BEV Backbone (REINS) 提取特征，跳过 Range 和 Fusion 层。
        适配 Stage A 训练和调用接口。
        
        Args:
            bev_img: [B, 3, H, W]
            return_all: 是否返回所有中间特征（用于兼容训练循环）
        
        Returns:
            final_desc: [B, embedding_dim] 全局地点描述符
            或 (bev_map, None, final_desc) 当 return_all=True
        """
        # 直接使用REINS的前向传播
        bev_map, local_feats, global_desc = self.bev_backbone(bev_img)
        
        # 归一化
        final_desc = F.normalize(global_desc, p=2, dim=1)
        
        if return_all:
            # 训练循环期望返回 3 个值: (local_feats, vlad_feats, global_desc)
            # 为了兼容，前两个返回 bev_map 和 None
            return bev_map, None, final_desc
        
        return final_desc