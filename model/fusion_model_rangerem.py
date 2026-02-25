import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.REIN import NetVLAD, REIN # 如果 NetVLAD 定义在 REIN.py 里，请导入它
# 如果 REM_SALAD.py 里没有 NetVLAD，请把下面的 NetVLAD 类粘贴进去或者单独导入
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
        # BEV 分支 (必须)
        # =======================================================
        print(f"🏗️  [Init] BEV Branch: REIN (Dim={self.feature_dim})")
        self.bev_backbone = REIN()
        
        # 加载 BEV 权重
        if bev_path and os.path.exists(bev_path):
            print(f"🔄 正在加载 BEV 权重: {bev_path}")
            ckpt = torch.load(bev_path, map_location="cpu", weights_only=False)
            if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
            new_state_dict = {k.replace('module.', '').replace('bev_backbone.', ''): v for k, v in ckpt.items()}
            self.bev_backbone.load_state_dict(new_state_dict, strict=False)
            print(f"✅ BEV Weights Loaded")

        # =======================================================
        # Stage B：融合层（只在需要时初始化）
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
        # 训练策略
        # =======================================================
        if stage == 'A':
            # Stage A: 训练 BEV backbone（从头或从预训练），Range不存在
            for p in self.bev_backbone.parameters():   
                p.requires_grad = True
        else:
            # Stage B: 冻结 BEV backbone（保持Stage A的特征），开启Range和融合层
            for p in self.bev_backbone.rem.parameters():   
                p.requires_grad = False
            for p in self.bev_backbone.pooling.parameters():   
                p.requires_grad = True
            
            # Range 和融合层需要训练
            if self.range_backbone is not None:
                for p in self.range_backbone.parameters(): 
                    p.requires_grad = True
            
            # Cross-Attention和LayerNorm自动开启
            for p in self.cross_attn.parameters(): 
                p.requires_grad = True
            for p in self.norm_bev.parameters(): 
                p.requires_grad = True
            for p in self.norm_range.parameters(): 
                p.requires_grad = True

    def forward(self, bev_img, range_img=None):
        """
        Args:
            bev_img: [B, 3, H, W]
            range_img: [B, 3, H_r, W_r] (仅 Stage B 需要)
        
        Returns:
            final_desc: [B, 4096] (全局地点描述符)
        """
        
        B = bev_img.shape[0]
        
        # ===== Stage A: BEV-Only（简洁版本）=====
        if self.stage == 'A':
            # 只提取 BEV 特征，直接进 NetVLAD
            bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
            global_desc = self.bev_backbone.pooling(bev_map)
            final_desc = F.normalize(global_desc, p=2, dim=1)
            return final_desc
        
        # ===== Stage B: BEV + Range 融合 =====
        else:  # self.stage == 'B'
            
            # 1. BEV 特征
            bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
            bev_tokens = bev_map.flatten(2).permute(0, 2, 1)  # [B, N_bev, 128]
            
            # 2. Range 特征
            range_map, _ = self.range_backbone(range_img)  # [B, 128, H/8, W/8]
            
            # 对齐到 BEV 的尺寸
            H_bev, W_bev = bev_map.shape[2], bev_map.shape[3]
            # range_map = F.interpolate(
            #     range_map, 
            #     size=(H_bev, W_bev), 
            #     mode='bilinear', 
            #     align_corners=False
            # )  # [B, 128, H/4, W/4]
            
            range_tokens = range_map.flatten(2).permute(0, 2, 1)  # [B, N_range, 128]
            
            # 3. Cross-Attention 融合
            q = self.norm_bev(bev_tokens)
            k = self.norm_range(range_tokens)
            v = k
            
            attn_out, _ = self.cross_attn(query=q, key=k, value=v)
            
            # 启用融合（BEV冻结，充分利用Range信息）
            fused_tokens = bev_tokens + 0.1*attn_out
            
            # 4. 转回 map 格式并聚合
            fused_map = fused_tokens.permute(0, 2, 1).view(B, self.feature_dim, H_bev, W_bev)
            # fused_map = F.normalize(fused_map, p=2, dim=1)
            
            # 5. NetVLAD 聚合
            global_desc = self.bev_backbone.pooling(fused_map)
            final_desc = F.normalize(global_desc, p=2, dim=1)
            
            return final_desc
    
    # =======================================================
    # [新增] BEV-Only 前向传播（Stage A 专用）
    # =======================================================
    def forward_bev_only(self, bev_img, return_all=False):
        """
        仅使用 BEV Backbone (REIN) 提取特征，跳过 Range 和 Fusion 层。
        适配 Stage A 训练和 bev_main.py 的调用接口。
        
        Args:
            bev_img: [B, 3, H, W]
            return_all: 是否返回所有中间特征（用于兼容训练循环）
        
        Returns:
            final_desc: [B, 4096] 全局地点描述符
            或 (bev_map, None, final_desc) 当 return_all=True
        """
        # 1. 提取特征图 (Local Features)
        bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
        
        # 2. NetVLAD Pooling -> Global Descriptor
        global_desc = self.bev_backbone.pooling(bev_map)
        
        # 3. 归一化
        final_desc = F.normalize(global_desc, p=2, dim=1)  # [B, 4096]
        
        if return_all:
            # 训练循环期望返回 3 个值: (local_feats, vlad_feats, global_desc)
            # 为了兼容，前两个返回 bev_map 和 None
            return bev_map, None, final_desc
        
        return final_desc