"""DualV-Loc4D v3 model implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from model.REIN import NetVLAD, REIN
from model.RangeRes import RangeREM 


class SemanticChannelFusion(nn.Module):
    def __init__(self, radar_dim=128, vision_dim=2048):
        super().__init__()
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, radar_dim),
            nn.ReLU(),
            nn.Linear(radar_dim, radar_dim),
            nn.Sigmoid()
        )

    def forward(self, f_range, dino_global):
        channel_weights = self.vision_proj(dino_global).unsqueeze(-1).unsqueeze(-1)
        return f_range * (0.8 + 0.4 * channel_weights)


class GatedFusionUnitV3(nn.Module):
    """
    Gate fusion on feature maps.

    Input:  F_bev [B, C, H, W], F_attn [B, C, H, W]
    Output: alpha [B, C, H, W], fused [B, C, H, W]
    """
    def __init__(self, embed_dim, init_value=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Use 1x1 conv as a per-location MLP on feature maps.
        self.gate_net = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim // 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        self._init_bias(init_value)

    def _init_bias(self, init_val):
        """初始化偏置令 α ≈ init_val"""
        import math
        bias_val = math.log(init_val / (1.0 - init_val))
        
        # 最后一层 Conv2d 的偏置
        final_conv = self.gate_net[2]
        nn.init.constant_(final_conv.bias, bias_val)
        nn.init.normal_(final_conv.weight, std=0.001)

    def forward(self, bev_map, attn_map):
        """
        Args:
            bev_map:   [B, C, H, W]
            attn_map:  [B, C, H, W]
        
        Returns:
            fused:     [B, C, H, W]
            alpha:     [B, C, H, W] - 门控权重
        """
        # Concatenate in channel dimension.
        gate_input = torch.cat([bev_map, attn_map], dim=1)  # [B, 2C, H, W]
        
        # Predict channel-spatial gate.
        alpha = self.gate_net(gate_input)  # [B, C, H, W]
        
        # Residual fusion.
        fused = bev_map + alpha * attn_map
        
        return fused, alpha


class FrozenOnlineDinoExtractor(nn.Module):
    """使用本地 VGGT(包含 DINOv2 表征)在线提取全局语义向量。"""

    def __init__(self, ckpt_path=''):
        super().__init__()
        project_root = Path(__file__).resolve().parent.parent
        vggt_root = project_root / 'vggt-main'
        if vggt_root.exists() and str(vggt_root) not in sys.path:
            sys.path.append(str(vggt_root))

        try:
            from vggt.models.vggt import VGGT
        except Exception as e:
            raise RuntimeError(
                "在线 DINO 路径初始化失败：无法导入 VGGT 依赖。"
                "请安装缺失依赖（例如: pip install einops）并确认 vggt-main 可用。"
            ) from e

        self.backbone = VGGT()
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if isinstance(state, dict):
                state = {k.replace('module.', ''): v for k, v in state.items()}
                self.backbone.load_state_dict(state, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, camera_img):
        # camera_img: [B, 3, H, W], 输入应为 [0,1] 范围的 RGB 张量
        with torch.no_grad():
            images = camera_img.unsqueeze(1)
            agg_list, patch_start_idx = self.backbone.aggregator(images)
            tokens = agg_list[-1]
            patch_tokens = tokens[:, :, patch_start_idx:, :]
            global_vec = patch_tokens.mean(dim=[1, 2])
        return global_vec


class FusionPlaceModel(nn.Module):
    def __init__(self, 
                 bev_path=None,    
                 embed_dim=128,      
                 num_heads=4,
                 freeze_backbones=False,
                 stage='B',  # 'A'=BEV only, 'B'=BEV+Range fusion
                 vision_dim=2048,
                 enable_semantic_fusion=False,
                 enable_gated_fusion=True,
                 enable_online_dino=False,
                 dino_ckpt_path='',
                 **kwargs): 
        super().__init__()
        
        self.stage = stage
        self.feature_dim = 128
        self.enable_semantic_fusion = enable_semantic_fusion
        self.enable_gated_fusion = enable_gated_fusion
        self.enable_online_dino = enable_online_dino
        
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

            if self.enable_semantic_fusion:
                print(f"🧩 [Init] SemanticChannelFusion: DINO({vision_dim}) -> Range({self.feature_dim})")
                self.semantic_fusion = SemanticChannelFusion(
                    radar_dim=self.feature_dim,
                    vision_dim=vision_dim
                )
                if self.enable_online_dino:
                    print("🖼️  [Init] Online frozen DINO(VGGT) enabled")
                    self.online_dino = FrozenOnlineDinoExtractor(ckpt_path=dino_ckpt_path)
                else:
                    self.online_dino = None
            else:
                self.semantic_fusion = None
                self.online_dino = None
            
            print(f"⚙️  [Init] Cross-Attention: Dim={self.feature_dim}")
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.feature_dim, 
                num_heads=num_heads, 
                batch_first=True
            )
            self.norm_bev = nn.LayerNorm(self.feature_dim)
            self.norm_range = nn.LayerNorm(self.feature_dim)
            
            if self.enable_gated_fusion:
                print(f"🚪 [Init] Gated Fusion V3 (feature-map gate)")
                self.fusion_gate = GatedFusionUnitV3(embed_dim=self.feature_dim, init_value=0.1)
            else:
                self.fusion_gate = None
        else:
            print(f"📍 [Init] Stage A: BEV-Only Mode (Range disabled)")
            self.range_backbone = None
            self.semantic_fusion = None
            self.fusion_gate = None
            self.online_dino = None

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
                p.requires_grad = True
            for p in self.bev_backbone.pooling.parameters():   
                p.requires_grad = True
            
            # Range 和融合层需要训练
            if self.range_backbone is not None:
                for p in self.range_backbone.parameters(): 
                    p.requires_grad = True

            if self.semantic_fusion is not None:
                for p in self.semantic_fusion.parameters():
                    p.requires_grad = True
                if self.online_dino is not None:
                    for p in self.online_dino.parameters():
                        p.requires_grad = False
            
            # Cross-Attention和LayerNorm自动开启
            for p in self.cross_attn.parameters(): 
                p.requires_grad = True
            for p in self.norm_bev.parameters(): 
                p.requires_grad = True
            for p in self.norm_range.parameters(): 
                p.requires_grad = True
            if self.fusion_gate is not None:
                for p in self.fusion_gate.parameters():
                    p.requires_grad = True

    def forward(self, bev_img, range_img=None, dino_global=None, camera_img=None):
        """
        Args:
            bev_img: [B, 3, H, W]
            range_img: [B, 3, H_r, W_r] (仅 Stage B 需要)
            dino_global: [B, D] (离线模式)
            camera_img: [B, 3, H, W] (在线模式)
        
        Returns:
            final_desc: [B, 4096] (全局地点描述符)
        """
        
        B = bev_img.shape[0]
        
        # ===== Stage A: BEV-Only =====
        if self.stage == 'A':
            bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
            global_desc = self.bev_backbone.pooling(bev_map)
            final_desc = F.normalize(global_desc, p=2, dim=1)
            return final_desc
        
        # ===== Stage B: BEV + Range 融合 =====
        else:  # self.stage == 'B'
            
            # 1. BEV 特征图
            bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
            H_bev, W_bev = bev_map.shape[2], bev_map.shape[3]
            
            # 2. Range 特征图
            range_map, _ = self.range_backbone(range_img)  # [B, 128, H/8, W/8]

            # 在线提取DINO语义向量
            if self.online_dino is not None and camera_img is not None:
                self.online_dino.eval()
                dino_global = self.online_dino(camera_img)

            # SCA 语义增强
            if self.semantic_fusion is not None and dino_global is not None:
                if dino_global.ndim != 2:
                    raise ValueError(f"dino_global 维度错误，期望 [B, D]，实际 {tuple(dino_global.shape)}")
                if dino_global.shape[0] != range_map.shape[0]:
                    raise ValueError(
                        f"SCA 批次不对齐: dino_global.B={dino_global.shape[0]} vs range_map.B={range_map.shape[0]}"
                    )
                range_map = self.semantic_fusion(range_map, dino_global)
            
            # 对齐 Range 特征图到 BEV 尺寸
            if range_map.shape[2:] != (H_bev, W_bev):
                range_map = F.interpolate(
                    range_map,
                    size=(H_bev, W_bev),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Cross-attention and gate fusion on feature maps.
            
            # 3. 转为 token 做交叉注意力
            bev_tokens = bev_map.flatten(2).permute(0, 2, 1)  # [B, N_bev, 128]
            range_tokens = range_map.flatten(2).permute(0, 2, 1)  # [B, N_range, 128]
            
            q = self.norm_bev(bev_tokens)
            k = self.norm_range(range_tokens)
            v = k
            
            attn_out, _ = self.cross_attn(query=q, key=k, value=v)  # [B, N_bev, 128]
            
            # 4. 转回特征图形式
            attn_map = attn_out.permute(0, 2, 1).view(B, self.feature_dim, H_bev, W_bev)  # [B, 128, H, W]
            
            # 5. Gate fusion on feature maps.
            if self.fusion_gate is not None:
                fused_map, gate_alpha = self.fusion_gate(bev_map, attn_map)  # [B, 128, H, W]
            else:
                fused_map = bev_map + 0.1 * attn_map
            
            # 6. NetVLAD 聚合
            global_desc = self.bev_backbone.pooling(fused_map)
            final_desc = F.normalize(global_desc, p=2, dim=1)
            
            return final_desc
    
    def forward_bev_only(self, bev_img, return_all=False):
        """
        仅使用 BEV Backbone (REIN) 提取特征，跳过 Range 和 Fusion 层。
        
        Args:
            bev_img: [B, 3, H, W]
            return_all: 是否返回所有中间特征（用于兼容训练循环）
        
        Returns:
            final_desc: [B, 4096] 全局地点描述符
        """
        bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
        global_desc = self.bev_backbone.pooling(bev_map)
        final_desc = F.normalize(global_desc, p=2, dim=1)  # [B, 4096]
        
        if return_all:
            return bev_map, None, final_desc
        
        return final_desc
