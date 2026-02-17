import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.REIN import NetVLAD, REIN
from model.RangeRes import RangeREM


class SemanticChannelFusion(nn.Module):
    """
    轻量语义融合：用 DINO 全局语义做 channel-wise 调制
    
    核心思想：
    - 灰度图（前视 FOV）和 Range（360°）空间不对应
    - 不做 pixel-level 注意力，只做全局语义调制
    - DINO 提取全局向量，用来调制 Range 的 channel 权重
    
    输入:
        f_range: [B, C_r, H_r, W_r] - Range 特征图
        dino_global: [B, C_v] - DINO 全局特征向量
    
    输出:
        [B, C_r, H_r, W_r] - 语义增强的 Range 特征
    """
    def __init__(self, radar_dim=128, vision_dim=2048):
        super().__init__()
        
        # 将 DINO 全局特征投影到 Range 通道维度
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, radar_dim),
            nn.ReLU(),
            nn.Linear(radar_dim, radar_dim),
            nn.Sigmoid()  # 生成 0-1 的 channel 权重
        )
        
    def forward(self, f_range, dino_global):
        """
        f_range: [B, C_r, H_r, W_r] - Range 特征图
        dino_global: [B, C_v] - DINO 全局特征
        """
        # 生成 channel-wise 权重
        channel_weights = self.vision_proj(dino_global)  # [B, C_r]
        
        # 广播到空间维度：[B, C_r] -> [B, C_r, 1, 1]
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Channel-wise 调制（保持主干，轻微增强）
        # 权重范围是 0-1，这里用 0.8 + 0.4*weights 让范围在 [0.8, 1.2]
        f_enhanced = f_range * (0.8 + 0.4 * channel_weights)
        
        return f_enhanced


class FusionPlaceModelLite(nn.Module):
    """
    轻量三模态融合模型：BEV + Range + 灰度图语义
    
    特点：
    - BEV 和 Range 做空间融合（Cross-Attention）
    - 灰度图只提供全局语义（channel-wise 调制）
    - 可选开关：enable_semantic_fusion
    """
    def __init__(
        self,
        bev_path=None,
        embed_dim=128,
        num_heads=4,
        freeze_backbones=False,
        stage="B",
        vision_dim=2048,  # DINO 全局特征维度
        enable_semantic_fusion=True,
        **kwargs,
    ):
        super().__init__()

        self.stage = stage
        self.feature_dim = 128
        self.enable_semantic_fusion = enable_semantic_fusion

        # =======================================================
        # BEV 分支 (必须)
        # =======================================================
        print(f"🏗️  [Init] BEV Branch: REIN (Dim={self.feature_dim})")
        self.bev_backbone = REIN()

        # 加载 BEV 权重
        if bev_path and os.path.exists(bev_path):
            print(f"🔄 正在加载 BEV 权重: {bev_path}")
            ckpt = torch.load(bev_path, map_location="cpu", weights_only=False)
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            new_state_dict = {k.replace("module.", "").replace("bev_backbone.", ""): v for k, v in ckpt.items()}
            self.bev_backbone.load_state_dict(new_state_dict, strict=False)
            print("✅ BEV Weights Loaded")

        # =======================================================
        # Stage B：融合层（只在需要时初始化）
        # =======================================================
        if stage == "B":
            print("🦕 [Init] Stage B: Range Branch + Cross-Attention")
            print(f"🔗 [Init] Range Branch: RangeREM (Dim={self.feature_dim})")
            self.range_backbone = RangeREM(rotations=8)

            if self.enable_semantic_fusion:
                print(f"🧩 [Init] SemanticChannelFusion: DINO({vision_dim}) -> Range({self.feature_dim})")
                self.semantic_fusion = SemanticChannelFusion(
                    radar_dim=self.feature_dim,
                    vision_dim=vision_dim,
                )
            else:
                self.semantic_fusion = None

            print(f"⚙️  [Init] Cross-Attention: Dim={self.feature_dim}")
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.norm_bev = nn.LayerNorm(self.feature_dim)
            self.norm_range = nn.LayerNorm(self.feature_dim)
        else:
            print("📍 [Init] Stage A: BEV-Only Mode (Range disabled)")
            self.range_backbone = None
            self.semantic_fusion = None

        # =======================================================
        # 训练策略
        # =======================================================
        if stage == "A":
            for p in self.bev_backbone.parameters():
                p.requires_grad = True
        else:
            # Stage B: 冻结 BEV backbone，开启 Range 和融合层
            for p in self.bev_backbone.rem.parameters():
                p.requires_grad = False
            for p in self.bev_backbone.pooling.parameters():
                p.requires_grad = True

            if self.range_backbone is not None:
                for p in self.range_backbone.parameters():
                    p.requires_grad = True

            if self.semantic_fusion is not None:
                for p in self.semantic_fusion.parameters():
                    p.requires_grad = True

            for p in self.cross_attn.parameters():
                p.requires_grad = True
            for p in self.norm_bev.parameters():
                p.requires_grad = True
            for p in self.norm_range.parameters():
                p.requires_grad = True

    def forward(self, bev_img, range_img=None, dino_global=None):
        """
        Args:
            bev_img: [B, 3, H, W]
            range_img: [B, 3, H_r, W_r] (仅 Stage B 需要)
            dino_global: [B, C_v] (可选，DINO 全局特征向量)

        Returns:
            final_desc: [B, 4096] (全局地点描述符)
        """

        B = bev_img.shape[0]

        # ===== Stage A: BEV-Only（简洁版本）=====
        if self.stage == "A":
            bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
            global_desc = self.bev_backbone.pooling(bev_map)
            final_desc = F.normalize(global_desc, p=2, dim=1)
            return final_desc

        # ===== Stage B: BEV + Range 融合 =====
        else:
            # 1. BEV 特征
            bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
            bev_tokens = bev_map.flatten(2).permute(0, 2, 1)  # [B, N_bev, 128]

            # 2. Range 特征
            range_map, _ = self.range_backbone(range_img)  # [B, 128, H/4, W/4]

            # 可选：用 DINO 全局语义增强 Range 分支
            if self.semantic_fusion is not None and dino_global is not None:
                range_map = self.semantic_fusion(range_map, dino_global)

            # 对齐到 BEV 的尺寸
            H_bev, W_bev = bev_map.shape[2], bev_map.shape[3]
            range_map = F.interpolate(
                range_map,
                size=(H_bev, W_bev),
                mode="bilinear",
                align_corners=False,
            )

            range_tokens = range_map.flatten(2).permute(0, 2, 1)  # [B, N_range, 128]

            # 3. Cross-Attention 融合
            q = self.norm_bev(bev_tokens)
            k = self.norm_range(range_tokens)
            v = k

            attn_out, _ = self.cross_attn(query=q, key=k, value=v)

            # 启用融合（BEV冻结，充分利用Range信息）
            fused_tokens = bev_tokens + 0.1 * attn_out

            # 4. 转回 map 格式并聚合
            fused_map = fused_tokens.permute(0, 2, 1).view(B, self.feature_dim, H_bev, W_bev)

            # 5. NetVLAD 聚合
            global_desc = self.bev_backbone.pooling(fused_map)
            final_desc = F.normalize(global_desc, p=2, dim=1)

            return final_desc

    def forward_bev_only(self, bev_img, return_all=False):
        """
        仅使用 BEV Backbone (REIN) 提取特征，跳过 Range 和 Fusion 层。
        适配 Stage A 训练和 bev_main.py 的调用接口。
        """
        bev_map, _ = self.bev_backbone.rem(bev_img)  # [B, 128, H/4, W/4]
        global_desc = self.bev_backbone.pooling(bev_map)
        final_desc = F.normalize(global_desc, p=2, dim=1)  # [B, 4096]

        if return_all:
            return bev_map, None, final_desc

        return final_desc
