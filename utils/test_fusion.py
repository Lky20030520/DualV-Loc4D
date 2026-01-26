import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ==============================================================================
# 1. 导入依赖
# ==============================================================================

# 尝试导入 REIN (BEVPlace++)
try:
    from REIN import REIN
except ImportError:
    print("⚠️  严重警告: 未找到 'REIN' 模块。")
    REIN = None

# 尝试导入 VGGT
# 假设 vggt 文件夹在当前目录下，或者已经被添加到 sys.path
try:
    from vggt.models.vggt import VGGT
except ImportError:
    # 尝试把当前目录下的 vggt-main 加入 path
    current_dir = os.getcwd()
    vggt_root = os.path.join(current_dir, "vggt-main")
    if os.path.exists(vggt_root) and vggt_root not in sys.path:
        sys.path.append(vggt_root)
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            print("⚠️  严重警告: 未找到 'VGGT' 模块。请检查路径。")
            VGGT = None
    else:
        print("⚠️  严重警告: 未找到 'VGGT' 模块。")
        VGGT = None

class FusionPlaceModel(nn.Module):
    def __init__(self, 
                 vggt_path=None,   # VGGT 权重路径
                 range_dim=2048,    # VGGT (基于 ViT-S) 输出通常是 384
                 bev_dim=128,      # REIN 输出 128
                 fusion_heads=4):
        super().__init__()
        
        # ======================================================================
        # 步骤 1: 初始化 Range 分支 (VGGT)
        # ======================================================================
        print("🦕 [Step 1] 初始化 Range 分支 (VGGT)...")
        if VGGT is None:
            raise ValueError("无法初始化：缺失 VGGT 代码库。")
            
        self.range_backbone = VGGT()
        
        # 加载 VGGT 权重
        if vggt_path and os.path.exists(vggt_path):
            print(f"   -> 加载 VGGT 权重: {vggt_path}")
            # map_location='cpu' 防止 GPU 显存峰值，后面会随 model.to(device) 转移
            state_dict = torch.load(vggt_path, map_location="cpu")
            # strict=True 保证权重完全匹配
            self.range_backbone.load_state_dict(state_dict, strict=True)
        else:
            print("⚠️  警告: 未提供 VGGT 路径或文件不存在，将使用随机初始化 (仅用于测试)。")

        self.range_dim = range_dim

        # ======================================================================
        # 步骤 2: 初始化 BEV 分支 (REIN)
        # ======================================================================
        print("🏗️  [Step 2] 初始化 BEV 分支 (REIN)...")
        if REIN is not None:
            self.bev_backbone = REIN()
        else:
            raise ValueError("无法初始化：缺失 REIN 模块。")
        
        self.bev_dim = bev_dim

        # ======================================================================
        # 步骤 3: 冻结骨干网络
        # ======================================================================
        print("🧊 [Step 3] 冻结双塔骨干网络参数...")
        
        # 冻结 VGGT
        for p in self.range_backbone.parameters(): p.requires_grad = False
        # 冻结 REIN
        for p in self.bev_backbone.parameters():   p.requires_grad = False

        # ======================================================================
        # 步骤 4: 定义融合层
        # ======================================================================
        print("🔗 [Step 4] 初始化 Cross Attention 融合模块...")
        
        # 4.1 维度对齐 (VGGT 384 -> BEV 128)
        self.range_proj = nn.Linear(self.range_dim, self.bev_dim)
        
        # 4.2 交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.bev_dim, 
                                                num_heads=fusion_heads, 
                                                batch_first=True)
        
        # 4.3 Norm
        self.norm = nn.LayerNorm(self.bev_dim)

    def forward(self, bev_img, range_img):
        """
        Inputs:
            bev_img:   (B, 3, H, W)   BEV 图片
            range_img: (B, 3, H_r, W_r) Range 图片
                       VGGT 推荐输入尺寸为 (70, 518)。
                       如果 dataloader 给的不是这个尺寸，这里会进行插值。
        """
        B = bev_img.shape[0]

        # ----------------------------------------------------------------------
        # 阶段 1: 特征提取
        # ----------------------------------------------------------------------
        
        # === 1.1 Range 分支 (VGGT) ===
        # VGGT 期望输入: (B, S, C, H, W) 其中 S 是序列长度，这里 S=1
        # 首先检查尺寸，VGGT 训练时用的是 (70, 518)，我们最好强制对齐到这个尺寸
        target_h, target_w = 70, 518
        if range_img.shape[-2:] != (target_h, target_w):
            range_img_resized = F.interpolate(range_img, size=(target_h, target_w), 
                                              mode='bilinear', align_corners=False)
        else:
            range_img_resized = range_img

        # 增加序列维度: (B, 3, 70, 518) -> (B, 1, 3, 70, 518)
        range_input_vggt = range_img_resized.unsqueeze(1)
        
        # VGGT 前向传播
        # 注意：我们使用 no_grad 确保不更新 VGGT
        with torch.no_grad():
            # aggregator 返回: (aggregated_tokens_list, patch_start_idx)
            aggregated_tokens_list, patch_start_idx = self.range_backbone.aggregator(range_input_vggt)
            
            # 【关键修改】仿照你的脚本，取倒数第 8 层
            target_tokens = aggregated_tokens_list[-8] 
            
            # 截取 Patch Tokens (去掉前面的特殊 Token)
            # target_tokens shape: (B, S, Total_Tokens, Dim)
            patch_tokens = target_tokens[:, :, patch_start_idx:, :]
            
            # 展平 S 维度 (因为 S=1) -> (B, N_patches, Dim)
            # range_features shape: (B, N, 384)
            range_features = patch_tokens.flatten(1, 2)

        # === 1.2 BEV 分支 (REIN) ===
        # REIN.rem 返回 (feature_map, descriptors)
        # feature_map shape: (B, 128, H_feat, W_feat)
        with torch.no_grad():
            bev_feature_map, _ = self.bev_backbone.rem(bev_img)

        # ----------------------------------------------------------------------
        # 阶段 2: 维度对齐
        # ----------------------------------------------------------------------
        
        # Query (BEV): Flatten -> (B, H*W, 128)
        B, C, H, W = bev_feature_map.shape
        query = bev_feature_map.flatten(2).permute(0, 2, 1)
        
        # Key/Value (Range): Linear -> (B, N_patches, 128)
        kv = self.range_proj(range_features)

        # ----------------------------------------------------------------------
        # 阶段 3: 融合
        # ----------------------------------------------------------------------
        
        # Attention: BEV 查询 Range
        attn_out, _ = self.cross_attn(query=query, key=kv, value=kv)
        
        # 残差连接
        fused_seq = self.norm(query + attn_out)

        # ----------------------------------------------------------------------
        # 阶段 4 & 5: 还原与聚合
        # ----------------------------------------------------------------------
        
        # 还原形状: (B, 128, H, W)
        fused_map = fused_seq.permute(0, 2, 1).view(B, C, H, W)
        
        # NetVLAD 聚合
        global_descriptor = self.bev_backbone.pooling(fused_map)
        
        return F.normalize(global_descriptor, p=2, dim=1)