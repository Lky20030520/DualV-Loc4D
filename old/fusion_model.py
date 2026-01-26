import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 保持之前的 import 不变
try:
    from REIN import REIN
except ImportError:
    print("⚠️  严重警告: 未找到 'REIN' 模块。")
    REIN = None

# try:
#     from vggt.models.vggt import VGGT
# except ImportError:
#     # 这里省略之前的路径查找逻辑，假设你已经配好了
#     VGGT = None
    
try:
    from vggt.models.vggt import VGGT
except ImportError:
    current_dir = os.getcwd()
    vggt_root = os.path.join(current_dir, "vggt-main")
    if os.path.exists(vggt_root) and vggt_root not in sys.path:
        sys.path.append(vggt_root)
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            VGGT = None
    else:
        VGGT = None

class FusionPlaceModel(nn.Module):
    def __init__(self, 
                 vggt_path=None,   # 接收 range 权重路径
                 bev_path=None,    # 🟢 [新增] 接收 bev 权重路径
                 range_dim=2048,    
                 bev_dim=128,       
                 fusion_heads=4,
                 verbose=True): 
        super().__init__()
        self.verbose = verbose
        
        # =================================================
        # 1. Range 分支 (VGGT)
        # =================================================
        print(f"🦕 [Init] Range 分支 (VGGT)")
        if VGGT is None: raise ValueError("缺失 VGGT 模块")
        self.range_backbone = VGGT()
        
        # 🟢 加载 Range 权重
        if vggt_path and os.path.exists(vggt_path):
            print(f"   -> Loading VGGT weights: {vggt_path}")
            # map_location='cpu' 防止显存不够
            state_dict = torch.load(vggt_path, map_location="cpu", weights_only=False)
            
            # 兼容性处理：如果保存时包含 'state_dict' 键
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # 处理可能的 module. 前缀
            state_dict = self._remove_prefix(state_dict)
            
            # strict=False 防止因为一些无关紧要的 key 不匹配报错
            msg = self.range_backbone.load_state_dict(state_dict, strict=False)
            print(f"   -> VGGT Load Info: {msg}")
        else:
            print(f"⚠️ 警告: 未找到 VGGT 权重文件: {vggt_path}，将使用随机初始化！")
        
        self.range_dim = range_dim

        # =================================================
        # 2. BEV 分支 (REIN)
        # =================================================
        print(f"🏗️  [Init] BEV 分支 (REIN)")
        if REIN is None: raise ValueError("缺失 REIN 模块")
        self.bev_backbone = REIN()
        
        # 🟢 [关键修改] 加载 BEV 权重
        if bev_path and os.path.exists(bev_path):
            print(f"   -> Loading BEV weights: {bev_path}")
            checkpoint = torch.load(bev_path, map_location="cpu", weights_only=False)
            
            # .pth.tar 通常包含 'state_dict', 'epoch' 等信息
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 处理可能的 DataParallel 'module.' 前缀
            state_dict = self._remove_prefix(state_dict)

            # strict=False 允许部分不匹配（只要核心层匹配即可）
            msg = self.bev_backbone.load_state_dict(state_dict, strict=False)
            print(f"   -> BEV Load Info: {msg}")
        else:
            print(f"⚠️ 警告: 未找到 BEV 权重文件: {bev_path}，将使用随机初始化！")

        self.bev_dim = bev_dim

        # =================================================
        # 3. 冻结骨干 (现在冻结是安全的，因为已经加载了权重)
        # =================================================
        print("🧊 [Init] Freezing backbones (with loaded weights)...")
        for p in self.range_backbone.parameters(): p.requires_grad = False
        for p in self.bev_backbone.parameters():   p.requires_grad = False

        # =================================================
        # 4. 融合层
        # =================================================
        print("🔗 [Init] Fusion Layers")
        self.range_proj = nn.Linear(self.range_dim, self.bev_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.bev_dim, 
                                                num_heads=fusion_heads, 
                                                batch_first=True)
        self.norm = nn.LayerNorm(self.bev_dim)

    # 🟢 辅助函数：去掉 module. 前缀
    def _remove_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    def forward(self, bev_img, range_img):
        # 保持之前的 forward 逻辑完全不变
        B = bev_img.shape[0]
        #if self.verbose: print(f"\n--- [Fusion Process Start] Batch Size: {B} ---")

        # [Range]
        target_h, target_w = 70, 518
        if range_img.shape[-2:] != (target_h, target_w):
            range_img = F.interpolate(range_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        range_input = range_img.unsqueeze(1)
        with torch.no_grad():
            agg_list, start_idx = self.range_backbone.aggregator(range_input)
            target_tokens = agg_list[-8]
            patch_tokens = target_tokens[:, :, start_idx:, :]
            range_features = patch_tokens.flatten(1, 2)

        # [BEV]
        with torch.no_grad():
            bev_map, _ = self.bev_backbone.rem(bev_img)

        # [Align & Fuse]
        query = bev_map.flatten(2).permute(0, 2, 1)
        kv = self.range_proj(range_features)
        attn_out, _ = self.cross_attn(query=query, key=kv, value=kv)
        fused_seq = self.norm(query + attn_out)

        # [Aggregate]
        fused_map = fused_seq.permute(0, 2, 1).view(B, self.bev_dim, bev_map.shape[2], bev_map.shape[3])
        global_desc = self.bev_backbone.pooling(fused_map)
        final_desc = F.normalize(global_desc, p=2, dim=1)
        
        return final_desc