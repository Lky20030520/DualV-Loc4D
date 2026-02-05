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
from REIN import REIN
from vggt.models.vggt import VGGT

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
                 freeze_backbones=False): 
        super().__init__()
        
        # -------------------------------------------------------
        # A. Range 分支
        # -------------------------------------------------------
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
        print(f"🏗️  [Init] Loading REIN (BEV)...")
        self.bev_backbone = REIN()
        
        if bev_path and os.path.exists(bev_path):
                    ckpt = torch.load(bev_path, map_location="cpu", weights_only=False)
                    if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
                    
                    # --- 核心修复：鲁棒的前缀处理 ---
                    new_state_dict = {}
                    for k, v in ckpt.items():
                        # 1. 尝试去掉 'module.' (多卡训练产生)
                        name = k.replace('module.', '')
                        # 2. 尝试去掉 'bev_backbone.' (之前的 Fusion 包装产生)
                        name = name.replace('bev_backbone.', '')
                        new_state_dict[name] = v

                    msg = self.bev_backbone.load_state_dict(new_state_dict, strict=False)
                    print(f"✅ BEV Weights Loaded: {bev_path}")
                    if len(msg.missing_keys) > 0:
                        print(f"ℹ️  注：BEV 部分缺失键 (如聚类中心): {msg.missing_keys[:3]}...")
            
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
        self.norm_bev_pre = nn.LayerNorm(self.bev_dim)
        self.norm_range_pre = nn.LayerNorm(self.bev_dim)
        self.norm = nn.LayerNorm(self.bev_dim)

    def _remove_prefix(self, state_dict):
        return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    def forward(self, bev_img, range_img):
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

        bev_map, _ = self.bev_backbone.rem(bev_img)
            
        b, c, h, w = bev_map.shape
        query = bev_map.flatten(2).permute(0, 2, 1)
        
        query_norm = self.norm_bev_pre(query)
        kv_norm = self.norm_range_pre(kv)

        attn_out, _ = self.cross_attn(query=query_norm, key=kv_norm, value=kv_norm)
        fused_seq = self.norm(query_norm + attn_out)

        fused_map = fused_seq.permute(0, 2, 1).view(b, c, h, w)
        global_desc = self.bev_backbone.pooling(fused_map)
        final_desc = F.normalize(global_desc, p=2, dim=1)
        
        return final_desc

    # ==========================================================================
    # [修改] 适配 bev_main.py 的调用接口 (增加 return_all 参数)
    # ==========================================================================
    def forward_bev_only(self, bev_img, return_all=False):
        """
        跳过 VGGT 和 Fusion 层，仅使用 REIN (BEV Backbone) 提取特征。
        支持 return_all=True 以兼容训练循环的解包需求。
        """
        # with torch.no_grad():
        # 1. 提取特征图 (Local Features)
        bev_map, _ = self.bev_backbone.rem(bev_img)
        
        # 2. Pooling (NetVLAD / GeM) -> Global Descriptor
        global_desc = self.bev_backbone.pooling(bev_map)
        
        # 3. 归一化
        final_desc = F.normalize(global_desc, p=2, dim=1)
            
        if return_all:
            # 训练循环期望返回 3 个值: (local_feats, vlad_feats, global_desc)
            # 我们这里为了兼容，前两个返回 bev_map 和 None
            return bev_map, None, final_desc
            
        return final_desc