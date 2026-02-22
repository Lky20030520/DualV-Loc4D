import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RangeREM(nn.Module):
    def __init__(self, rotations=8):
        super(RangeREM, self).__init__()
        
        # 1. 自动加载 ImageNet 预训练权重
        encoder = models.resnet34(pretrained=True)
        
        # 2. 截断网络 (与 REM 对齐)
        # REM 的截断逻辑是 list(encoder.children())[:-4]
        # 这意味着去掉了: fc, avgpool, layer4, layer3
        # 只保留到 layer2 (输出 128 通道)
        layers = list(encoder.children())[:-4] 
        self.encoder = nn.Sequential(*layers)
        
        self.num_shifts = rotations
        self.out_channels = 128  # 明确输出维度

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        equ_features = []
        
        for i in range(self.num_shifts):
            # # 计算平移量
            # shift = int(W * (i / self.num_shifts))
            
            # # 1. 前向变换 (平移)
            # warped_im = torch.roll(x, shifts=shift, dims=-1)
            
            # 2. 提取特征
            out = self.encoder(x) 
            
            # 3. 逆向变换 (对齐)
            _, _, h_feat, w_feat = out.shape
            # feat_shift = int(w_feat * (i / self.num_shifts))
            # out = torch.roll(out, shifts=-feat_shift, dims=-1)
            
            equ_features.append(out.unsqueeze(-1))

        # 4. 聚合 (Max Pooling)
        equ_features = torch.cat(equ_features, axis=-1)
        out_max = torch.max(equ_features, dim=-1)[0] 
        
        # 5. 上采样 (Upsample) - 为了与 REM 的行为完全一致
        # REM 在输出前做了两次 grid_sample 来调整分辨率
        # RangeREM 也简单做一下上采样，保证返回两组特征 (Low-Res, High-Res)
        # out1 (用于 NetVLAD/SALAD): 下采样版
        # out2 (用于可视化/关键点): 全分辨率版 (可选)
        
        # REM 的 out1 是 H//4, RangeREM 这里是 H//8 (因为Layer2是Stride=8)
        # 为了逻辑统一，我们直接返回特征图即可，主模型那边会处理
        
        # return out_max, out_max # 返回两次是为了兼容 REIN.rem 的接口 (out1, out2)
    
    
        # 在 forward 的最后，return 前加：
        B, C, H_feat, W_feat = out_max.shape
        # out_max 现在是 [B, 128, H/8, W/8]

        # Upsample 到 H/4（与 REM 保持一致）
        out_max_upsample = F.interpolate(
            out_max, 
            size=(H_feat * 2, W_feat * 2),  # H/8 * 2 = H/4
            mode='bilinear', 
            align_corners=False
        )

        return out_max_upsample, out_max_upsample