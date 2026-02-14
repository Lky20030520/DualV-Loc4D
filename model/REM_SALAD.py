import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torchvision.models as models

from model.salad import SALAD

class REM(nn.Module):
    def __init__(self, from_scratch=False, rotations=8):
        super(REM, self).__init__()
        
        # cnn backbone
        pretrain = not from_scratch
        encoder = models.resnet34(pretrained=pretrain) #resnet34
        layers = list(encoder.children())[:-4]
        self.encoder = nn.Sequential(*layers)

        # rotations
        self.angles = -torch.arange(0,359.00001,360.0/rotations)/180*torch.pi

    
    def forward(self, x):
        
        equ_features = []
        
        batch_size = x.size(0)

        for i in range(len(self.angles)):

            # input warp grids
            aff = torch.zeros(batch_size,2,3).to(device=x.device)
            aff[:,0,0]=torch.cos(-self.angles[i])
            aff[:,0,1]=torch.sin(-self.angles[i])
            aff[:,1,0]=-torch.sin(-self.angles[i])
            aff[:,1,1]=torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, torch.Size(x.size()),align_corners=True).type(x.type())
            
            # input warp
            warped_im = F.grid_sample(x, grid,align_corners=True,mode='bicubic')
                                    
            # cnn backbone feature
            out = self.encoder(warped_im) 

            # output feature warp grids           
            if i==0:
                im1_init_size = out.size()

            aff = torch.zeros(batch_size,2,3).to(device=x.device)
            aff[:,0,0]=torch.cos(self.angles[i])
            aff[:,0,1]=torch.sin(self.angles[i])
            aff[:,1,0]=-torch.sin(self.angles[i])
            aff[:,1,1]=torch.cos(self.angles[i])
            grid = F.affine_grid(aff, torch.Size(im1_init_size),align_corners=True).type(x.type())

            # output feature warp    
            out = F.grid_sample(out, grid ,align_corners=True,mode='bicubic')

            equ_features.append(out.unsqueeze(-1))
        

        equ_features = torch.cat(equ_features, axis=-1)  # B C H W R

        B, C, H, W, R = equ_features.shape
        equ_features=torch.max(equ_features,dim=-1,keepdim=False)[0] # max pooling along rotations

        aff = torch.zeros(batch_size,2,3).to(device=x.device)
        aff[:,0,0]=1
        aff[:,0,1]=0
        aff[:,1,0]=0
        aff[:,1,1]=1

        
        # upsample for NetVLAD
        B,C,H,W = x.size()
        grid = F.affine_grid(aff, torch.Size((B, C, H//4, W//4)),align_corners=True).type(x.type())#,align_corners=True)
        out1 = F.grid_sample(equ_features, grid,align_corners=True,mode='bicubic')
        out1 = F.normalize(out1, dim=1)
        
        # upsample for keypoints
        grid = F.affine_grid(aff, torch.Size((B, C, H, W)),align_corners=True).type(x.type())#,align_corners=True)
        out2 = F.grid_sample(equ_features, grid,align_corners=True,mode='bicubic')
        out2 = F.normalize(out2, dim=1)
        
        return out1, out2

# 修改后的 REINS 类
class REINS(nn.Module):
    def __init__(self):
        super(REINS, self).__init__()
        self.rem = REM() # 保持不变
        
        # [修改] 替换 NetVLAD 为 SALAD
        # 注意：num_channels 必须匹配 REM 输出的维度 (128)
        self.pooling = SALAD(
            num_channels=128, 
            num_clusters=64, 
            cluster_dim=128, 
            token_dim=256
        )

    def forward(self, x):
        out1, local_feats = self.rem(x)
        
        # [修改] 构造全局 Token (GAP) 以适配 SALAD
        # out1 shape: [B, 128, H/4, W/4] -> token: [B, 128]
        global_token = torch.mean(out1, dim=[2, 3]) 
        
        # 传入元组 (feature_map, global_token)
        global_desc = self.pooling((out1, global_token))

        return out1, local_feats, global_desc