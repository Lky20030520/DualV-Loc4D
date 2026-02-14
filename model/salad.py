import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Optimal Transport Solvers (Sinkhorn)
# Adapted from OpenGlue
# ==============================================================================

def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem."""
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """Sinkhorn algorithm wrapper to compute matching probabilities."""
    batch_size, m, n = S.size()
    # augment scores matrix with dustbin
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm

# ==============================================================================
# SALAD Module
# ==============================================================================

class SALAD(nn.Module):
    """
    Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.
    Optimized with Matrix Multiplication to avoid OOM.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        
        # MLP for local features f_i (implemented as Conv2d)
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        """
        Args:
            x (tuple): (features, global_token)
                - features: [B, C, H, W]  (Local feature map)
                - global_token: [B, C]    (Global context vector)

        Returns:
            f (torch.Tensor): The global descriptor [B, num_clusters * cluster_dim + token_dim]
        """
        x, t = x # Unpack input tuple

        # 1. Extract Features
        # x shape: [B, C_in, H, W]
        # f shape: [B, C_out, N] where N = H*W
        f = self.cluster_features(x).flatten(2)
        
        # p shape: [B, Num_Clusters, N]
        p = self.score(x).flatten(2)
        
        # t shape: [B, Token_Dim]
        t = self.token_features(t)

        # 2. Sinkhorn Algorithm (Optimal Transport)
        # Calculates soft assignment probabilities
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        
        # Drop the dustbin column (last row in dimension 1)
        # p shape: [B, Num_Clusters, N]
        p = p[:, :-1, :]

        # 3. Weighted Aggregation (Optimized with MatMul)
        # -----------------------------------------------------------
        # Goal: Sum(Feature * Prob) for each cluster
        # Original (OOM risk): (f.unsqueeze * p.unsqueeze).sum()
        # Optimized: Matrix Multiplication
        # -----------------------------------------------------------
        
        # p: [B, K, N]  (K=Num_Clusters)
        # f: [B, C, N]  (C=Cluster_Dim)
        
        # Perform: [B, K, N] @ [B, N, C] = [B, K, C]
        vlad = torch.matmul(p, f.transpose(1, 2))
        
        # To match original normalization logic (along Channel dim):
        # Permute to [B, C, K]
        vlad = vlad.permute(0, 2, 1)

        # 4. Normalization and Concatenation
        # Intra-normalization (normalize feature vector within each cluster)
        vlad = F.normalize(vlad, p=2, dim=1)
        
        # Flatten: [B, C, K] -> [B, C * K]
        vlad = vlad.flatten(1)

        # Normalize Global Token
        t_norm = F.normalize(t, p=2, dim=-1)

        # Concatenate Global Token + VLAD Descriptor
        final_desc = torch.cat([t_norm, vlad], dim=-1)

        # Final L2 Normalization
        return F.normalize(final_desc, p=2, dim=-1)