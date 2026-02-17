import argparse
from pathlib import Path
import sys
import os

import torch
import torch.nn.functional as F
import numpy as np


def _add_vggt_to_path(project_root: Path) -> None:
    vggt_root = project_root / "vggt-main"
    if vggt_root.exists() and str(vggt_root) not in sys.path:
        sys.path.append(str(vggt_root))


def _load_model(ckpt_path: Path, device: torch.device):
    from vggt.models.vggt import VGGT

    model = VGGT()
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_image_simple(image_path: str):
    """Load single image without preprocessing, just read and normalize."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((518, 518), Image.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]
    
    # Normalize (ResNet std)
    img_tensor[0] = (img_tensor[0] - 0.485) / 0.229
    img_tensor[1] = (img_tensor[1] - 0.456) / 0.224
    img_tensor[2] = (img_tensor[2] - 0.406) / 0.225
    
    return img_tensor.unsqueeze(0)  # [1, C, H, W]


def _extract_feature_simple(model, image_path: str, device: torch.device):
    """
    Extract feature from single image with intelligent dimensionality reduction.
    
    提取逻辑：
    1. 图像过 Aggregator -> 得到多层 tokens
    2. 提取 patch tokens: [1, 1, N_patch, 2048]
    3. Reshape 成空间形式: [1, 2048, 37, 37]
    4. Spatial 池化: [1, 2048, 37, 37] -> [1, 2048, 10, 10]
    5. Channel 投影: [1, 2048, 10, 10] -> [1, 256, 10, 10]
    6. 拉平: [1, 256, 10, 10] -> [25600]
    """
    img = _load_image_simple(image_path).to(device)
    
    with torch.no_grad():
        # Add sequence dimension: [1, 1, C, H, W]
        img = img.unsqueeze(1)
        agg_list, patch_start_idx = model.aggregator(img)
        
        # 取最后一层: [1, 1, P, 2C]
        tokens = agg_list[-1]
        
        # 提取纯 patch tokens
        patch_tokens = tokens[:, :, patch_start_idx:, :]  # [1, 1, N_patch, 2048]
        
        # Reshape 成空间形式: [1, 1, N_patch, 2048] -> [1, 37, 37, 2048] -> [1, 2048, 37, 37]
        bsz, seq, n_patch, dim = patch_tokens.shape
        h_p = w_p = int(n_patch ** 0.5)  # 37
        
        spatial = patch_tokens.view(bsz, seq, h_p, w_p, dim)  # [1, 1, 37, 37, 2048]
        spatial = spatial.squeeze(1).permute(0, 3, 1, 2)  # [1, 2048, 37, 37]
        
        # Spatial avg pooling: [1, 2048, 37, 37] -> [1, 2048, 10, 10]
        spatial = F.avg_pool2d(spatial, kernel_size=4, stride=4)  # ~37/4 ≈ 9-10
        
        # Channel 投影并降维: [1, 2048, 10, 10] -> [1, 256, 10, 10]
        # 用简单的平均方式: 2048 -> 256 (分成 8 组取平均)
        bs, c, h, w = spatial.shape
        spatial = spatial.view(bs, 256, 8, h, w).mean(dim=2)  # [1, 256, 10, 10]
        
        # 拉平: [1, 256, 10, 10] -> [25600]
        feature = spatial.reshape(-1)  # [25600]
        feature = F.normalize(feature, p=2, dim=0).unsqueeze(0)
    
    return feature.cpu().squeeze(0)  # [25600]


def _collect_images(dir_path: str, max_images: int = None) -> list:
    """Collect all .jpg images from directory."""
    dir_path = Path(dir_path)
    images = sorted(dir_path.glob("*.jpg"))
    if max_images:
        images = images[:max_images]
    return [str(p) for p in images]


def main():
    parser = argparse.ArgumentParser(description="Simple DINO feature validation")
    parser.add_argument("--same_scene_dir", required=True, help="Same-scene images directory")
    parser.add_argument("--diff_scene_dirs", nargs="+", required=True, help="Different-scene directories")
    parser.add_argument("--ckpt", default="/home/kaiyan/BEVPlace3/runs/vggt_model/vggtmodel.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_images", type=int, default=10, help="Max images per dir")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    _add_vggt_to_path(project_root)

    device = torch.device(args.device)
    model = _load_model(Path(args.ckpt), device=device)

    # Load same-scene
    same_paths = _collect_images(args.same_scene_dir, max_images=args.max_images)
    print(f"Extracting {len(same_paths)} same-scene images...")
    same_feats = []
    for path in same_paths:
        feat = _extract_feature_simple(model, path, device)
        same_feats.append(feat)
        print(f"  [{len(same_feats)}] {Path(path).name}")

    # Load different-scene
    diff_feats = []
    for dir_path in args.diff_scene_dirs:
        paths = _collect_images(dir_path, max_images=args.max_images)
        print(f"Extracting {len(paths)} images from {Path(dir_path).name}...")
        for path in paths:
            feat = _extract_feature_simple(model, path, device)
            diff_feats.append(feat)
            print(f"  [{len(diff_feats)}] {Path(path).name}")

    # Compute similarities
    print("\n" + "="*60)
    same_sims = []
    for i in range(len(same_feats)):
        for j in range(i+1, len(same_feats)):
            sim = torch.dot(same_feats[i], same_feats[j]).item()
            same_sims.append(sim)

    diff_sims = []
    for i in range(len(same_feats)):
        for j in range(len(diff_feats)):
            sim = torch.dot(same_feats[i], diff_feats[j]).item()
            diff_sims.append(sim)

    # Print results
    print("SAME-SCENE similarities:")
    print(f"  Count: {len(same_sims)}")
    print(f"  Mean:  {np.mean(same_sims):.4f}")
    print(f"  Std:   {np.std(same_sims):.4f}")
    print(f"  Range: [{np.min(same_sims):.4f}, {np.max(same_sims):.4f}]")

    print("\nDIFFERENT-SCENE similarities:")
    print(f"  Count: {len(diff_sims)}")
    print(f"  Mean:  {np.mean(diff_sims):.4f}")
    print(f"  Std:   {np.std(diff_sims):.4f}")
    print(f"  Range: [{np.min(diff_sims):.4f}, {np.max(diff_sims):.4f}]")

    gap = np.mean(same_sims) - np.mean(diff_sims)
    print(f"\n▶ Gap (Same - Different): {gap:.4f}")
    
    if gap > 0.15:
        print("  ✅ EXCELLENT! Strong discriminability.")
    elif gap > 0.10:
        print("  ✅ GOOD! Clear separation.")
    elif gap > 0.05:
        print("  ⚠️  MODERATE. Some discriminability.")
    else:
        print("  ❌ POOR. Features not discriminative.")
    print("="*60)


if __name__ == "__main__":
    main()
