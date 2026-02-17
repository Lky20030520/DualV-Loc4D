import argparse
from pathlib import Path
import os
import sys

import torch


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


def _load_images(image_paths, mode: str):
    from vggt.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images(image_paths, mode=mode)


def _extract_global_vector(model, images: torch.Tensor) -> torch.Tensor:
    """
    提取 DINO 全局特征向量（用于语义调制）
    
    Returns:
        global_vec: [B, 2048] 全局语义向量
    """
    # Aggregator expects [B, S, 3, H, W]
    images = images.unsqueeze(1)

    with torch.no_grad():
        agg_list, patch_start_idx = model.aggregator(images)

    tokens = agg_list[-1]  # [B, S, P, 2C]
    patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, S, N_patch, 2048]

    # 全局平均池化：[B, S, N_patch, 2048] -> [B, 2048]
    global_vec = patch_tokens.mean(dim=[1, 2])  # [B, 2048]
    
    return global_vec


def main():
    parser = argparse.ArgumentParser(description="Extract VGGT (DINOv2) patch feature maps")
    parser.add_argument("--image", nargs="+", required=True, help="Input image path(s)")
    parser.add_argument(
        "--ckpt",
        default="/home/kaiyan/BEVPlace3/runs/vggt_model/vggtmodel.pt",
        help="Path to vggt checkpoint",
    )
    parser.add_argument("--mode", choices=["crop", "pad"], default="pad")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="", help="Optional .pt output path")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    _add_vggt_to_path(project_root)

    image_paths = [str(Path(p)) for p in args.image]
    images = _load_images(image_paths, mode=args.mode)

    device = torch.device(args.device)
    model = _load_model(Path(args.ckpt), device=device)

    images = images.to(device)
    global_vec = _extract_global_vector(model, images)

    print("images shape:", tuple(images.shape))
    print("global_vec shape:", tuple(global_vec.shape))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_vec": global_vec.cpu(),
                "image_paths": image_paths,
                "mode": args.mode,
            },
            str(out_path),
        )
        print("saved to:", str(out_path))


if __name__ == "__main__":
    main()
