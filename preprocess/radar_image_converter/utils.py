"""
Utility functions for image processing
"""
import numpy as np


def normalize_image(img, method='minmax', percentile=99):
    """
    Normalize multi-channel image
    
    Args:
        img: (H, W, C) array
        method: 'minmax', 'percentile', or 'standardize'
        percentile: for percentile normalization (default: 99)
        
    Returns:
        normalized image (same shape)
    """
    img_norm = np.zeros_like(img)
    
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        
        if method == 'minmax':
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                img_norm[:, :, c] = (channel - min_val) / (max_val - min_val)
        
        elif method == 'percentile':
            min_val = np.percentile(channel, 100 - percentile)
            max_val = np.percentile(channel, percentile)
            if max_val > min_val:
                img_norm[:, :, c] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
        
        elif method == 'standardize':
            mean = channel.mean()
            std = channel.std()
            if std > 0:
                img_norm[:, :, c] = (channel - mean) / std
    
    return img_norm


def save_visualization(img, save_path, backend='opencv'):
    """
    Save image visualization (first 3 channels as RGB)
    
    Args:
        img: (H, W, C) normalized image [0, 1]
        save_path: output file path (.png/.jpg)
        backend: 'opencv' or 'pil'
    """
    # Convert first 3 channels to RGB
    vis = (img[:, :, :3] * 255).astype(np.uint8)
    
    if backend == 'opencv':
        import cv2
        cv2.imwrite(save_path, vis)
    elif backend == 'pil':
        from PIL import Image
        Image.fromarray(vis).save(save_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")
