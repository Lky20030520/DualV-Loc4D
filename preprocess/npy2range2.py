import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = r'/home/kaiyan/BEVPlace3/datasets/snail/radar/if_20240116_5/enhanced_rich_npy'
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), "range_image_v2")

# 雷达参数
H_FOV = 113.0
V_FOV = 45.0
IMG_H = 64
IMG_W = 518

# ================= 改进的投影函数 =================
def do_range_projection_v2(points):
    """
    改进版 Range Projection：
    1. 使用 Max Pooling 聚合重复像素
    2. 垂直拉伸增强立面特征
    BGR 映射: Ch0=Doppler, Ch1=Range, Ch2=Power
    """
    # 1. 初始化背景 (ImageNet 均值色，用于填充无数据区域)
    # 通道顺序: BGR
    bg_color = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    image = np.full((IMG_H, IMG_W, 3), bg_color, dtype=np.float32)

    if points is None or len(points) == 0:
        return image

    # 读取数据 [x, y, z, power, doppler]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    power = points[:, 3]
    doppler = points[:, 4] if points.shape[1] >= 5 else np.zeros_like(power)

    # 计算距离 r
    r = torch_r = np.sqrt(x**2 + y**2 + z**2)
    mask = r > 0.5 # 过滤自车近距离噪点
    x, y, z, r, power, doppler = x[mask], y[mask], z[mask], r[mask], power[mask], doppler[mask]

    if len(x) == 0: return image

    # 坐标转换
    yaw = np.arctan2(y, x)
    pitch = np.arcsin(z / r)

    fov_h_rad = np.deg2rad(H_FOV)
    fov_v_rad = np.deg2rad(V_FOV)

    # 投影坐标计算
    u = 0.5 * (1.0 - yaw / (fov_h_rad / 2.0)) * IMG_W
    v = 0.5 * (1.0 - pitch / (fov_v_rad / 2.0)) * IMG_H

    u = np.floor(u).astype(int)
    v = np.floor(v).astype(int)

    # 边界过滤
    mask_valid = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    u, v, power, r, doppler = u[mask_valid], v[mask_valid], power[mask_valid], r[mask_valid], doppler[mask_valid]

    # --- 归一化特征数据 ---
    norm_power = np.clip(power / 70.0, 0, 1)        # 假设最大强度为 70dB
    norm_range = np.clip(r / 100.0, 0, 1)          # 100m 范围
    norm_doppler = np.clip((doppler + 10) / 20.0, 0, 1) # -10~10 m/s 映射到 0~1

    # --- [核心改进] 像素聚合逻辑 (Max Pooling) ---
    # 使用 np.maximum.at 确保 7000 个点中，落在同一像素的特征被最大化保留
    flat_indices = v * IMG_W + u
    
    # 临时展平数组
    ch_b = np.zeros(IMG_H * IMG_W, dtype=np.float32)
    ch_g = np.zeros(IMG_H * IMG_W, dtype=np.float32)
    ch_r = np.zeros(IMG_H * IMG_W, dtype=np.float32)

    np.maximum.at(ch_b, flat_indices, norm_doppler)
    np.maximum.at(ch_g, flat_indices, norm_range)
    np.maximum.at(ch_r, flat_indices, norm_power)

    temp_img = np.stack([
        ch_b.reshape(IMG_H, IMG_W),
        ch_g.reshape(IMG_H, IMG_W),
        ch_r.reshape(IMG_H, IMG_W)
    ], axis=2)

    # --- [核心改进] 垂直列增强 (Vertical Pillar Dilation) ---
    # 使用垂直卷积核，将单点雷达反射“拉伸”成线，模拟物体的垂直立面
    # 这能极大增加图像均值和纹理感
    img_uint8 = (temp_img * 255).astype(np.uint8)
    
    # 垂直核 (5x1)，仅在高度方向膨胀
    v_kernel = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)
    
    img_dilated = cv2.dilate(img_uint8, v_kernel, iterations=1)

    # 融合背景：将雷达信号覆盖到背景色之上
    is_radar = np.max(img_dilated, axis=2) > 5 # 判定是否有信号
    dilated_float = img_dilated.astype(np.float32) / 255.0
    
    # 在背景上覆盖雷达点
    image[is_radar] = dilated_float[is_radar]

    return image

# ================= 主循环 =================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    npy_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))
    print(f"🚀 开始处理 {len(npy_files)} 个 4D 雷达文件...")

    for file_path in tqdm(npy_files, desc="Rendering Range"):
        try:
            data = np.load(file_path, allow_pickle=True)
            if data.ndim == 0: data = data.item()
            
            # 生成增强后的投影图
            proj_img = do_range_projection_v2(data)
            
            # 保存
            img_save = (proj_img * 255).astype(np.uint8)
            base_name = os.path.basename(file_path).replace('.npy', '.png')
            cv2.imwrite(os.path.join(OUTPUT_DIR, base_name), img_save)

        except Exception as e:
            print(f"Error: {file_path} -> {e}")

    print(f"✅ 完成！Range 图像已增强并保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()