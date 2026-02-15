import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = r'/home/kaiyan/BEVPlace3/datasets/snail_radar_acum/bc/20230920_1_preprocessed_accm7/pointclouds'
# 输出文件夹自动创建在同级目录
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), "range_image")

# 雷达参数
H_FOV = 113.0
V_FOV = 45.0
IMG_H = 64
IMG_W = 518

# 数据范围参数（与BEV和pcs_preprocess.py保持一致）
MAX_RANGE = 120.0   # 与pcs_preprocess.py的maximum_range一致
Z_MIN = -13.0       # 实际NPY数据范围: [-13.24, 38.06]
Z_MAX = 38.0
POWER_MIN = 5.0     # 实际NPY数据范围: [5.08, 32.18]
POWER_MAX = 32.0

# ================= 核心投影函数 =================
def do_range_projection_visual(points):
    """
    生成 Range Projection 图像
    
    【重要】拖尾效果说明：
    - NPY数据来自 accum_win=7 的多帧堆叠（约0.7秒，7帧合并）
    - 虽然GPS已对齐到中心帧坐标系，但7帧内物体的运动轨迹仍被保留
    - 这是正常现象，提供了隐式的运动信息，有助于定位
    - 如需去除拖尾，需重新生成NPY时使用 accum_win=1（单帧模式）
    
    输出 BGR 通道对应:
      Ch0 (Blue):  Height/Z (高度，-13~38米)
      Ch1 (Green): Range (距离，0~120米)
      Ch2 (Red):   Intensity/Power (强度，5~32)
    """
    # 1. 初始化背景 (使用 ImageNet 均值灰)
    image = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
    image[:, :, 0] = 0.406 # B: Height background
    image[:, :, 1] = 0.456 # G: Range background
    image[:, :, 2] = 0.485 # R: Power background

    if points is None or len(points) == 0:
        return image

    # 读取数据 [x, y, z, power, doppler]
    if points.shape[1] >= 4:
        x, y, z, power = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    else:
        # 数据格式不对
        return image

    # 坐标系转换与投影
    r = np.sqrt(x**2 + y**2 + z**2)
    mask = r > 0.5
    x, y, z, r, power = x[mask], y[mask], z[mask], r[mask], power[mask]

    if len(x) == 0: return image

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
    u, v = u[mask_valid], v[mask_valid]
    power, r, z = power[mask_valid], r[mask_valid], z[mask_valid]

    # 2. 准备绘制层 (临时画布)
    temp_img = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
    
    # --- 归一化通道数据 ---
    
    # Power (强度) -> 对应 Red
    norm_power = np.clip((power - POWER_MIN) / (POWER_MAX - POWER_MIN), 0, 1)
    
    # Range (距离) -> 对应 Green
    norm_range = np.clip(r / MAX_RANGE, 0, 1)
    
    # Height (高度Z) -> 对应 Blue
    norm_height = np.clip((z - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)

    # --- 填入通道 (BGR 顺序) ---
    temp_img[v, u, 0] = norm_height  # Blue: 高度
    temp_img[v, u, 1] = norm_range   # Green: 距离
    temp_img[v, u, 2] = norm_power   # Red: 强度

    # 3. 形态学膨胀 (Dilation)
    # 让稀疏点变大，利于网络提取特征
    img_uint8 = (temp_img * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8) 
    img_dilated = cv2.dilate(img_uint8, kernel, iterations=1)

    # 4. 融合背景
    # 生成掩码：只要任意通道有值，就认为是前景
    is_radar = np.max(img_uint8, axis=2) > 10
    
    # 将膨胀后的雷达点覆盖到背景上
    dilated_float = img_dilated.astype(np.float32) / 255.0
    image[is_radar] = dilated_float[is_radar]

    return image

# ================= 主程序 =================
def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 输入文件夹不存在 -> {INPUT_DIR}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 已创建输出文件夹 -> {OUTPUT_DIR}")

    npy_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))
    total_files = len(npy_files)
    print(f"🚀 发现 {total_files} 个 .npy 文件，准备开始处理...")

    for file_path in tqdm(npy_files, desc="Processing"):
        try:
            # 加载数据
            data = np.load(file_path, allow_pickle=True)
            if data.ndim == 0: data = data.item()
            
            # 生成投影图 (RGB Float 0-1)
            # 注意：do_range_projection_visual 内部虽然按 BGR 逻辑填了值，
            # 但它返回的是 float32 的 0-1 矩阵。
            # OpenCV imwrite 把它当做 BGR 处理时，通道顺序就是我们设定好的。
            proj_img = do_range_projection_visual(data)
            
            # 转换为 0-255 uint8 以便保存Height, Range, Power]
            img_save = (proj_img * 255).astype(np.uint8)
            
            # 构造输出文件名
            base_name = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(base_name)[0]
            save_path = os.path.join(OUTPUT_DIR, f"{file_name_no_ext}.png")
            
            # 保存 (cv2.imwrite 默认按照 BGR 顺序写入文件)
            # 结果 png 里的通道：Ch0=HeightBGR 顺序写入文件)
            # 结果 png 里的通道：Ch0=Speed, Ch1=Range, Ch2=Power
            cv2.imwrite(save_path, img_save)

        except Exception as e:
            print(f"\n⚠️ 处理失败: {os.path.basename(file_path)} -> {e}")

    print(f"\n✅ 全部完成！图片已保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()