import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random


def mask_to_visual(mask_path, output_path=None, colormap='random', show=False):
    """
    将单通道索引掩码图转换为彩色可视化图。

    参数:
        mask_path: str, 掩码图路径（支持中文）
        output_path: str, 输出图片保存路径（若为None则不保存）
        colormap: str or dict, 颜色映射方式。'random'随机生成，或手动传入{类别索引: (R,G,B)}
        show: bool, 是否显示图片
    """
    # 1. 读取掩码图（支持中文路径）
    mask = np.array(Image.open(mask_path))

    # 确保是单通道
    if len(mask.shape) != 2:
        raise ValueError(f"期望单通道图像，实际形状为 {mask.shape}")

    # 2. 获取所有唯一类别
    unique_labels = np.unique(mask)
    print(f"检测到的类别索引: {unique_labels}")

    # 3. 生成颜色映射
    if isinstance(colormap, dict):
        # 使用用户提供的映射
        color_map = colormap
        # 检查是否有缺失的类别
        for label in unique_labels:
            if label not in color_map:
                # 随机生成缺失类别的颜色
                color_map[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                print(f"警告: 类别 {label} 未在colormap中，已随机分配颜色 {color_map[label]}")
    else:
        # 随机生成颜色
        color_map = {}
        # 背景通常设为黑色或白色，可单独处理
        for label in unique_labels:
            if label == 0:
                color_map[label] = (0, 0, 0)  # 背景黑色
            else:
                color_map[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 4. 创建RGB图像
    h, w = mask.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in color_map.items():
        rgb_img[mask == label] = color

    # 5. 保存或显示
    if output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(rgb_img).save(output_path)
        print(f"可视化图已保存至: {output_path}")

    if show:
        plt.imshow(rgb_img)
        plt.axis('off')
        plt.show()

    return rgb_img, color_map


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 示例：替换为您的实际路径
    mask_path = r"d:\data\PV08_350252_1189866.png"
    mask_path= r"d:\data\PV08_331828_1180598.png"
    output_path = r"d:\data\PV08_350252_1189866_vis.png"

    # 先以二进制方式读取文件，再用 cv2.imdecode 解码
    with open(mask_path, 'rb') as f:
        img_data = f.read()
    mask = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)

    print("图像形状:", mask.shape)
    print("数据类型:", mask.dtype)
    print("唯一像素值:", np.unique(mask))
    print("像素值统计:", np.bincount(mask.flatten()))

    # 调用函数
    rgb, colormap = mask_to_visual(mask_path, output_path, colormap='random', show=True)

    # 打印颜色映射关系
    print("类别 -> RGB颜色映射:")
    for label, color in colormap.items():
        print(f"  类别 {label}: {color}")