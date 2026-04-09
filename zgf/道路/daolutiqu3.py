import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmengine import Config

from mmseg.apis import init_model, inference_model, MMSegInferencer
from mmseg.models.utils import resize as mmseg_resize
from scipy import ndimage
# 解决torch版本过高，与mmseg不兼容的问题
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
# ====================== 0. 辅助函数：解决中文路径读取问题 ======================
def cv_imread(file_path):
    """
    解决 cv2.imread 无法读取中文路径的问题
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    img_data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图片，请检查文件是否损坏: {file_path}")
    return img


# ====================== 1. 骨架化函数 ======================
def skeletonize(img):
    """使用 scipy.ndimage 实现骨架化"""
    img_bin = img > 0
    skel = np.zeros_like(img_bin)
    element = ndimage.generate_binary_structure(2, 1)

    while True:
        eroded = ndimage.binary_erosion(img_bin, structure=element)
        temp = ndimage.binary_dilation(eroded, structure=element)
        temp = img_bin ^ temp
        skel = skel | temp
        img_bin = eroded.copy()
        if not img_bin.any():
            break
    return (skel * 255).astype(np.uint8)


# ====================== 2. 智能断点连接（核心修复逻辑） ======================
def connect_gaps_without_widening(mask, max_gap=8):
    """
    仅连接断点，严格保持原始道路宽度
    """
    # 1. 骨架化（提取中心线）
    skeleton = skeletonize(mask)

    # 2. 闭运算连接短距离断点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
    connected_skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)

    # 3. 与原始掩码 OR 运算：保留原始宽度 + 添加连接线
    final_mask = cv2.bitwise_or(mask, connected_skeleton)

    # 4. 轻微开运算消除毛刺
    if max_gap > 5:
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, clean_kernel)

    return final_mask


# ====================== 3. 获取道路概率图 ======================
def get_road_probability(img_path, target_size=None):
    """
    target_size: (width, height) - 注意OpenCV格式
    """
    result = inference_model(model, img_path)
    seg_logits = result.seg_logits.data

    # 获取类别索引
    if hasattr(model, 'dataset_meta'):
        classes = model.dataset_meta['classes']
        ROAD_CLASS_INDEX = classes.index('road') if 'road' in classes else 1
    else:
        ROAD_CLASS_INDEX = 1

    # 计算概率
    probs = torch.softmax(seg_logits, dim=0)
    road_prob = probs[ROAD_CLASS_INDEX].cpu().numpy()

    # 调整大小以匹配原始图像
    if target_size is not None:
        # 使用 mmseg 的 resize 工具，注意它通常接受 (H, W) 或 (W, H) 取决于实现
        # 这里为了保险，直接使用 cv2 处理 numpy 数组
        road_prob_resized = cv2.resize(road_prob, target_size, interpolation=cv2.INTER_LINEAR)
        return road_prob_resized

    return road_prob


# ====================== 4. 多阈值修复 ======================
def repair_road_mask_multi_scale(prob_map, thresholds=[0.1, 0.3, 0.5], kernel_sizes=[3, 5, 7]):
    masks = []
    for th, ks in zip(thresholds, kernel_sizes):
        bin_mask = (prob_map > th).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        closed_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        masks.append(closed_mask)
    return np.maximum.reduce(masks)


# ====================== 5. 可视化 ======================
def visualize_results(original_img, prob_map, repaired_mask, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Road Probability Map')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(repaired_mask, cmap='gray', vmin=0, vmax=255)
    plt.title('Connected Road Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.jpg', bbox_inches='tight', dpi=150)
    plt.close()
    cv2.imwrite(f'{output_dir}/repaired_mask.png', repaired_mask)


# ====================== 6. 主流程 ======================
if __name__ == '__main__':
    # 参数配置
    config_file = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\road3.py'
    checkpoint_file = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\best_mIoU_iter_424156.pth'
    img_path = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\ceshitupian\2019_human2-2.png'
    output_dir = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\save_ceshi'

    # 初始化模型
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(config_file, checkpoint_file, device=device)

    # 1. 读取图像 (使用修复后的中文路径读取函数)
    original_img = cv_imread(img_path)
    if original_img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取原始尺寸 (宽, 高) -> 注意 OpenCV 的 shape 是 (高, 宽, 通道)
    h, w = original_img.shape[:2]
    target_size = (w, h)  # 这里定义为 (宽, 高) 以适配 cv2.resize

    print(f"原始图像尺寸: 宽={w}, 高={h}")

    # 2. 获取概率图
    road_prob = get_road_probability(img_path, target_size=target_size)
    print(f"概率图尺寸: {road_prob.shape}")

    # 3. 后处理流程
    # 3.1 多阈值修复
    thresholds = [0.1, 0.25, 0.4]
    kernel_sizes = [3, 5, 7]
    repaired_mask = repair_road_mask_multi_scale(road_prob, thresholds, kernel_sizes)

    # 3.2 智能断点连接
    MAX_GAP = 8
    repaired_mask = connect_gaps_without_widening(repaired_mask, max_gap=MAX_GAP)

    # 4. 关键修复：确保掩码尺寸与原图完全一致
    # 检查尺寸
    if repaired_mask.shape != original_img.shape[:2]:
        print(f"尺寸不匹配！正在修正掩码尺寸: {repaired_mask.shape} -> {original_img.shape[:2]}")
        repaired_mask = cv2.resize(repaired_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 5. 可视化与保存
    visualize_results(original_img, road_prob, repaired_mask, output_dir)

    # 6. 生成叠加图 (确保这里不再报错)
    overlay_img = original_img.copy()
    # 只有当掩码尺寸正确时，这一步才会成功
    overlay_img[repaired_mask > 0] = [0, 0, 255]  # 红色标记道路

    alpha = 0.4
    blended = cv2.addWeighted(original_img, 1 - alpha, overlay_img, alpha, 0)
    cv2.imwrite(f'{output_dir}/overlay_result.jpg', blended)

    # 保存热力图
    plt.imsave(f'{output_dir}/road_prob_heatmap.png', road_prob, cmap='jet', vmin=0, vmax=1)

    print(f"✅ 处理完成! 结果保存至: {output_dir}")