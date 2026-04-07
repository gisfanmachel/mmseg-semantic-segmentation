import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmengine import Config

from mmseg.apis import init_model, inference_model, MMSegInferencer
from mmseg.structures import SegDataSample
from mmseg.models.utils import resize
from scipy import ndimage  # 仅使用标准库已导入的模块
# 解决torch版本过高，与mmseg不兼容的问题
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ====================== 1. 模型初始化 ======================
config_file = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\road3.py'
checkpoint_file = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\best_mIoU_iter_424156.pth'

model = init_model(config_file, checkpoint_file, device=device)


# ====================== 2. 骨架化函数（纯 scipy 实现） ======================
def skeletonize(img):
    """使用 scipy.ndimage 实现骨架化，无需 OpenCV Contrib"""
    img_bin = img > 0
    skel = np.zeros_like(img_bin)
    element = ndimage.generate_binary_structure(2, 1)  # 8-邻域结构元

    while True:
        eroded = ndimage.binary_erosion(img_bin, structure=element)
        temp = ndimage.binary_dilation(eroded, structure=element)
        temp = img_bin ^ temp  # 边界像素
        skel = skel | temp
        img_bin = eroded.copy()
        if not img_bin.any():
            break
    return (skel * 255).astype(np.uint8)


# ====================== 3. 智能断点连接（关键：不增宽） ======================
def connect_gaps_without_widening(mask, max_gap=8):
    """
    仅连接断点，严格保持原始道路宽度
    """
    # 1. 骨架化（提取中心线）
    skeleton = skeletonize(mask)

    # 2. 小核闭运算连接短距离断点（关键：核大小 <= max_gap）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
    connected_skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)

    # 3. 与原始掩码 OR 运算：保留原始宽度 + 添加连接线
    final_mask = cv2.bitwise_or(mask, connected_skeleton)

    # 4. （可选）轻微开运算消除毛刺（不改变主体宽度）
    if max_gap > 5:
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, clean_kernel)

    return final_mask


# ====================== 4. 获取道路概率图 ======================
def get_road_probability(img_path, original_size=None):
    result = inference_model(model, img_path)
    seg_logits = result.seg_logits.data

    if hasattr(model, 'dataset_meta'):
        classes = model.dataset_meta['classes']
        ROAD_CLASS_INDEX = classes.index('road') if 'road' in classes else 1

    probs = torch.softmax(seg_logits, dim=0)
    road_prob = probs[ROAD_CLASS_INDEX].cpu().numpy()

    if original_size is not None:
        road_prob_tensor = torch.from_numpy(road_prob)[None, None, ...]
        road_prob_tensor = resize(
            road_prob_tensor,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        road_prob = road_prob_tensor[0, 0].numpy()

    return road_prob


# ====================== 5. 多阈值修复 ======================
def repair_road_mask_multi_scale(prob_map, thresholds=[0.1, 0.3, 0.5], kernel_sizes=[3, 5, 7]):
    masks = []
    for th, ks in zip(thresholds, kernel_sizes):
        bin_mask = (prob_map > th).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        closed_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        masks.append(closed_mask)
    return np.maximum.reduce(masks)  # 修复：使用 reduce 处理任意数量掩码


# ====================== 6. 可视化 ======================
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
    plt.title('Connected Road Mask (No Width Increase)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.jpg', bbox_inches='tight', dpi=150)
    plt.close()
    cv2.imwrite(f'{output_dir}/repaired_mask.png', repaired_mask)


# ====================== 7. 主流程 ======================
if __name__ == '__main__':
    # 参数配置（关键：max_gap 控制连接距离）
    thresholds = [0.1, 0.25, 0.4]  # 降低阈值捕获弱信号
    kernel_sizes = [3, 5, 7]
    MAX_GAP = 8  # 仅连接 <=8像素的断点（避免误连）

    img_path = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\ceshitupian\2019_human2-2.png'
    output_dir = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\save_ceshi'
    IMAGE_DIR = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\道路\ceshitupian'
    CLASSES = ["background", "road"]  # 类别顺序：0=背景，1=道路
    PALETTE = [[255, 255, 255], [0, 0, 255]]

    BATCH_SIZE = 1
    RESIZE_SCALE = (512, 512)
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ===================== 步骤1：加载并修改配置文件 =====================
    cfg = Config.fromfile(config_file)

    # 修改测试阶段的预处理管道：添加 Resize 步骤（在 LoadImageFromFile 之后）
    load_img_idx = None
    for idx, transform in enumerate(cfg.test_pipeline):
        if transform['type'] == 'LoadImageFromFile':
            load_img_idx = idx
            break

    if load_img_idx is not None:
        cfg.test_pipeline.insert(
            load_img_idx + 1,
            dict(type='Resize', scale=RESIZE_SCALE, keep_ratio=False)
        )

    # ===================== 步骤2：初始化推理器（确保调色板生效） =====================
    # 方法1：直接默认提取
    inferencer = MMSegInferencer(
        model=cfg,
        weights=checkpoint_file,
        classes=CLASSES,  # 强制指定类别
        palette=PALETTE,  # 强制指定调色板（覆盖配置文件默认值）
        device=DEVICE
    )
    # ===================== 步骤3：获取测试图片列表 =====================
    image_list = [
                     os.path.join(IMAGE_DIR, img_name)
                     for img_name in os.listdir(IMAGE_DIR)
                     if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
                 ][:200]

    if not image_list:
        raise FileNotFoundError(f"错误：在 {IMAGE_DIR} 中未找到任何图片！")

    print(f"找到 {len(image_list)} 张测试图片，开始分批次分割...")

    # ===================== 步骤4：分批次推理 =====================
    os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    autocast_context = torch.cuda.amp.autocast() if DEVICE == 'cuda:0' else nullcontext()

    for batch_idx in range(0, len(image_list), BATCH_SIZE):
        batch_imgs = image_list[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"\n处理第 {batch_num} 批，共 {len(batch_imgs)} 张图...")

        with autocast_context:
            inferencer(
                batch_imgs,
                batch_size=BATCH_SIZE,
                out_dir=output_dir,
                show=False,
                wait_time=0,
                img_out_dir='vis',
                pred_out_dir='mask',
                return_vis=False,
                return_datasamples=False,
                with_labels=False
            )

        if DEVICE == 'cuda:0':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



    # 方法2，用底层api提取，并做后处理
    # 读取图像
    # original_img = cv2.imread(img_path)

    # 先以二进制方式读取文件，再用 cv2.imdecode 解码
    with open(img_path, 'rb') as f:
        img_data = f.read()
    original_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if original_img is None:
        raise FileNotFoundError(f"图像不存在: {img_path}")

    original_size = (original_img.shape[1], original_img.shape[0])
    print(f"原始图像尺寸: {original_size}")

    # 获取概率图
    road_prob = get_road_probability(img_path, original_size=original_size)
    print(f"概率图尺寸: {road_prob.shape}")

    # 多阈值修复
    repaired_mask = repair_road_mask_multi_scale(road_prob, thresholds, kernel_sizes)

    # 智能断点连接（核心改进）
    repaired_mask = connect_gaps_without_widening(repaired_mask, max_gap=MAX_GAP)

    # 尺寸对齐（防止OpenCV尺寸问题）
    repaired_mask = cv2.resize(repaired_mask, (original_img.shape[1], original_img.shape[0]))

    # 保存结果
    visualize_results(original_img, road_prob, repaired_mask, output_dir)

    # 生成叠加图
    overlay_img = original_img.copy()
    overlay_img[repaired_mask > 0] = [0, 0, 255]  # 红色标记道路
    alpha = 0.4
    blended = cv2.addWeighted(original_img, 1 - alpha, overlay_img, alpha, 0)
    cv2.imwrite(f'{output_dir}/overlay_result.jpg', blended)

    # 保存热力图
    plt.imsave(f'{output_dir}/road_prob_heatmap.png', road_prob, cmap='jet', vmin=0, vmax=1)

    print(f"✅ 处理完成! 结果保存至: {output_dir}")
    print(f"   - 断点连接距离: ≤{MAX_GAP} 像素")
    print(f"   - 道路宽度: 严格保持原始掩码宽度")