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
from mmseg.utils import get_palette

# 解决torch版本兼容问题
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ====================== 1. 模型初始化 ======================
config_file = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\水系\shuixiconfig_2.py'
checkpoint_file = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\水系\best_mIoU_iter_1207120.pth'

model = init_model(config_file, checkpoint_file, device=device)

# ====================== 2. 获取概率图函数 ======================
def get_water_probability(img_path, original_size=None):
    """
    获取水体类别的概率图 (H, W)

    参数:
        img_path: 输入图像路径
        original_size: 原始图像尺寸 (H, W) —— 注意顺序为 (height, width)
    """
    result = inference_model(model, img_path)
    seg_logits = result.seg_logits.data  # [num_classes, H, W]

    # 获取水体类别索引
    if hasattr(model, 'dataset_meta'):
        classes = model.dataset_meta['classes']
        WATER_CLASS_INDEX = classes.index('water') if 'water' in classes else 1
    else:
        WATER_CLASS_INDEX = 1  # 默认假设水体为第二类

    # 转换为概率
    probs = torch.softmax(seg_logits, dim=0)  # [num_classes, H, W]
    water_prob = probs[WATER_CLASS_INDEX].cpu().numpy()  # [H, W]

    # 上采样到原始图像尺寸
    if original_size is not None:
        water_prob_tensor = torch.from_numpy(water_prob)[None, None, ...]  # [1,1,H,W]
        water_prob_tensor = resize(
            water_prob_tensor,
            size=original_size,          # original_size应为 (height, width)
            mode='bilinear',
            align_corners=False
        )
        water_prob = water_prob_tensor[0, 0].numpy()

    return water_prob

# ====================== 3. 水道修复函数 ======================
def repair_water_mask(prob_map, threshold=0.3, kernel_size=3):
    """修复断裂水道：二值化 + 形态学闭运算"""
    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask

# ====================== 4. 可视化对比工具 ======================
def visualize_results(original_img, prob_map, repaired_mask, threshold, kernel_size, output_dir='./results'):
    """保存对比结果"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Water Probability Map')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(repaired_mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Repaired Mask (Th={threshold}, KS={kernel_size})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    cv2.imwrite(f'{output_dir}/repaired_mask.png', repaired_mask)

# ====================== 5. 主处理流程 ======================
if __name__ == '__main__':
    # 配置参数
    threshold = 0.3
    kernel_size = 3

    # 输入/输出路径
    img_path = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\水系\ceshitupian\shuixi\TW2021_4326_XIAO_MORE2-2.png'
    output_dir = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\水系\save_ceshi'
    IMAGE_DIR=r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\水系\ceshitupian\shuixi'

    CLASSES = ["background", "water"]  # 类别顺序：0=背景，1=水体
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


    # 1. 读取原始图像
    with open(img_path, 'rb') as f:
        img_data = f.read()
    original_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if original_img is None:
        raise FileNotFoundError(f"图像不存在: {img_path}")

    # 获取原始尺寸 (height, width) —— 关键修正点
    h, w = original_img.shape[:2]
    original_size = (h, w)          # 注意顺序：(高度, 宽度)
    print(f"原始图像尺寸 (H, W): {original_size}")

    # 2. 获取概率图 (上采样到原始尺寸)
    water_prob = get_water_probability(img_path, original_size=original_size)
    print(f"概率图尺寸: {water_prob.shape}")   # 应为 (h, w)

    # 3. 修复水道断裂
    repaired_mask = repair_water_mask(water_prob, threshold, kernel_size)
    print(f"修复掩码尺寸: {repaired_mask.shape}") # 应为 (h, w)

    # 4. 保存可视化结果（传入 threshold 和 kernel_size）
    visualize_results(original_img, water_prob, repaired_mask, threshold, kernel_size, output_dir)

    # 5. 生成叠加效果图
    overlay_img = original_img.copy()
    overlay_img[repaired_mask > 0] = [0, 0, 255]   # 红色标记水体 (BGR)
    alpha = 0.4
    blended = cv2.addWeighted(original_img, 1 - alpha, overlay_img, alpha, 0)
    cv2.imwrite(f'{output_dir}/overlay_result.jpg', blended)

    # 6. 保存概率热力图（用于调试）
    plt.imsave(f'{output_dir}/water_prob_heatmap.png', water_prob, cmap='jet', vmin=0, vmax=1)

    print(f"✅ 处理完成! 结果保存至: {output_dir}")
    print(f"   - 原始尺寸 (H,W): {original_size}")
    print(f"   - 概率图尺寸: {water_prob.shape}")
    print(f"   - 修复掩码尺寸: {repaired_mask.shape}")