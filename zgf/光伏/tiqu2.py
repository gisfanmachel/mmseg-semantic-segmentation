import os
import shutil

import cv2
import torch
from contextlib import nullcontext
from mmseg.apis import MMSegInferencer
from mmengine.config import Config
# 解决torch版本过高，与mmseg不兼容的问题
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# ===================== 【配置参数】 =====================
# zgf deeplabv3plus训练
CONFIG_PATH = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\光伏\spvguangfu.py'
CHECKPOINT_PATH = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\光伏\best_mIoU_iter_553040.pth'


# CHECKPOINT_PATH=r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\work_dirs\photovoltaic_panel_mmseg_v1-semseg-pspnet_mask\20260407_145007train\best_aAcc_epoch_10.pth'


# cxw pspnet训练结果
# CHECKPOINT_PATH=r'E:\AI\train_result\mmseg\guangfu\20260408_164134\best_mIoU_iter_15000.pth'
# CONFIG_PATH=r'E:\AI\train_result\mmseg\guangfu\20260408_164134\photovoltaic_panel_mmseg_v1-semseg-pspnet_mask_zihaodata2.py'
#

# # cxw segformer训练结果
# CHECKPOINT_PATH=r'E:\AI\train_result\mmseg\guangfu\20260408_191415\best_mIoU_iter_36000.pth'
# CONFIG_PATH = r'E:\AI\train_result\mmseg\guangfu\20260408_191415\photovoltaic_panel_mmseg_v1-semseg-segformer_mask.py'

# cxw deeplabv3plus训练结果
# CHECKPOINT_PATH=r'E:\AI\train_result\mmseg\guangfu\20260409_092725\best_mIoU_iter_34500.pth'
# CONFIG_PATH=r'E:\AI\train_result\mmseg\guangfu\20260409_092725\photovoltaic_panel_mmseg_v1-semseg-deeplabv3plus_mask.py'

# cxw fastscnn训练结果
# CHECKPOINT_PATH = r'E:\AI\train_result\mmseg\guangfu\20260409_133937\best_mIoU_iter_29000.pth'
# CONFIG_PATH = r'E:\AI\train_result\mmseg\guangfu\20260409_133937\photovoltaic_panel_mmseg_v1-semseg-fastscnn_mask.py'

# cxw knet训练结果

# cxw mask2former训练结果

# cxw unet训练结果
# CHECKPOINT_PATH = r'E:\AI\train_result\mmseg\guangfu\20260409_094936\best_mIoU_iter_26500.pth'
# CONFIG_PATH = r'E:\AI\train_result\mmseg\guangfu\20260409_094936\photovoltaic_panel_mmseg_v1-semseg-unet_mask.py'

IMAGE_DIR = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\光伏\ceshitupian'
OUTPUT_DIR = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\光伏\save_ceshi'
CLASSES = ["background", "pv"]  # 类别顺序：0=背景，1=光伏

# ========== 核心修改：自定义调色板（RGB值范围 0-255） ==========
# 格式：[[背景的RGB], [光伏的RGB]]，可根据需求修改
# 示例1：背景白色(255,255,255)、光伏蓝色(0,0,255)（推荐）
PALETTE = [[255, 255, 255], [0, 0, 255]]
# 示例2：背景黑色(0,0,0)、光伏绿色(0,255,0)（备用，取消注释即可）
# PALETTE = [[0, 0, 0], [0, 255, 0]]
# 示例3：背景灰色(128,128,128)、光伏黄色(255,255,0)
# PALETTE = [[128, 128, 128], [255, 255, 0]]

BATCH_SIZE = 1
RESIZE_SCALE = (512, 512)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # ===================== 步骤0：手动缩放所有图片 =====================
# from mmcv import imread, imwrite   # 添加导入
#
# RESIZED_DIR = IMAGE_DIR + '_resized_512'
# if os.path.exists(RESIZED_DIR):
#     shutil.rmtree(RESIZED_DIR)
# os.makedirs(RESIZED_DIR, exist_ok=True)
#
# print(f"正在缩放图片到 {RESIZE_SCALE} ...")
# for fname in os.listdir(IMAGE_DIR):
#     if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
#         img_path = os.path.join(IMAGE_DIR, fname)
#         img = imread(img_path)                 # 支持中文路径
#         if img is not None:
#             resized = cv2.resize(img, RESIZE_SCALE, interpolation=cv2.INTER_LINEAR)
#             save_path = os.path.join(RESIZED_DIR, fname)
#             imwrite(resized, save_path)        # 支持中文路径
# print("缩放完成。")
#
# IMAGE_DIR = RESIZED_DIR   # 后续推理使用缩放后的目录
# ===================== 步骤1：加载并修改配置文件 =====================
cfg = Config.fromfile(CONFIG_PATH)

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
inferencer = MMSegInferencer(
    model=cfg,
    weights=CHECKPOINT_PATH,
    classes=CLASSES,       # 强制指定类别
    palette=PALETTE,       # 强制指定调色板（覆盖配置文件默认值）
    device=DEVICE
)

# ===================== 步骤3：获取测试图片列表 =====================
image_list = [
    os.path.join(IMAGE_DIR, img_name)
    for img_name in os.listdir(IMAGE_DIR)
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png','.tif'))
][:200]

if not image_list:
    raise FileNotFoundError(f"错误：在 {IMAGE_DIR} 中未找到任何图片！")

print(f"找到 {len(image_list)} 张测试图片，开始分批次分割...")

# ===================== 步骤4：分批次推理 =====================
os.makedirs(os.path.join(OUTPUT_DIR, 'vis'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'mask'), exist_ok=True)

autocast_context = torch.cuda.amp.autocast() if DEVICE == 'cuda:0' else nullcontext()

for batch_idx in range(0, len(image_list), BATCH_SIZE):
    batch_imgs = image_list[batch_idx:batch_idx + BATCH_SIZE]
    batch_num = batch_idx // BATCH_SIZE + 1
    print(f"\n处理第 {batch_num} 批，共 {len(batch_imgs)} 张图...")

    with autocast_context:
        inferencer(
            batch_imgs,
            batch_size=BATCH_SIZE,
            out_dir=OUTPUT_DIR,
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

# ===================== 结果提示 =====================
print(f"\n分割完成！结果保存路径：")
print(f"- 可视化图：{os.path.join(OUTPUT_DIR, 'vis')}")
print(f"- 掩码图：{os.path.join(OUTPUT_DIR, 'mask')}")