import os
import torch
from contextlib import nullcontext
from mmseg.apis import MMSegInferencer
from mmengine.config import Config
# 解决torch版本过高，与mmseg不兼容的问题
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
# ===================== 【配置参数】 =====================
# zgf-deeplabv3plus训练
CONFIG_PATH = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\建筑\buildconfig2.py'
CHECKPOINT_PATH = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\建筑\best_mIoU_iter_388240.pth'

# cxw-segformer训练
# CHECKPOINT_PATH=r'E:\AI\train_result\mmseg\building\20260412_235257\best_mIoU_iter_37000.pth'
# CONFIG_PATH=r'E:\AI\train_result\mmseg\building\20260412_235257\building_mmseg_v1-semseg-deeplabv3plus_mask.py'

#

IMAGE_DIR = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\建筑\ceshitupian\build'
OUTPUT_DIR = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\建筑\save_ceshi'
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