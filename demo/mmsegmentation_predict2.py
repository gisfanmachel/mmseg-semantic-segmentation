"""
#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
@Project :mmseg-semantic-segmentation
@File    :mmsegmentation_predict2.py.py
@IDE     :PyCharm
@Author  :chenxw
@Date    :2024/6/11 17:59
@Descr:
"""
import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmengine import Config

from mmseg.apis import init_model, inference_model, show_result_pyplot

# 测试西瓜分割
checkpoint_path = "../watermelon_best_mIoU_iter_29000.pth"
cfg = Config.fromfile('../Zihao-Configs/ZihaoDataset_PSPNet_20230818.py')
img_path = "watermelon_test1.jpg"


# 测试landcover分割
classes = ['backgroud', 'building', 'forest', 'water', 'road']
palette =  [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
                [0, 11, 123]]
cfg = Config.fromfile('../Zihao-Configs/LandcoverDataset_PSPNet_20240611.py')
checkpoint_path = "../landcover_best_mIoU_iter_36500.pth"
img_path = "M-33-32-B-b-4-4.tif"

# 开始推理
model = init_model(cfg, checkpoint_path, 'cuda:0')
img = mmcv.imread(img_path)
result = inference_model(model, img)

# # 分割结果和图片叠加
# plt.figure(figsize=(8, 6))
# vis_result = show_result_pyplot(model, img, result)
# plt.imshow(mmcv.bgr2rgb(vis_result))

# 生成mask多值图
sem_seg = result.pred_sem_seg.cpu().data
ids = np.unique(sem_seg)[::-1]
num_classes = len(classes)
legal_indices = ids < num_classes
ids = ids[legal_indices]
labels = np.array(ids, dtype=np.int64)
colors = [palette[label] for label in labels]
# 对分割出来的各个类别实例进行不同颜色（指定）渲染
image = mmcv.imread(img_path, channel_order='rgb')
mask = np.zeros_like(image, dtype=np.uint8)
for label, color in zip(labels, colors):
    mask[sem_seg[0] == label, :] = [label, label, label]
mask_image_path = "{}_mask.png".format(img_path.split(".")[0])
cv2.imwrite(mask_image_path, mask)
