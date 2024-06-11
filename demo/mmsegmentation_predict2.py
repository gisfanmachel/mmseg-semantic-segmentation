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
import mmcv
from matplotlib import pyplot as plt
from mmengine import Config

from mmseg.apis import init_model, inference_model, show_result_pyplot

checkpoint_path = "../watermelon_best_mIoU_iter_29000.pth"
cfg = Config.fromfile('../Zihao-Configs/ZihaoDataset_PSPNet_20230818.py')
model = init_model(cfg, checkpoint_path, 'cuda:0')
img_path = "watermelon_test1.jpg"
img = mmcv.imread(img_path)
result = inference_model(model, img)

plt.figure(figsize=(8, 6))
vis_result = show_result_pyplot(model, img, result)
plt.imshow(mmcv.bgr2rgb(vis_result))
