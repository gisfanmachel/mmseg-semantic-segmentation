# -*- coding: utf-8 -*-
import os

import cv2
import mmcv
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""We need to convert the annotation into semantic map format as an image."""

# define dataset root and directory for images and annotations
data_root = '../data/cityscapes/iccv09Data'
img_dir = 'images'
ann_dir = 'labels'
# define class and palette for better visualization
classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

import numpy as np

"""After downloading the data, we need to implement `load_annotations` function in the new dataset class `StanfordBackgroundDataset`."""

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class StanfordBackgroundDataset(BaseSegDataset):
    METAINFO = dict(classes=classes, palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)


"""### Create a config file
In the next step, we need to modify the config for the training. To accelerate the process, we finetune the model from trained weights.
"""

# Download config and checkpoint files
# !mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .

from mmengine import Config

cfg = Config.fromfile('../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py')
print(f'Config:\n{cfg.pretty_text}')

"""Since the given config is used to train PSPNet on the cityscapes dataset, we need to modify it accordingly for our new dataset.  """

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.crop_size = (256, 256)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 8
cfg.model.auxiliary_head.num_classes = 8

# Modify dataset type and path
cfg.dataset_type = 'StanfordBackgroundDataset'
cfg.data_root = data_root

cfg.train_dataloader.batch_size = 8

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 240), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader

# Load the pretrained weights
cfg.load_from = '../pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '../work_dirs/tutorial'

cfg.train_cfg.max_iters = 200
cfg.train_cfg.val_interval = 200
cfg.default_hooks.logger.interval = 10
cfg.default_hooks.checkpoint.interval = 200

# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

cfg.dump('pspnet_dump.py')

"""### Train and Evaluation"""
# 注释掉训练
# from mmengine.runner import Runner
#
# runner = Runner.from_cfg(cfg)
#
# # start training
# runner.train()

"""Inference with trained model"""

from mmseg.apis import init_model, inference_model, show_result_pyplot

# Init the model from the config and the checkpoint
checkpoint_path = '../work_dirs/tutorial/iter_200.pth'
checkpoint_path = '../work_dirs/scene_mmseg_v1-semseg-pspnet_mask2/iter_200.pth'

# checkpoint_path="../watermelon_best_mIoU_iter_29000.pth"
# cfg = Config.fromfile('../Zihao-Configs/ZihaoDataset_PSPNet_20230818.py')

model = init_model(cfg, checkpoint_path, 'cuda:0')

img_path = '6000035.jpg'
# img_path="watermelon_test1.jpg"
img = mmcv.imread(img_path)
result = inference_model(model, img)

plt.figure(figsize=(8, 6))
vis_result = show_result_pyplot(model, img, result)
plt.imshow(mmcv.bgr2rgb(vis_result))

# pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

# model.dataset_meta = {'classes': classes, 'palette': palette}
# classes = model.dataset_meta.get('classes', None)
# palette = model.dataset_meta.get('palette', None)

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
# for label, color in zip(labels, colors):
#     mask[sem_seg[0] == label, :] = color
for label, color in zip(labels, colors):
    mask[sem_seg[0] == label, :] = [label, label, label]
# 生成mask多值图
import uuid

tempvalue = str(uuid.uuid4())
mask_image_path = "mask_{}.png".format(tempvalue)
cv2.imwrite(mask_image_path, mask)

# # 对每个标签类都获取一个mask（false,true）
# if isinstance(sem_seg[0], torch.Tensor):
#     masks = sem_seg[0].numpy() == labels[:, None, None]
# else:
#     masks = sem_seg[0] == labels[:, None, None]
# masks = masks.astype(np.uint8)

# 参考变化监测里的sam分割结果转换
# 参考mmseg/visualization/local_visualizer.py
# for mask in masks:
#     # 生成一个灰度图图片
#     # 获取图片每个通道数据
#     r, g, b, a = mask_image[:, :, 0], mask_image[:, :, 1], mask_image[:, :, 2], mask_image[:, :, 3]
#     # 目标是255白色，背景是0黑色
#     mask_image_grey = np.where(r > 0, 255, 0)
#     # 保存mask的多值图
#     mask_image_path = "mask.png"
#     cv2.imwrite(mask_image_path, mask_image_grey)

# 将mask多值图转为shp
from vgis_rs.rsTools import RsToolsOperatoer

world_file_path = "6000035.jgw"
shp_file_path = "6000035_{}.shp".format(tempvalue)
src_transform = RsToolsOperatoer.convert_jgw_to_gdaltransform(world_file_path)
RsToolsOperatoer.raster_to_vector(mask_image_path, src_transform, shp_file_path, "ESRI Shapefile")
# 生成的shp字段值value为为labelid
