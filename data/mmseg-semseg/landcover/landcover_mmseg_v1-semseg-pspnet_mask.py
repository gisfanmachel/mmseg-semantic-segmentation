_base_ = '../../../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
data_root = './data/mmseg-semseg/landcover5/'
load_from = './data/mmseg-semseg/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'  # noqa
dataset_type = 'BaseSegDataset'
img_dir = 'images'
ann_dir = 'labels'
# 背景（0）,建筑(1)，林地(2)，水(3)，道路(4)
metainfo = {
    'classes': ('backgroud', 'building', 'forest', 'water', 'road'),
    'palette': [(128, 128, 128), (129, 127, 38), (120, 69, 125), (53, 125, 34),
                (0, 11, 123)]
}

# metainfo = {
#     'classes': ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj'),
#     'palette': [(128, 128, 128), (129, 127, 38), (120, 69, 125), (53, 125, 34),
#                 (0, 11, 123), (118, 20, 12), (122, 81, 25), (241, 134, 51)]
# }
num_classes = len(metainfo["classes"])

# 训练单卡 bs= 16
# 这个跟显存大小有关系
train_batch_size_per_gpu = 8
# 可以根据自己的电脑修改
train_num_workers = 4
# 验证集 batch size 为 1
val_batch_size_per_gpu = 1
val_num_workers = 2

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 1024)

model = dict(
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        num_classes=num_classes),
    backbone=dict(
        norm_cfg=norm_cfg,
        norm_eval=False,
        num_stages=4),
    data_preprocessor=dict(
        size=crop_size),
    decode_head=dict(
        norm_cfg=norm_cfg,
        num_classes=num_classes
    ))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048,1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048,1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 数据集不同，dataset 输入参数也不一样
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_root=data_root,
        ann_file='splits/train.txt',
        pipeline=train_pipeline,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        type=dataset_type)
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_root=data_root,
        ann_file='splits/val.txt',
        pipeline=test_pipeline,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir), type=dataset_type))

test_dataloader = val_dataloader

train_cfg = dict(max_iters=200, type='IterBasedTrainLoop', val_interval=200)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=200, type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
