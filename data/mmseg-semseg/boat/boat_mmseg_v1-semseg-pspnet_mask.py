_base_ = '../../../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
data_root = './data/mmseg-semseg/boat/'
load_from = './data/mmseg-semseg/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'  # noqa
dataset_type = 'BaseSegDataset'
img_dir = 'images'
ann_dir = 'labels'
img_suffix = '.jpg'
seg_map_suffix = '.png'
metainfo = {
    'classes': ('unlabeled', 'submarine', 'cargoship', 'warship', 'liner', 'ship'),
    'palette': [(128, 128, 128), (129, 127, 38), (120, 69, 125), (53, 125, 34),
                (0, 11, 123), (118, 20, 12)]
}
num_classes = len(metainfo["classes"])

# 训练 200 epoch
max_epochs = 200
# 训练单卡 bs= 16
# 这个跟显存大小有关系
train_batch_size_per_gpu = 4
# 可以根据自己的电脑修改
train_num_workers = 4
# 验证集 batch size 为 1
val_batch_size_per_gpu = 1
val_num_workers = 2

norm_cfg = dict(type='BN', requires_grad=True)
# batch 改变了，学习率也要跟着改变， 0.004 是 8卡x32 的学习率
base_lr = train_batch_size_per_gpu * 0.004 / (32 * 8)
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=base_lr, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=base_lr, momentum=0.9, type='SGD', weight_decay=0.0005)
# crop_size = (800, 800)

model = dict(
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        num_classes=num_classes),
    backbone=dict(
        norm_cfg=norm_cfg,
        norm_eval=False,
        num_stages=4),
    # data_preprocessor=dict(
    #     size=crop_size),
    decode_head=dict(
        norm_cfg=norm_cfg,
        num_classes=num_classes
    ))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='RandomResize', scale=(800, 800), ratio_range=(0.5, 2.0), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(800, 800), keep_ratio=True),
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
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
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
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        data_root=data_root,
        ann_file='splits/val.txt',
        pipeline=test_pipeline,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir), type=dataset_type))

test_dataloader = val_dataloader

train_cfg = dict(max_iters=max_epochs, type='IterBasedTrainLoop', val_interval=10)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10, type='CheckpointHook', save_best='auto'),
    logger=dict(interval=10, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
