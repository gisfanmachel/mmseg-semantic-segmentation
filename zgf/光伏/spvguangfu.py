import os
# 数据集配置
dataset_type = 'PvDataset'
data_root = '/datas/guangfu/pngs2'

# ========== 新增：定义类别和调色板（核心） ==========
metainfo = {
    'classes': ["background", "pv"],  # 类别顺序（必须与训练标签一致）
    'palette': [[255, 255, 255], [0, 0, 255]]  # 自定义颜色：背景白色，光伏蓝色
}

# 图像尺寸，根据你的数据调整
img_size = (512, 512)

# 训练数据处理流程
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

# 验证数据处理流程
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 测试数据处理流程
test_pipeline = val_pipeline  # 测试流程通常与验证流程相同

# 训练/验证/测试数据加载器（batch_size保持2不变）
train_dataloader = dict(
    batch_size=2,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,  # ========== 新增：关联metainfo ==========
        data_prefix=dict(
            img_path='train_aug2/',
            seg_map_path='train_mask_aug2/'
        ),
        pipeline=train_pipeline
    ))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,  # ========== 新增：关联metainfo ==========
        data_prefix=dict(
            img_path='val_aug2/',
            seg_map_path='val_mask2/'
        ),
        pipeline=val_pipeline
    ))

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,  # ========== 新增：关联metainfo ==========
        data_prefix=dict(img_path='test/', seg_map_path='test/masks/'),
        pipeline=test_pipeline
    ))

# 关键修改1：Checkpoint保存间隔（每5轮保存1次 → 6913×5=34565）
checkpoint_config = dict(
    by_epoch=False,
    interval=34565,  # 替换原15300
    save_best='mIoU',
    rule='greater'
)

# 评估指标
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# 关键修改2：总迭代数+验证间隔（80轮总迭代=553040，每1轮验证=6913）
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=553040,  # 替换原244800
    val_interval=6913  # 替换原3060
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器（batch_size=2保持不变，学习率无需调整）
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00009,  # 保持batch_size=2对应的学习率
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }),
    loss_scale='dynamic'
)

# 关键修改3：学习率调度器结束迭代数（与总迭代数553040一致）
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=0.9,
        begin=2000,
        end=553040,  # 替换原244800
        by_epoch=False
    )
]

# 模型配置
_base_ = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\configs\segformer\segformer_mit-b2_8xb2-160k_ade20k-512x512.py'
checkpoint = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\configs\segformer\segforme_pths\mit_b2_20220624-66e8bf70.pth'
work_dir = r'D:\系统开发\AI\AI_Study\图像分割\mmseg\mmseg-semantic-segmentation\zgf\光伏\work_dir'
os.makedirs(work_dir, exist_ok=True)

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=2,  # ========== 确认：类别数与metainfo一致 ==========
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,  # ========== 确认：类别数与metainfo一致 ==========
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            avg_non_ignore=True
        )
    )
)

# 关键修改4：default_hooks中checkpoint间隔同步更新为34565
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=34565,  # 替换原15300，与checkpoint_config一致
        save_best='mIoU',
        rule='greater'
    ),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        min_delta=0.0005,
        patience=15,  # 连续15轮验证mIoU无提升则早停（15×6913=103695迭代）
        rule='greater'
    )
)