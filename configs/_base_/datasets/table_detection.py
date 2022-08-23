# dataset settings
dataset_type = 'TableDetectionDataset'
data_root = '/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

## ICDAR 2019
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/general_table_detection/general_coco.json',
#         img_prefix=data_root + 'images/dataset_train_v0.1.0/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/icdar_2019/icdar_19_td_test_coco.json',
#         img_prefix=data_root + 'images/icdar_2019_test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/icdar_2019/icdar_19_td_test_coco.json',
#         img_prefix=data_root + 'images/icdar_2019_test/',
#         pipeline=test_pipeline))


## SEDAR
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'niranjan/sedar/positive/sedar_positive_train.json',
        img_prefix=data_root + 'niranjan/sedar/positive/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'niranjan/sedar/positive/sedar_positive_test.json',
        img_prefix=data_root + 'niranjan/sedar/positive/test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'niranjan/sedar/positive/sedar_positive_test.json',
        img_prefix=data_root + 'niranjan/sedar/positive/test/',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='bbox')
