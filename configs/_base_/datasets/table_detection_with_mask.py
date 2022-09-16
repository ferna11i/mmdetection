
# dataset settings
dataset_type = 'TableDetectionDataset'
data_root = '/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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

# # ICDAR 2019 test
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/0.1.1/dataset_train_v0.1.1_cascadetabnet.json',
#         img_prefix=data_root + 'images/dataset_train_v0.1.0_temp/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/0.1.1/icdar_2019_test/annotations/icdar_19_td_gt.json',
#         img_prefix=data_root + 'images/icdar_2019_test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/0.1.1/icdar_2019_test/annotations/icdar_19_td_gt.json',
#         img_prefix=data_root + 'images/icdar_2019_test/',
#         pipeline=test_pipeline))


# # ICDAR 2019 training
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/0.1.1/icdar_2019_train/annotations/icdar_19_td_train_coco.json',
#         img_prefix=data_root + 'images/icdar_2019_train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/0.1.1/icdar_2019_test/annotations/icdar_19_td_gt.json',
#         img_prefix=data_root + 'images/icdar_2019_test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/0.1.1/icdar_2019_test/annotations/icdar_19_td_gt.json',
#         img_prefix=data_root + 'images/icdar_2019_test/',
#         pipeline=test_pipeline))



# ## SEDAR
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/sedar/annotations/sedar_positive_train.json',
#         img_prefix=data_root + 'tabular/sedar/images/train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/sedar/annotations/sedar_positive_gt.json',
#         img_prefix=data_root + 'tabular/sedar/images/test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/sedar/annotations/sedar_positive_gt.json',
#         img_prefix=data_root + 'tabular/sedar/images/test/',
#         pipeline=test_pipeline))


# ## CanDev
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/candev/annotations/candev_positive_train.json',
#         img_prefix=data_root + 'tabular/candev/images/train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/candev/annotations/candev_positive_gt.json',
#         img_prefix=data_root + 'tabular/candev/images/test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/candev/annotations/candev_positive_gt.json',
#         img_prefix=data_root + 'tabular/candev/images/test/',
#         pipeline=test_pipeline))


# ## Combined
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/combined/annotations/train_0.1.1.json',
#         img_prefix=data_root + 'tabular/combined/images/train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/combined/annotations/gt_0.1.1.json',
#         img_prefix=data_root + 'tabular/combined/images/test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'tabular/combined/annotations/gt_0.1.1.json',
#         img_prefix=data_root + 'tabular/combined/images/test/',
#         pipeline=test_pipeline))

## Combined + IC 19
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'tabular/combined_ic19/annotations/combined_ic19_train.json',
        img_prefix=data_root + 'tabular/combined_ic19/images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'tabular/combined_ic19/annotations/combined_ic19_test.json',
        img_prefix=data_root + 'tabular/combined_ic19/images/test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'tabular/combined_ic19/annotations/combined_ic19_test.json',
        img_prefix=data_root + 'tabular/combined_ic19/images/test/',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='bbox')