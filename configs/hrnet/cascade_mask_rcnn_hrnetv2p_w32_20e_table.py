_base_ = ['../cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_table.py']
model = dict(
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
#         init_cfg=dict(
#             type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32')
    ),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256))

# Original LR not changed by Johan----------------------
# learning policy
# lr_config = dict(step=[19, 25])
# runner = dict(type='EpochBasedRunner', max_epochs=80)

# LR & optimizer changed by Johan from old config----------------------
#optimizer
optimizer = dict(type='SGD', lr=0.0012, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[13, 17])
# lr_config = dict(step=[20, 35])
runner = dict(type='EpochBasedRunner', max_epochs=20)





# # optimizer for general dataset v0.1.0 new
# optimizer = dict(
#     type='AdamW',
#     lr=2e-4,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1),
#             'sampling_offsets': dict(lr_mult=0.1),
#             'reference_points': dict(lr_mult=0.1)
#         }))
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# # learning policy
# lr_config = dict(policy='step', step=[20, 35])
# runner = dict(type='EpochBasedRunner', max_epochs=50)


# # optimizer for icdar 2019 train
# optimizer = dict(
#     type='AdamW',
#     lr=2e-4,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1),
#             'sampling_offsets': dict(lr_mult=0.1),
#             'reference_points': dict(lr_mult=0.1)
#         }))
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# # learning policy
# lr_config = dict(policy='step', step=[19, 25])
# runner = dict(type='EpochBasedRunner', max_epochs=30)



work_dir = '/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/workdir_1class/'

# Original CascadeTabNet weights
# load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/johan/AlternateModels/weights/epoch_24_updated.pth"
# load_from = None
# resume_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/johan/AlternateModels/weights/cascadetabnet_epoch_24_min_prototype0.pth"


# Trained on General Training set
load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/workdir_1class/cascadetabnet_epoch_14_min_prototype2_ic19.pth"
resume_from = None

