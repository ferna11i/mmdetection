_base_ = '../cascade_rcnn/cascade_rcnn_r50_fpn_1x_table.py'
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

#optimizer
optimizer = dict(type='SGD', lr=0.0012, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 19])
# lr_config = dict(step=[20, 35])
runner = dict(type='EpochBasedRunner', max_epochs=80)


work_dir = '/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/workdir_1class/'

# Training with COCO weights
# load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/johan/AlternateModels/weights/cascade_rcnn_hrnetv2p_w32_20e_coco.pth"

# Original CascadeTabNet weights
# load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/johan/AlternateModels/weights/epoch_24_updated.pth"
load_from = None
resume_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/johan/AlternateModels/weights/cascadetabnet_epoch_24_min_prototype0.pth"

# # Train from General dataset weights
# load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/workdir_1class/cascadetabnet_epoch_69_prototype2.pth"