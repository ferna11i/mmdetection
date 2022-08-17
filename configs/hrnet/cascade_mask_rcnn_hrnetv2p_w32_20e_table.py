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

# learning policy
lr_config = dict(step=[19, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)

work_dir = '/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/workdir_1class/'
load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/johan/AlternateModels/weights/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.pth"
