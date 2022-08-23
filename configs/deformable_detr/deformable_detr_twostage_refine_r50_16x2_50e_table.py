_base_ = 'deformable_detr_refine_r50_16x2_50e_table.py'
model = dict(bbox_head=dict(as_two_stage=True))

## From scratch
# load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/johan/AlternateModels/weights/deformable_detr_twostage_refine_r50_16x2_50e_coco.pth"

# Trained on general dataset
load_from = "/home/jovyan/dev-hydra-module-tabular-1-gpu-vol-1/workdir_1class/deformable_detr_2_stage_epoch_25_min.pth"
