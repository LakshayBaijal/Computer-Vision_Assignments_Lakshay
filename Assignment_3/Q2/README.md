# Q2 - Multi-Task Learning (Segmentation + Depth)

This folder implements end-to-end training/evaluation for Assignment Q2 with three variants:

1. `vanilla` Multi-Task U-Net
2. `noskip` U-Net without skip connections
3. `residual` U-Net with residual blocks

All variants use the same data split, optimizer, batch size, loss weights, and seed for fair comparison.

## Dataset expectations

The loader supports common layouts. Use one of:

- `root/train/images`, `root/train/labels`, `root/train/depth`
- `root/train/images`, `root/train/masks`, `root/train/depth`
- `root/images`, `root/labels|masks`, `root/depth` (train/val split created internally)
- `root/test/images`, `root/test/labels|masks`, `root/test/depth` for test evaluation

Image/mask/depth are resized to `256x256` (configurable).

## Loss definition (explicit)

Combined loss used for all runs:

`total_loss = lambda_seg * CE(segmentation) + lambda_depth * L1(depth)`

Default:

- `lambda_seg = 1.0`
- `lambda_depth = 1.0`

## What is logged each epoch

- Combined train/val loss
- Individual train/val segmentation loss
- Individual train/val depth loss
- Validation mIoU
- Validation RMSE

Curves are saved under each run directory:

- `combined_loss_curve.png`
- `individual_loss_curves.png`
- `val_miou_curve.png`
- `val_rmse_curve.png`

## Install

```powershell
Set-Location "d:\CSIS_IIIT_Hyderabad\4th Semester\CV\Assignment 3\Q2"
python -m pip install -r requirements.txt
```

## Train (at least 10 epochs)

Use the same hyperparameters for all three runs.

Fair comparison requirement (must be fixed across variants):

- same LR
- same batch size
- same loss weights (`lambda_seg`, `lambda_depth`)
- same data split seed
- same augmentations

```powershell
python train.py --config config/default.yaml --variant vanilla --data_root "<DATASET_ROOT>" --epochs 10
python train.py --config config/default.yaml --variant noskip --data_root "<DATASET_ROOT>" --epochs 10
python train.py --config config/default.yaml --variant residual --data_root "<DATASET_ROOT>" --epochs 10
```

## Evaluate on test set + save qualitative samples

```powershell
python eval.py --config config/default.yaml --variant vanilla --data_root "<DATASET_ROOT>" --checkpoint "outputs/q2_multitask_unet_vanilla/best.pt"
python eval.py --config config/default.yaml --variant noskip --data_root "<DATASET_ROOT>" --checkpoint "outputs/q2_multitask_unet_noskip/best.pt"
python eval.py --config config/default.yaml --variant residual --data_root "<DATASET_ROOT>" --checkpoint "outputs/q2_multitask_unet_residual/best.pt"
```

`eval.py` saves:

- `test_metrics.json` (`mIoU`, `RMSE`)
- `qualitative_samples/sample_00.png ... sample_09.png` (at least 10 side-by-side predictions)

## wandb

Enable/disable in `config/default.yaml`:

- `wandb.enabled: true|false`
- set `wandb.project`, `wandb.entity`, `wandb.run_name`

## Comparative analysis checklist (for report)

For each variant compare:

- final test `mIoU` and `RMSE`
- segmentation boundary sharpness
- depth smoothness and edge quality
- failure cases on difficult objects/backgrounds
