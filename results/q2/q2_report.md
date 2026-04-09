# Q2 Report: Multi-Task Learning (Segmentation + Depth Estimation)

## 1) Experimental Setup

- Input resolution: `256 x 256`
- Tasks: semantic segmentation + depth estimation
- Validation split: `25%`
- Trained variants:
	- Vanilla Multi-Task U-Net
	- Multi-Task U-Net without skip connections
	- Multi-Task U-Net with residual blocks

## 2) Combined Loss

The model is optimized using a combined multi-task objective:

- Segmentation loss (cross-entropy style)
- Depth loss (regression loss)
- Total loss = weighted combination of both

All three variants were trained with the same base training setup for fair comparison.

## 3) Weights & Biases Logging

- Project: https://wandb.ai/mastermindlakshaybaijal-iiit-hyderabad/cv-assignment-q2
- Vanilla run: https://wandb.ai/mastermindlakshaybaijal-iiit-hyderabad/cv-assignment-q2/runs/8fz0ch4k
- NoSkip run: https://wandb.ai/mastermindlakshaybaijal-iiit-hyderabad/cv-assignment-q2/runs/47bslxn4
- Residual run: https://wandb.ai/mastermindlakshaybaijal-iiit-hyderabad/cv-assignment-q2/runs/napdthdd

## 4) Test Set Metrics (Reported)

| Model | Test mIoU | Test RMSE |
|---|---:|---:|
| Vanilla | 0.6671 | 0.0322 |
| Without Skip | 0.6707 | 0.0419 |
| Residual | 0.7425 | 0.0313 |

## 5) Curve Outputs (Per Model)

### Vanilla (`results/q2/vanialla_unet`)
- Losses: ![vanilla_losses](vanialla_unet/losses.png)
- Validation mIoU: ![vanilla_miou](vanialla_unet/mIOU_plot.png)
- Validation RMSE: ![vanilla_rmse](vanialla_unet/RMSE_plot.png)

### Without Skip (`results/q2/without_skip`)
- Losses: ![noskip_losses](without_skip/losses.png)
- Validation mIoU: ![noskip_miou](without_skip/mIOU_plot.png)
- Validation RMSE: ![noskip_rmse](without_skip/RMSE_plot.png)

### Residual (`results/q2/residual`)
- Losses: ![residual_losses](residual/losses.png)
- Validation mIoU: ![residual_miou](residual/mIOU_plot.png)
- Validation RMSE: ![residual_rmse](residual/RMSE_plot.png)

## 6) Qualitative Outputs

For each model, qualitative predictions are available in:

- `vanialla_unet/qualitative_samples/` (10 samples)
- `without_skip/qualitative_samples/` (10 samples)
- `residual/qualitative_samples/` (10 samples)

Representative combined visualization files:

- Vanilla: ![vanilla_qual](vanialla_unet/qualitative_results.png)
- Without Skip: ![noskip_qual](without_skip/qualitative_results.png)
- Residual: ![residual_qual](residual/qualitative_results.png)

## 7) Comparative Analysis

- **Residual U-Net** performs best overall, with the highest segmentation quality (`mIoU = 0.7425`) and best depth accuracy (`RMSE = 0.0313`).
- **Without Skip** is slightly better than Vanilla in mIoU but much worse in RMSE, indicating weaker depth reconstruction without encoder-decoder skip transfer.
- **Vanilla** provides balanced baseline behavior but is outperformed by Residual in both tasks.

## 8) Conclusion

Residual connections improved feature learning and produced the best joint segmentation-depth performance in this assignment setup.
