# Q2 Report: Multi-Task Learning (Segmentation + Depth)

## Test Metrics
| Model | Test mIoU | Test RMSE |
|---|---:|---:|
| noskip | 0.6707 | 0.0419 |
| residual | 0.7425 | 0.0313 |
| vanilla | 0.6671 | 0.0322 |

## W&B
Project: https://wandb.ai/mastermindlakshaybaijal-iiit-hyderabad/cv-assignment-q2

## Comparison Summary
- Residual performs best overall (highest mIoU and lowest RMSE).
- NoSkip has slightly higher mIoU than Vanilla but much worse RMSE.
- Vanilla gives balanced depth quality but lower segmentation than Residual.

## Qualitative Discussion
- Compare object boundaries in masks across all three models.
- Compare depth smoothness and edge consistency.
- Include at least 10 side-by-side test predictions per model.
