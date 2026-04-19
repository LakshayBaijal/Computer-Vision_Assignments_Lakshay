# Q3 Report: Learning on Point Clouds

## 3.1 Data Preprocessing and Vanilla PointNet

### Data preprocessing
- Input point clouds are centered (zero-mean) and normalized to unit scale before model input.
- A fixed number of points is sampled per object to support batched training.
- Only xyz geometry is used as model input (no mesh connectivity).

### Vanilla PointNet (without T-Net)
- Shared point-wise feature extractor implemented with `Conv1d` layers.
- Symmetric global feature aggregation via max pooling.
- MLP classifier head maps global features to ModelNet-10 class logits.

### Training behavior and final metrics (from saved outputs)
From `history.json` / `history.csv`:
- Epochs trained: **40**
- Best validation accuracy: **0.9707** at epoch **32**
- Best validation loss: **0.1209** at epoch **32**
- Final epoch train loss / val loss: **0.0638 / 0.1294**
- Final epoch train acc / val acc: **0.9740 / 0.9665**

From `test_metrics.json`:
- **Test loss:** `0.2446621029298502`
- **Test accuracy:** `0.9176334106728539` (91.76%)

Saved visuals:
- `training_curves.png`
- `losses.png`

---

## 3.2 Permutation Invariance

Permutation test results are taken from `analysis_metrics.json`:
- **Test accuracy (analysis run):** `0.9164733178654292`
- **Predictions changed after random point permutation:** `0.0%`

### Conclusion
The model output is unchanged under random re-ordering of input points in this evaluation. This matches PointNet’s design: shared point-wise processing followed by a symmetric max operation yields permutation invariance.

---

## 3.3 Critical Point Analysis and Robustness

### Critical-point extraction
- Critical points are identified as those contributing to max-pooled feature dimensions.
- Combined visualization is available in `critical_points.png`.
- Per-sample visualizations are available in:
  - `critical_analysis/critical_vis_01.png`
  - `critical_analysis/critical_vis_02.png`
  - `critical_analysis/critical_vis_03.png`
  - `critical_analysis/critical_vis_04.png`
  - `critical_analysis/critical_vis_05.png`

### Robustness using only critical points
From `analysis_metrics.json`:
- **Original analysis accuracy:** `0.9164733178654292`
- **Sparse critical-only accuracy:** `0.919953596287703`

### Interpretation
Accuracy is preserved (slightly higher in this run) even when using only critical points. This supports the PointNet hypothesis that a sparse subset of informative points is sufficient to retain most discriminative global shape information.

---

## Files Included (`results/q3`)
- `best.pt`
- `pointnet_model.pt`
- `history.csv`
- `history.json`
- `test_metrics.json`
- `analysis_metrics.json`
- `metrics.json`
- `training_curves.png`
- `losses.png`
- `critical_points.png`
- `critical_analysis/critical_vis_01.png`
- `critical_analysis/critical_vis_02.png`
- `critical_analysis/critical_vis_03.png`
- `critical_analysis/critical_vis_04.png`
- `critical_analysis/critical_vis_05.png`
- `q3_report.md`
