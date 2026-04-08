# Q3 Report: Learning on Point Clouds

## 3.1 Data Preprocessing and Vanilla PointNet

### Preprocessing
- Loaded `.ply` point clouds from ModelNet-10 classes.
- Used xyz coordinates as input points.
- Centered each cloud to zero-mean.
- Normalized each cloud to unit sphere (max radius = 1).
- Resampled each cloud to a fixed number of points for batching.

### Vanilla PointNet (without T-Net)
- Shared MLP using `Conv1d` layers on unordered points.
- Global aggregation using max pooling to produce a global feature vector.
- Classification head using fully connected layers.

### Training Curves and Metrics
- Saved plots in `losses.png` (train/val loss and train/val accuracy).

Final test metrics:
- **Test loss:** `0.2447`
- **Test accuracy:** `0.9176` (91.76%)

---

## 3.2 Permutation Invariance

During inference, points in each test cloud were randomly permuted and predictions were compared with original-order predictions.

- **Original test accuracy:** `0.9165` (from analysis run)
- **Permutation-changed predictions:** `0.0%`
- **Permuted accuracy:** effectively unchanged (same predictions as original order)

### Observation
This is expected for PointNet: the network applies shared point-wise functions and then a symmetric max-pooling operation, making predictions invariant to input point order.

---

## 3.3 Critical Point Analysis and Robustness

### Critical Points
- Extracted pre-max-pooling features and identified critical point indices that contribute to pooled global features.
- Visualized 5 test samples in `critical_points.png`:
  - Left: original full point cloud
  - Right: critical points highlighted over faint full cloud

### Robustness with Critical-Points-Only Input
- Constructed sparse clouds using only extracted critical points.
- Evaluated trained model on these sparse inputs.

Results:
- **Original test accuracy (analysis):** `0.9165`
- **Sparse critical-only accuracy:** `0.9200`

### Interpretation
Accuracy does not drop in this run (slight increase due to sampling/statistical variation). This aligns with PointNet’s max-pooling design, where a sparse subset of critical points can preserve the dominant global features needed for classification.

---

## Files Submitted (Q3)
- `best.pt`
- `history.csv`
- `history.json`
- `test_metrics.json`
- `analysis_metrics.json`
- `losses.png`
- `critical_points.png`
- `q3_report.md`
