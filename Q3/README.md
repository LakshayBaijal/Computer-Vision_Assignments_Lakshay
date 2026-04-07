# Q3 - Learning on Point Clouds (Simplified PointNet)

This folder contains a full implementation for Assignment Q3:

- `dataset.py`: ModelNet-10 `.ply` loader with centering/normalization and point sampling.
- `models.py`: Vanilla PointNet (no T-Net) with shared MLP + global max pool + classifier.
- `train.py`: Training/validation pipeline with saved curves and checkpoint.
- `analysis.py`: Permutation invariance, critical-point visualization, and sparse-critical robustness analysis.
- `config/default.yaml`: Default experiment settings.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Train

```bash
python train.py --config config/default.yaml --data_root ./ModelNet-10
```

Outputs are saved by default in `outputs/q3_pointnet/`:

- `best.pt`
- `history.json`
- `history.csv`
- `losses.png` (training/validation loss + accuracy curves)
- `test_metrics.json`

## 3) Run analysis (Q3.2 + Q3.3)

```bash
python analysis.py --config config/default.yaml --data_root ./ModelNet-10 --checkpoint outputs/q3_pointnet/best.pt
```

This saves:

- `analysis_metrics.json`
  - `test_accuracy`
  - `permutation_changed_pct`
  - `sparse_critical_accuracy`
- `critical_points.png` (at least 5 side-by-side visualizations)

## 4) Optional W&B logging

Set in `config/default.yaml`:

```yaml
wandb:
  enabled: true
  project: cv-assignment-q3
  entity: <your_entity_or_null>
  run_name: <optional_run_name>
```

Then run training normally.

## 5) Suggested report artifacts

For Moodle/submission, include:

- `q3_report.md`
- `losses.png`
- `critical_points.png`
- metrics from `test_metrics.json` and `analysis_metrics.json`
