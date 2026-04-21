# Assignment 4 - Computer Vision

This folder contains the implementation and outputs for Assignment 4.  
Current focus in this upload is **Q1 (ViT and CvT on CIFAR-10)** with training artifacts and visualizations.

## Repository Structure

- `Q1/code/` - model implementations, training scripts, and visualization runners
- `Q1/training_curves/` - checkpoints (`.pth`), metric json files, and loss/accuracy plots
- `Q1/visualizations/` - generated attention maps and interpretability outputs
- `Q2/` - placeholder for Q2
- `CV_Assignment_4.pdf` - assignment handout used as reference

## Input and Output Summary (Q1)

### Input
- **Dataset:** CIFAR-10 (`torchvision.datasets.CIFAR10`)
- **Image size:** `32x32` (no resizing)
- **Normalization:** CIFAR-10 mean/std used in training and evaluation
- **ViT positional embedding variants:** `none`, `1D`, `2D`, `sinusoidal`

### Output
- **Training outputs:** best checkpoints, curves, and metrics JSON
- **Quantitative outputs:** best validation accuracy per experiment
- **Visualization outputs:**
  - DINO attention maps
  - ViT CLS-to-patch maps (per-head + aggregated + all layers)
  - ViT attention rollout
  - Positional embedding self-similarity map
  - CvT CLS-to-spatial maps (per-head + aggregated)

## Q1 - Reported Results from `training_curves`

### 1.1 ViT Patch Size Variation

| Model | Best Val Acc (%) | Best Epoch | Params |
|---|---:|---:|---:|
| `vit_patch_2` | 73.68 | 49 | 384,330 |
| `vit_patch_4` | 75.40 | 42 | 364,362 |
| `vit_patch_8` | 65.01 | 47 | 376,650 |

### 1.1 ViT Hyperparameter Exploration

| Model | Config (dim, depth, heads, mlp_dim, lr) | Best Val Acc (%) |
|---|---|---:|
| `vit_hparam_1` | 128, 3, 2, 192, 4e-4 | 74.31 |
| `vit_hparam_2` | 144, 4, 3, 224, 3e-4 | 77.61 |
| `vit_hparam_3` | 160, 5, 4, 256, 2e-4 | 78.37 |

### 1.1 ViT Data Augmentations

| Model | Best Val Acc (%) |
|---|---:|
| `vit_aug_none` | 72.85 |
| `vit_aug_base` | 78.30 |
| `vit_aug_color` | 78.43 |
| `vit_aug_affine` | 76.19 |
| `vit_aug_autoaugment` | 76.07 |

### 1.1 Positional Embedding Ablation

| Positional Embedding | Best Val Acc (%) |
|---|---:|
| `none` | 70.48 |
| `1D` | 78.37 |
| `2D` | 78.53 |
| `sinusoidal` | 69.29 |

### 1.2 CvT Experiments

| Experiment | Model | Best Val Acc (%) | Params |
|---|---|---:|---:|
| Standard comparison | `best_vit_compare` | 78.63 | 950,410 |
| Standard comparison | `cvt_standard` | 80.38 | 2,937,546 |
| Positional embedding ablation | `cvt_with_pos_embed` | 80.53 | 2,941,898 |
| Conv projection ablation | `conv_embed_linear_proj_ablation` | 67.92 | 533,258 |

## Generated Visualization Outputs (Q1)

All outputs are present in `Q1/visualizations/`:

- `Q1/visualizations/1.1_ViT/dino attention maps/`
  - `img0/attn-head0.png ... attn-head5.png`
  - `img1/attn-head0.png ... attn-head5.png`
- `Q1/visualizations/1.1_ViT/vit_cifar10_maps/`
  - per-head and aggregated maps for 2 CIFAR-10 test images
  - layer-wise CLS attention maps
- `Q1/visualizations/1.1_ViT/attention_rollout/`
  - rollout overlays for 2 CIFAR-10 test images
- `Q1/visualizations/1.1_ViT/pos_embed_similarity/`
  - positional embedding self-similarity heatmap
- `Q1/visualizations/1.2_CvT/cvt_attention_maps/`
  - per-head and aggregated CvT maps for 2 CIFAR-10 test images

## Run Instructions

From `Q1/code/`:

```bash
python train.py --mode vit_hparam --epochs 50 --output_root ../training_curves
python train.py --mode vit_pos --epochs 50 --output_root ../training_curves
python train.py --mode cvt --epochs 50 --output_root ../training_curves

python run_vit_attention_maps.py --training_root ../training_curves --visualization_root ../visualizations/1.1_ViT --indices 0 1
python run_cvt_attention_maps.py --training_root ../training_curves --visualization_root ../visualizations/1.2_CvT/cvt_attention_maps --indices 0 1
```
