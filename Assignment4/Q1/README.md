# Assignment 4 - Q1 (ViT and CvT)

This folder contains a full PyTorch pipeline for Assignment 4 Q1:
- ViT from scratch (manual Scaled Dot Product Attention + Multi-Head Attention)
- CvT with convolutional token embedding + convolutional projections
- CIFAR-10 training experiments and ablations
- Attention visualizations and rollout

## Directory Usage
- `code/`: all source scripts
- `training_curves/`: saved curves + metrics JSON + checkpoints
- `visualizations/`: attention maps and similarity heatmaps

## Kaggle Setup (Recommended)
1. Upload the whole `Q1` folder as a Kaggle Dataset.
    - Do not upload local `venv/` folder.
2. In notebook, copy from input to working:

```python
import os, glob, shutil
source_code_dirs = glob.glob('/kaggle/input/**/code', recursive=True)
source_q1_dir = os.path.dirname(source_code_dirs[0])
if os.path.exists('/kaggle/working/q1'):
    shutil.rmtree('/kaggle/working/q1')
shutil.copytree(source_q1_dir, '/kaggle/working/q1')
%cd /kaggle/working/q1/code
```

3. Install optional dependencies (if needed):

```bash
pip install -r /kaggle/working/q1/requirements.txt
```

## Training Commands
Run from `q1/code`:

```bash
python train.py --mode vit_patch --output_root ../training_curves
python train.py --mode vit_hparam --output_root ../training_curves
python train.py --mode vit_aug --output_root ../training_curves
python train.py --mode vit_pos --output_root ../training_curves
python train.py --mode cvt --output_root ../training_curves
python train.py --mode all --output_root ../training_curves

# Faster runs (recommended while debugging)
python train.py --mode all --epochs 20 --output_root ../training_curves

# Final long run (if needed)
python train.py --mode all --epochs 50 --output_root ../training_curves
```

## Visualization Usage
Example after training:

```python
import torch
from vit import ViT
from cvt import CvT
from visualize import (
    save_vit_attention_maps,
    save_vit_rollout_map,
    save_positional_embedding_similarity,
    save_cvt_attention_maps,
)

image = next(iter(test_loader))[0][0].to(device)

vit = ViT(pos_type='1D').to(device)
vit.load_state_dict(torch.load('...vit_checkpoint.pth', map_location=device))
save_vit_attention_maps(vit, image, '../visualizations/1.1_ViT/vit_cifar10_maps', 'sample0')
save_vit_rollout_map(vit, image, '../visualizations/1.1_ViT/attention_rollout/sample0.png')
save_positional_embedding_similarity(vit, '../visualizations/1.1_ViT/pos_embed_similarity/pos_sim.png')

cvt = CvT().to(device)
cvt.load_state_dict(torch.load('...cvt_checkpoint.pth', map_location=device))
save_cvt_attention_maps(cvt, image, '../visualizations/1.2_CvT/cvt_attention_maps', 'sample0')
```

## Notes
- CIFAR-10 images are not resized (32x32 kept as-is).
- ViT uses PreNorm as required.
- CvT stage-1 stride is not 4 (uses 2) to avoid spatial collapse on CIFAR-10.
- Save all plots/metrics within this repo for grading.
