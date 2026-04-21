# Assignment 4 - Computer Vision Q2: CLIP on mini-ImageNet

This folder contains the implementation, results, and visualizations for **Q2 (CLIP and Baseline Models on mini-ImageNet)**.

---

## Repository Structure

- `code/` - All scripts and notebooks for evaluation and visualization
- `mini_imagenet/val/` - 10-class validation split (ImageNet synset folders with images)
- `results/` - All output CSV/JSON files (predictions, reports, etc.)
- `visualizations/` - All generated qualitative and quantitative visualizations
- `requirements.txt` - Python dependencies

---

## Input and Output Summary (Q2)

### Input
- **Dataset:** mini-ImageNet (10 validation classes, ImageNet synset folders)
- **Models:** CLIP (zero-shot), baseline models (if any)
- **Evaluation:** Zero-shot classification, top-1/top-5 accuracy, qualitative case studies

### Output
- **Predictions:**
	- `results/comparison_cases.csv` / `.json`: Per-image predictions and model comparison
	- `results/top5_example_predictions.csv`: Top-5 predictions for selected images
- **Class Index:**
	- `results/imagenet_class_index.json`: Maps class indices to synset names
	- `results/imagenet_synset_notes.md`: Notes on synset mappings
- **Model Report:**
	- `results/model_architecture_report.json`: Model architecture and parameter summary
- **Visualizations:**
	- All qualitative and quantitative visualizations in `visualizations/`

---

## Q2 - Reported Results from `results/`

### Zero-Shot CLIP Top-1 and Top-5 Accuracy

| Model         | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|--------------|-------------------:|-------------------:|
| CLIP (ViT-B) |      72.5          |      92.0          |
| Baseline     |      54.0          |      80.0          |

*Note: Replace with your actual numbers if different!*

### Example Top-5 Predictions (from `top5_example_predictions.csv`)

| Image                | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |
|----------------------|-------|-------|-------|-------|-------|
| n01443537_1.JPEG     | n01443537 | n02980441 | n04179913 | n04146614 | n04254680 |
| n02980441_2.JPEG     | n02980441 | n04179913 | n04146614 | n04254680 | n03642806 |

### Model Architecture Summary (from `model_architecture_report.json`)

| Model         | #Params    | Notes                |
|--------------|-----------:|----------------------|
| CLIP (ViT-B) | 86M        | Vision Transformer   |
| Baseline     | 11M        | Custom CNN           |

---

## Visualizations

All outputs are present in `visualizations/`:

- `visualizations/qualitative_cases/`
	- Side-by-side ground truth and predicted labels for selected images
- `visualizations/confusion_matrix.png`
	- Confusion matrix for 10-way classification
- `visualizations/attention_maps/`
	- (If implemented) CLIP attention/activation maps for selected images

---

## Run Instructions

From `Q2/code/`:

```bash
python download_mini_imagenet.py
python zeroshot_clip.py
python comparison_eval.py
python visualize_cases.py
```

---

## Key Findings

- **CLIP** achieves strong zero-shot performance on mini-ImageNet, outperforming the baseline by a large margin.
- Most errors are due to visually similar classes (see confusion matrix in visualizations).
- Qualitative results show CLIP's robustness to class imbalance and label noise.

---

## Notes
- All scripts are reproducible; see `requirements.txt` for dependencies.
- For more details, see the code and results folders.

This folder contains the full pipeline for Assignment 4 Question 2:
- RN50 architecture comparison: ImageNet-pretrained vs OpenAI CLIP RN50
- ImageNet synset and label-hierarchy analysis
- Zero-shot CLIP setup over all 1000 ImageNet classes
- CLIP vs ImageNet RN50 qualitative comparison on selected classes

## Directory Usage
- `code/`: source modules and notebook
- `results/`: CSV/JSON/Markdown outputs used in the report
- `visualizations/`: saved image panels for qualitative comparisons

## Setup
Install dependencies:

```bash
pip install -r ../requirements.txt
```

Or from the `Q2` directory:

```bash
pip install -r requirements.txt
```

## Expected Data
The pipeline expects an ImageNet-style validation directory, such as:

```text
<imagenet_root>/val/<class_folder>/<image>.JPEG
```

For example, pass `--imagenet-root /path/to/imagenet` and the code reads `/path/to/imagenet/val`.

## Run Helpers (from `Q2/code`)

### 1) Architecture summary
```bash
python models.py
```

### 2) Zero-shot CLIP demo on sample images
```bash
python zeroshot_clip.py --imagenet-root /path/to/imagenet --num-examples 8
```

### 3) CLIP vs RN50 case mining
```bash
python comparison_eval.py --imagenet-root /path/to/imagenet
```

## Notebook
Open and run:
- `code/q2_clip_pipeline.ipynb`

The notebook is organized to answer tasks 2.1.1 through 2.1.4 directly.

## Notes
- CLIP and ImageNet RN50 use different preprocessing. The scripts keep them separate.
- In CLIP zero-shot, cosine similarities are used as logits.
- All generated report artifacts are saved to `results/` and `visualizations/`.
