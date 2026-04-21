# Assignment 4 - Q2 (CLIP)

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
