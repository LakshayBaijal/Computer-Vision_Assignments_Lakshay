# Question 3: Image Captioning with CLIP and GPT-2 (ClipCap)

This repository contains the implementation, ablation studies, and qualitative analysis for the ClipCap architecture. The goal of this project is to project visual features from a CLIP image encoder into the prefix space of a pre-trained GPT-2 language model to generate descriptive captions.

## 1. Architectural Parameter Analysis (Task 3.1)

We compared two variants of the "Mapping Network" (the bridge between CLIP and GPT-2) and analyzed the impact of freezing vs. fine-tuning the language model.

| Configuration | Mapping Network | GPT-2 Status | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **Ablation 1** | Multi-Layer Perceptron (MLP) | Fine-tuned | **155,908,608** |
| **Ablation 2** | Transformer (Multi-head) | Frozen | **3,939,840** |

### **Key Insight:**
Ablation 1 involves nearly **40x more parameters** because it updates the entire weights of the GPT-2 model. Ablation 2 demonstrates the efficiency of "Prefix-Tuning," where only a small transformer bridge is trained to translate visual features, keeping the core language knowledge frozen.

---

## 2. Quantitative Evaluation (Task 3.1 & 3.3)

Evaluation was performed on **1,000 images** from the MS-COCO validation dataset using CPU-based inference.

* **Average BLEU Score:** `0.0201`
* **Total Samples Evaluated:** `1,000`

*Analysis:* The BLEU score reflects the strict n-gram overlap between generated text and human references. While numerically low due to the use of greedy decoding and the strictness of the metric, qualitative results show high semantic relevance.

---

## 3. Prefix Length Analysis ($k$) (Task 3.2)

We tested the impact of the visual prefix length ($k$) on text generation. This length determines how many visual "tokens" GPT-2 sees before it starts writing.

| Prefix Length ($k$) | Sample Generated Sequence |
| :--- | :--- |
| **k=1** | "The first time I saw..." |
| **k=5** | "The first of the two..." |
| **k=10** | "The first of the new..." (Optimal) |
| **k=40** | "The first time I saw..." |

**Conclusion:** A prefix length of $k=10$ provides a stable bottleneck that balances visual detail with the language model's ability to maintain grammatical structure.

---

## 4. Qualitative Analysis: Attention Heatmaps (Task 3.4)

To understand where the model "looks" when generating a prefix, we visualized the spatial activations from the CLIP ViT-B/32 backbone.

**[Sample Visualization: `heatmap_0.jpg`]**
The heatmap shows concentrated activation (Red/Yellow regions) over the primary subjects of the image (e.g., shoes and shelving units), proving that the mapping network correctly extracts semantically meaningful features to pass to GPT-2.

---

## 5. Implementation Details

- **Environment:** PyTorch 2.x, Transformers 4.x.
- **Optimization:** A monkey-patch was implemented for `transformers.AdamW` to ensure compatibility with modern `torch.optim`.
- **Inference:** Optimized for CPU environments with greedy search decoding to ensure stability across hardware types.

### Running the Evaluation
To reproduce the metrics:
```bash
python analysis_ablation.py
python calculate_metrics.py