from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import List, Sequence, Tuple

import clip
import torch
import torch.nn.functional as F
from PIL import Image

from imagenet_labels import build_prompt_texts, load_imagenet_class_index
from models import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))


@torch.no_grad()
def encode_text_prompts(
    model: torch.nn.Module, device: torch.device, prompts: Sequence[str]
) -> torch.Tensor:
    tokens = clip.tokenize(list(prompts)).to(device)
    text_features = model.encode_text(tokens)
    text_features = F.normalize(text_features, dim=-1)
    return text_features


@torch.no_grad()
def predict_topk_clip(
    model: torch.nn.Module,
    preprocess,
    image_path: str,
    text_features: torch.Tensor,
    k: int = 5,
) -> Tuple[List[int], List[float]]:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(text_features.device)
    image_features = model.encode_image(image)
    image_features = F.normalize(image_features, dim=-1)

    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1)
    top_probs, top_idx = probs.topk(k, dim=-1)
    return top_idx.squeeze(0).tolist(), top_probs.squeeze(0).tolist()


def get_val_images(imagenet_root: str, max_images: int = 8) -> List[str]:
    val_dir = os.path.join(imagenet_root, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    class_dirs = sorted(
        [d for d in glob.glob(os.path.join(val_dir, "*")) if os.path.isdir(d)]
    )
    images: List[str] = []
    for cls in class_dirs:
        for p in sorted(glob.glob(os.path.join(cls, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                images.append(p)
                if len(images) >= max_images:
                    return images
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-root", type=str, required=True)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(RESULTS_DIR, "top5_example_predictions.csv"),
    )
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")
    args = parser.parse_args()

    device = get_device()
    model, preprocess = clip.load("RN50", device=device)
    model.eval()

    class_index = load_imagenet_class_index(cache_dir=RESULTS_DIR)
    prompts = build_prompt_texts(class_index, template=args.prompt_template)
    text_features = encode_text_prompts(model, device, prompts)

    image_paths = get_val_images(args.imagenet_root, max_images=args.num_examples)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "rank",
                "pred_idx",
                "pred_synset",
                "pred_label",
                "probability",
            ]
        )
        for image_path in image_paths:
            top_idx, top_probs = predict_topk_clip(
                model=model,
                preprocess=preprocess,
                image_path=image_path,
                text_features=text_features,
                k=5,
            )
            for rank, (idx, prob) in enumerate(zip(top_idx, top_probs), start=1):
                row = class_index[idx]
                writer.writerow(
                    [image_path, rank, idx, row["synset"], row["label"], float(prob)]
                )

    print(f"Saved predictions to: {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
