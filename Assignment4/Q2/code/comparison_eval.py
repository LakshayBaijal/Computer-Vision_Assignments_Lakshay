from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import clip
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet50_Weights

from imagenet_labels import build_prompt_texts, load_imagenet_class_index
from models import get_device, load_imagenet_rn50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))


DEFAULT_CLASS_HINTS = [
    "goldfish",
    "school bus",
    "chain saw",
    "pineapple",
    "volcano",
    "airliner",
    "laptop",
    "soccer ball",
    "castle",
    "sewing machine",
]


def build_label_to_idx(class_index: Dict[int, Dict[str, str]]) -> Dict[str, int]:
    return {v["label"].lower(): k for k, v in class_index.items()}


def choose_target_class_indices(
    class_index: Dict[int, Dict[str, str]],
    class_hints: List[str],
) -> List[int]:
    label_to_idx = build_label_to_idx(class_index)
    selected = []
    for hint in class_hints:
        key = hint.strip().lower()
        if key in label_to_idx:
            selected.append(label_to_idx[key])
            continue
        # fallback substring match
        candidates = [idx for idx, v in class_index.items() if key in v["label"].lower()]
        if not candidates:
            raise ValueError(f"Could not match class hint: {hint}")
        selected.append(candidates[0])
    return selected


def list_images_for_class_synset(imagenet_root: str, synset: str) -> List[str]:
    class_dir = os.path.join(imagenet_root, "val", synset)
    if not os.path.isdir(class_dir):
        return []
    all_paths = sorted(glob.glob(os.path.join(class_dir, "*")))
    return [
        p
        for p in all_paths
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]


@torch.no_grad()
def infer_rn50(
    model: torch.nn.Module,
    preprocess: T.Compose,
    image_path: str,
    device: torch.device,
    k: int = 5,
) -> List[int]:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    logits = model(image)
    top_idx = logits.topk(k, dim=-1).indices.squeeze(0).tolist()
    return top_idx


@torch.no_grad()
def infer_clip(
    clip_model: torch.nn.Module,
    clip_preprocess,
    image_path: str,
    text_features: torch.Tensor,
    k: int = 5,
) -> List[int]:
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(text_features.device)
    image_features = clip_model.encode_image(image)
    image_features = F.normalize(image_features, dim=-1)
    logits = image_features @ text_features.T
    top_idx = logits.topk(k, dim=-1).indices.squeeze(0).tolist()
    return top_idx


def top5_hit(pred_idx: List[int], target_idx: int) -> bool:
    return target_idx in pred_idx


def write_synset_notes(path: str | None = None) -> None:
    note = """# ImageNet Synset Notes

## Label hierarchy used by ImageNet
ImageNet-1K categories are linked to WordNet noun synsets. WordNet forms a lexical-semantic hierarchy
with hypernym-hyponym relations (is-a relations), so each class belongs to a concept graph rather than
a flat independent list.

## What a synset means
A synset (synonym set) is a set of words/lemmas that represent a single concept sense in WordNet.
In ImageNet, each class index maps to one WordNet synset ID (wnid such as n02123045).

## Why synset-based grouping can be problematic for visual recognition
- Semantic similarity does not always equal visual similarity.
- Some synsets are visually broad and include high intra-class variation.
- Fine-grained neighboring synsets can be visually near-indistinguishable in unconstrained photos.

## Three visual differences inside one synset
1. Viewpoint and pose variation.
2. Scale, occlusion, and background-context variation.
3. Lighting, color appearance, and capture-domain variation (studio vs in-the-wild, motion blur, etc).
"""
    out = (
        os.path.join(RESULTS_DIR, "imagenet_synset_notes.md")
        if path is None
        else os.path.abspath(path)
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(note)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(RESULTS_DIR, "comparison_cases.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=os.path.join(RESULTS_DIR, "comparison_cases.json"),
    )
    parser.add_argument("--class-hints", nargs="*", default=DEFAULT_CLASS_HINTS)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    class_index = load_imagenet_class_index(cache_dir=RESULTS_DIR)
    target_classes = choose_target_class_indices(class_index, args.class_hints)

    rn50 = load_imagenet_rn50(device)
    rn50_preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()

    clip_model, clip_preprocess = clip.load("RN50", device=device)
    clip_model.eval()

    prompt_texts = build_prompt_texts(class_index)
    tokens = clip.tokenize(prompt_texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    rows: List[Dict[str, object]] = []
    selected_by_class: Dict[int, Dict[str, List[Dict[str, object]]]] = defaultdict(
        lambda: {"clip_better": [], "rn50_better": []}
    )

    for class_idx in target_classes:
        synset = class_index[class_idx]["synset"]
        label = class_index[class_idx]["label"]
        images = list_images_for_class_synset(args.imagenet_root, synset)
        if not images:
            print(f"Warning: No images found for {synset} ({label})")
            continue

        for image_path in images:
            rn50_top5 = infer_rn50(rn50, rn50_preprocess, image_path, device, k=5)
            clip_top5 = infer_clip(clip_model, clip_preprocess, image_path, text_features, k=5)

            rn50_ok = top5_hit(rn50_top5, class_idx)
            clip_ok = top5_hit(clip_top5, class_idx)

            record = {
                "target_idx": class_idx,
                "target_synset": synset,
                "target_label": label,
                "image_path": image_path,
                "rn50_top5": rn50_top5,
                "clip_top5": clip_top5,
                "rn50_top5_hit": rn50_ok,
                "clip_top5_hit": clip_ok,
            }
            rows.append(record)

            if clip_ok and not rn50_ok and len(selected_by_class[class_idx]["clip_better"]) < 2:
                selected_by_class[class_idx]["clip_better"].append(record)
            if rn50_ok and not clip_ok and len(selected_by_class[class_idx]["rn50_better"]) < 1:
                selected_by_class[class_idx]["rn50_better"].append(record)

            if (
                len(selected_by_class[class_idx]["clip_better"]) >= 2
                and len(selected_by_class[class_idx]["rn50_better"]) >= 1
            ):
                break

    final_records: List[Dict[str, object]] = []
    for class_idx in target_classes:
        final_records.extend(selected_by_class[class_idx]["clip_better"][:2])
        final_records.extend(selected_by_class[class_idx]["rn50_better"][:1])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df = pd.DataFrame(final_records)
    df.to_csv(args.out_csv, index=False)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2)

    write_synset_notes()

    print(f"Selected {len(final_records)} records")
    print(f"CSV: {os.path.abspath(args.out_csv)}")
    print(f"JSON: {os.path.abspath(args.out_json)}")

    missing = []
    for class_idx in target_classes:
        c = selected_by_class[class_idx]
        if len(c["clip_better"]) < 2 or len(c["rn50_better"]) < 1:
            missing.append(
                {
                    "idx": class_idx,
                    "synset": class_index[class_idx]["synset"],
                    "label": class_index[class_idx]["label"],
                    "clip_better_found": len(c["clip_better"]),
                    "rn50_better_found": len(c["rn50_better"]),
                }
            )
    if missing:
        print("Classes with incomplete 2+1 contrast:")
        for m in missing:
            print(m)


if __name__ == "__main__":
    main()
