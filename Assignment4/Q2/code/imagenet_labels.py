from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List, Tuple

IMAGENET_CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
IMAGENET_JSON_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _download(url: str, target: str) -> None:
    _ensure_parent(target)
    urllib.request.urlretrieve(url, target)


def load_imagenet_class_index(cache_dir: str | None = None) -> Dict[int, Dict[str, str]]:
    """
    Returns a mapping:
        idx -> {"synset": wnid, "label": human_label}
    """
    cache_dir = RESULTS_DIR if cache_dir is None else os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    json_path = os.path.join(cache_dir, "imagenet_class_index.json")

    if not os.path.exists(json_path):
        _download(IMAGENET_JSON_URL, json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: Dict[int, Dict[str, str]] = {}
    for k, (synset, label) in raw.items():
        mapping[int(k)] = {"synset": synset, "label": label.replace("_", " ")}
    return mapping


def load_imagenet_class_names(cache_dir: str | None = None) -> List[str]:
    """
    Returns ordered class names (1000), index aligned.
    """
    cache_dir = RESULTS_DIR if cache_dir is None else os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    txt_path = os.path.join(cache_dir, "imagenet_classes.txt")

    if not os.path.exists(txt_path):
        _download(IMAGENET_CLASS_INDEX_URL, txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def build_prompt_texts(
    class_index: Dict[int, Dict[str, str]],
    template: str = "a photo of a {}",
) -> List[str]:
    prompts = []
    for idx in range(1000):
        label = class_index[idx]["label"]
        prompts.append(template.format(label))
    return prompts


def idx_to_synset_label(
    class_index: Dict[int, Dict[str, str]],
) -> List[Tuple[int, str, str]]:
    out: List[Tuple[int, str, str]] = []
    for idx in range(1000):
        out.append((idx, class_index[idx]["synset"], class_index[idx]["label"]))
    return out
