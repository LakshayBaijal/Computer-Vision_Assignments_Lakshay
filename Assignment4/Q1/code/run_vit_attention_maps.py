import argparse
import json
import os
from typing import Dict, List

import torch
import torchvision
import torchvision.transforms as transforms

from vit import ViT
from visualize import (
    save_positional_embedding_similarity,
    save_vit_attention_maps,
    save_vit_rollout_map,
)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _build_vit_from_hparams(best_hparams_path: str) -> ViT:
    cfg = _load_json(best_hparams_path)
    return ViT(
        image_size=32,
        patch_size=4,
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        mlp_dim=cfg["mlp_dim"],
        pos_type="1D",
    )


def _get_test_dataset() -> torchvision.datasets.CIFAR10:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    return torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


def _ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _resolve_path(base: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(base, rel))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ViT CIFAR-10 attention visualizations.")
    parser.add_argument(
        "--training_root",
        type=str,
        default="../training_curves",
        help="Root directory containing 1.1_ViT outputs.",
    )
    parser.add_argument(
        "--visualization_root",
        type=str,
        default="../visualizations/1.1_ViT",
        help="Root directory to save visualization outputs.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="CIFAR-10 test indices to visualize.",
    )
    args = parser.parse_args()

    vit_root = os.path.join(args.training_root, "1.1_ViT")
    best_hparams_path = os.path.join(vit_root, "hyperparameter_tuning", "best_hparams.json")
    pos_metrics_path = os.path.join(vit_root, "positional_embeddings", "vit_pos_1D_metrics.json")
    if not os.path.exists(pos_metrics_path):
        pos_metrics_path = best_hparams_path

    if not os.path.exists(best_hparams_path):
        raise FileNotFoundError(
            f"Missing best_hparams.json at {best_hparams_path}. Run vit_hparam training first."
        )
    if not os.path.exists(pos_metrics_path):
        raise FileNotFoundError(
            "Could not find a 1D positional embedding metrics file for checkpoint resolution."
        )

    model = _build_vit_from_hparams(best_hparams_path)
    metrics = _load_json(pos_metrics_path)
    checkpoint_rel = metrics["checkpoint"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = _resolve_path(script_dir, checkpoint_rel)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train model again to generate .pth file."
        )

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    vit_maps_dir = os.path.join(args.visualization_root, "vit_cifar10_maps")
    rollout_dir = os.path.join(args.visualization_root, "attention_rollout")
    pos_dir = os.path.join(args.visualization_root, "pos_embed_similarity")
    _ensure_dirs(vit_maps_dir, rollout_dir, pos_dir)

    dataset = _get_test_dataset()
    for idx in args.indices:
        image, label = dataset[idx]
        prefix = f"img_{idx}_label_{label}"
        save_vit_attention_maps(model, image, vit_maps_dir, prefix)
        save_vit_rollout_map(model, image, os.path.join(rollout_dir, f"{prefix}_rollout.png"))

    save_positional_embedding_similarity(model, os.path.join(pos_dir, "pos_embed_similarity.png"))
    print("Saved ViT visualizations successfully.")


if __name__ == "__main__":
    main()