import argparse
import json
import os

import torch
import torchvision
import torchvision.transforms as transforms

from cvt import CvT
from visualize import save_cvt_attention_maps


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(base: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(base, rel))


def _get_test_dataset() -> torchvision.datasets.CIFAR10:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    return torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CvT attention visualizations.")
    parser.add_argument(
        "--training_root",
        type=str,
        default="../training_curves",
        help="Root directory containing 1.2_CvT outputs.",
    )
    parser.add_argument(
        "--visualization_root",
        type=str,
        default="../visualizations/1.2_CvT/cvt_attention_maps",
        help="Directory to save CvT attention outputs.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="CIFAR-10 test indices to visualize.",
    )
    args = parser.parse_args()

    metrics_path = os.path.join(args.training_root, "1.2_CvT", "standard_vs_cvt", "cvt_standard_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Missing CvT metrics at {metrics_path}. Run `python train.py --mode cvt` first."
        )

    metrics = _load_json(metrics_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = _resolve_path(script_dir, metrics["checkpoint"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train model again to generate .pth file."
        )

    model = CvT(num_classes=10, use_pos_embed=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    os.makedirs(args.visualization_root, exist_ok=True)
    dataset = _get_test_dataset()
    for idx in args.indices:
        image, label = dataset[idx]
        prefix = f"img_{idx}_label_{label}"
        save_cvt_attention_maps(model, image, args.visualization_root, prefix)

    print("Saved CvT attention visualizations successfully.")


if __name__ == "__main__":
    main()
