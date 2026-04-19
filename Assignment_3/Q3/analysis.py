from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from models import PointNetClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Q3 analysis: permutation invariance + critical points")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_visualize", type=int, default=5)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def predict(model: PointNetClassifier, points: torch.Tensor) -> torch.Tensor:
    logits = model(points)
    return torch.argmax(logits, dim=1)


def pad_or_sample(points_xyz: torch.Tensor, target_points: int) -> torch.Tensor:
    num_points = points_xyz.shape[0]
    if num_points == target_points:
        return points_xyz
    if num_points > target_points:
        idx = torch.randperm(num_points)[:target_points]
        return points_xyz[idx]
    idx = torch.randint(0, num_points, (target_points - num_points,))
    return torch.cat([points_xyz, points_xyz[idx]], dim=0)


def evaluate_accuracy(model: PointNetClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            preds = predict(model, points)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def permutation_invariance_score(model: PointNetClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    changed = 0
    total = 0
    with torch.no_grad():
        for points, _ in tqdm(loader, desc="Permutation invariance"):
            points = points.to(device)
            original_preds = predict(model, points)

            permuted = points.clone()
            batch_size = permuted.size(0)
            n_points = permuted.size(2)
            for idx in range(batch_size):
                perm = torch.randperm(n_points, device=device)
                permuted[idx] = permuted[idx, :, perm]

            permuted_preds = predict(model, permuted)
            changed += (original_preds != permuted_preds).sum().item()
            total += points.size(0)

    return (changed / max(1, total)) * 100.0


def collect_critical_points(
    model: PointNetClassifier,
    points: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits, details = model(points, return_details=True)
        _ = logits
        critical_indices = details["critical_indices"][0].detach().cpu().numpy()

    unique_indices = np.unique(critical_indices)
    xyz = points[0, :3, :].detach().cpu().numpy().T
    critical_xyz = xyz[unique_indices]
    return xyz, critical_xyz


def visualize_critical_points(model: PointNetClassifier, loader: DataLoader, device: torch.device, save_path: Path, num_visualize: int):
    model.eval()
    samples: List[Tuple[np.ndarray, np.ndarray]] = []

    for points, _ in loader:
        for idx in range(points.size(0)):
            batch_points = points[idx : idx + 1].to(device)
            xyz, critical_xyz = collect_critical_points(model, batch_points)
            samples.append((xyz, critical_xyz))
            if len(samples) >= num_visualize:
                break
        if len(samples) >= num_visualize:
            break

    rows = len(samples)
    fig = plt.figure(figsize=(10, 4 * rows))

    for i, (xyz, critical_xyz) in enumerate(samples):
        ax1 = fig.add_subplot(rows, 2, 2 * i + 1, projection="3d")
        ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c="royalblue")
        ax1.set_title(f"Sample {i + 1}: Original")
        ax1.set_axis_off()

        ax2 = fig.add_subplot(rows, 2, 2 * i + 2, projection="3d")
        ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c="gray", alpha=0.2)
        ax2.scatter(critical_xyz[:, 0], critical_xyz[:, 1], critical_xyz[:, 2], s=8, c="crimson")
        ax2.set_title(f"Sample {i + 1}: Critical ({len(critical_xyz)} pts)")
        ax2.set_axis_off()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def sparse_critical_accuracy(
    model: PointNetClassifier,
    loader: DataLoader,
    device: torch.device,
    target_points: int,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Sparse critical evaluation"):
            points = points.to(device)
            labels = labels.to(device)

            _, details = model(points, return_details=True)
            critical_idx = details["critical_indices"].detach().cpu().numpy()

            sparse_batch = []
            for b in range(points.size(0)):
                xyz = points[b, :3, :].detach().cpu().T
                unique_idx = np.unique(critical_idx[b])
                critical_xyz = xyz[unique_idx]
                sparse_xyz = pad_or_sample(critical_xyz, target_points)

                if points.size(1) > 3:
                    zeros_extra = torch.zeros((target_points, points.size(1) - 3), dtype=sparse_xyz.dtype)
                    sparse_full = torch.cat([sparse_xyz, zeros_extra], dim=1)
                else:
                    sparse_full = sparse_xyz

                sparse_batch.append(sparse_full.T)

            sparse_points = torch.stack(sparse_batch).to(device)
            preds = predict(model, sparse_points)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / max(1, total)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.data_root is not None:
        config["data"]["root_dir"] = args.data_root

    output_dir = Path(args.save_dir or config["train"]["save_dir"]) 
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint or (output_dir / "best.pt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = build_datasets(
        root_dir=config["data"]["root_dir"],
        num_points=config["data"]["num_points"],
        use_normals=config["data"].get("use_normals", False),
    )
    _ = train_ds, val_ds

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = 6 if config["data"].get("use_normals", False) else 3
    model = PointNetClassifier(
        num_classes=config["model"]["num_classes"],
        input_dim=input_dim,
        emb_dims=config["model"]["emb_dims"],
        dropout=config["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_acc = evaluate_accuracy(model, test_loader, device)
    changed_pct = permutation_invariance_score(model, test_loader, device)
    sparse_acc = sparse_critical_accuracy(model, test_loader, device, target_points=config["data"]["num_points"])

    visualize_critical_points(
        model,
        test_loader,
        device,
        save_path=output_dir / "critical_points.png",
        num_visualize=args.num_visualize,
    )

    metrics = {
        "test_accuracy": test_acc,
        "permutation_changed_pct": changed_pct,
        "sparse_critical_accuracy": sparse_acc,
    }

    with open(output_dir / "analysis_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Analysis complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
