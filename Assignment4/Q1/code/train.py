import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from cvt import CvT
from vit import ViT


@dataclass
class TrainConfig:
    model_name: str
    epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    patience: int = 5
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def maybe_load_best_hparams(base_dir: str) -> Dict | None:
    best_file = os.path.join(base_dir, "1.1_ViT", "hyperparameter_tuning", "best_hparams.json")
    if not os.path.exists(best_file):
        return None
    with open(best_file, "r", encoding="utf-8") as file:
        return json.load(file)


def get_cifar10_loaders(batch_size: int, aug_name: str = "base") -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    aug_options = {
        "none": [transforms.ToTensor(), normalize],
        "base": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],
        "color": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ],
        "affine": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ],
        "autoaugment": [
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize,
        ],
    }
    if aug_name not in aug_options:
        raise ValueError(f"Unknown aug_name: {aug_name}")

    train_transform = transforms.Compose(aug_options[aug_name])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
    return loss_sum / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = loss_sum / len(loader.dataset)
    acc = 100.0 * correct / len(loader.dataset)
    return avg_loss, acc


def plot_curves(history: Dict[str, List[float]], save_path: str, title: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title(f"{title} Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="val_acc")
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def fit_model(
    model: nn.Module,
    config: TrainConfig,
    output_dir: str,
    aug_name: str = "base",
) -> Dict:
    ensure_dir(output_dir)
    set_seed(config.seed)
    device = get_device()
    model = model.to(device)

    train_loader, val_loader = get_cifar10_loaders(batch_size=config.batch_size, aug_name=aug_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0


    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{config.model_name}] Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Stop training if accuracy >= 80%
        if val_acc >= 80.0:
            print(f"Reached {val_acc:.2f}% accuracy at epoch {epoch + 1}, stopping early!")
            break

        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = os.path.join(output_dir, f"{config.model_name}_best.pth")
    torch.save(model.state_dict(), ckpt_path)

    curves_path = os.path.join(output_dir, f"{config.model_name}_curves.png")
    plot_curves(history, curves_path, config.model_name)

    metrics = {
        "model_name": config.model_name,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
        "num_params": count_parameters(model),
        "augmentation": aug_name,
        "config": asdict(config),
        "checkpoint": ckpt_path,
        "curve_plot": curves_path,
        # Add full history for plotting
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
    }
    with open(os.path.join(output_dir, f"{config.model_name}_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def run_vit_patch_size_variation(base_dir: str, epochs: int = 20) -> List[Dict]:
    output_dir = os.path.join(base_dir, "1.1_ViT", "patch_size_variation")
    results = []
    # Use smaller ViT: dim=128, depth=3, heads=2, mlp_dim=192
    for patch_size in [2, 4, 8]:
        model = ViT(
            image_size=32,
            patch_size=patch_size,
            dim=128,
            depth=3,
            heads=2,
            mlp_dim=192,
            pos_type="1D",
        )
        config = TrainConfig(model_name=f"vit_patch_{patch_size}", epochs=epochs)
        result = fit_model(model, config, output_dir=output_dir, aug_name="base")
        result["patch_size"] = patch_size
        results.append(result)
    return results


def run_vit_hparam_search(base_dir: str, epochs: int = 20) -> Dict:
    output_dir = os.path.join(base_dir, "1.1_ViT", "hyperparameter_tuning")
    # Smaller models for speed
    candidates = [
        dict(dim=128, depth=3, heads=2, mlp_dim=192, lr=4e-4),
        dict(dim=144, depth=4, heads=3, mlp_dim=224, lr=3e-4),
        dict(dim=160, depth=5, heads=4, mlp_dim=256, lr=2e-4),
    ]

    best = None
    for index, params in enumerate(candidates, start=1):
        model = ViT(
            image_size=32,
            patch_size=4,
            dim=params["dim"],
            depth=params["depth"],
            heads=params["heads"],
            mlp_dim=params["mlp_dim"],
            pos_type="1D",
        )
        config = TrainConfig(model_name=f"vit_hparam_{index}", lr=params["lr"], epochs=epochs)
        metrics = fit_model(model, config, output_dir=output_dir, aug_name="base")
        metrics.update(params)
        if best is None or metrics["best_val_acc"] > best["best_val_acc"]:
            best = metrics

    with open(os.path.join(output_dir, "best_hparams.json"), "w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)
    return best


def run_vit_data_augmentations(base_dir: str, best_cfg: Dict, epochs: int = 20) -> List[Dict]:
    output_dir = os.path.join(base_dir, "1.1_ViT", "data_augmentations")
    results = []
    for aug in ["none", "base", "color", "affine", "autoaugment"]:
        model = ViT(
            image_size=32,
            patch_size=4,
            dim=best_cfg.get("dim", 128),
            depth=best_cfg.get("depth", 3),
            heads=best_cfg.get("heads", 2),
            mlp_dim=best_cfg.get("mlp_dim", 192),
            pos_type="1D",
        )
        config = TrainConfig(model_name=f"vit_aug_{aug}", lr=best_cfg.get("lr", 4e-4), epochs=epochs)
        metrics = fit_model(model, config, output_dir=output_dir, aug_name=aug)
        metrics["augmentation"] = aug
        results.append(metrics)
    return results


def run_vit_positional_embedding_ablation(base_dir: str, best_cfg: Dict, epochs: int = 20) -> List[Dict]:
    output_dir = os.path.join(base_dir, "1.1_ViT", "positional_embeddings")
    results = []
    for pos_type in ["none", "1D", "2D", "sinusoidal"]:
        model = ViT(
            image_size=32,
            patch_size=4,
            dim=best_cfg.get("dim", 128),
            depth=best_cfg.get("depth", 3),
            heads=best_cfg.get("heads", 2),
            mlp_dim=best_cfg.get("mlp_dim", 192),
            pos_type=pos_type,
        )
        config = TrainConfig(model_name=f"vit_pos_{pos_type}", lr=best_cfg.get("lr", 4e-4), epochs=epochs)
        metrics = fit_model(model, config, output_dir=output_dir, aug_name="base")
        metrics["pos_type"] = pos_type
        results.append(metrics)
    return results


def run_cvt_experiments(base_dir: str, best_vit_cfg: Dict, epochs: int = 20) -> Dict[str, Dict]:
    out_standard = os.path.join(base_dir, "1.2_CvT", "standard_vs_cvt")
    out_pos = os.path.join(base_dir, "1.2_CvT", "ablation_pos_embed")
    out_conv = os.path.join(base_dir, "1.2_CvT", "ablation_conv_proj")

    best_vit = ViT(
        image_size=32,
        patch_size=4,
        dim=best_vit_cfg["dim"],
        depth=best_vit_cfg["depth"],
        heads=best_vit_cfg["heads"],
        mlp_dim=best_vit_cfg["mlp_dim"],
        pos_type="1D",
    )
    vit_metrics = fit_model(
        best_vit,
        TrainConfig(model_name="best_vit_compare", lr=best_vit_cfg["lr"], epochs=epochs),
        output_dir=out_standard,
    )

    cvt = CvT(num_classes=10, use_pos_embed=False)
    cvt_metrics = fit_model(
        cvt,
        TrainConfig(model_name="cvt_standard", lr=3e-4, epochs=epochs),
        output_dir=out_standard,
    )

    cvt_with_pos = CvT(num_classes=10, use_pos_embed=True)
    cvt_pos_metrics = fit_model(
        cvt_with_pos,
        TrainConfig(model_name="cvt_with_pos_embed", lr=3e-4, epochs=epochs),
        output_dir=out_pos,
    )

    cvt_linear_proj = CvT(num_classes=10, use_pos_embed=False, use_conv_proj=False)
    linear_proj_metrics = fit_model(
        cvt_linear_proj,
        TrainConfig(model_name="cvt_linear_proj_ablation", lr=3e-4, epochs=epochs),
        output_dir=out_conv,
    )

    summary = {
        "standard_vs_cvt": {
            "vit": vit_metrics,
            "cvt": cvt_metrics,
        },
        "ablation_pos_embed": {
            "cvt_no_pos": cvt_metrics,
            "cvt_with_pos": cvt_pos_metrics,
        },
        "ablation_conv_proj": {
            "cvt_linear_proj": linear_proj_metrics,
        },
    }
    with open(os.path.join(base_dir, "1.2_CvT", "cvt_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViT and CvT on CIFAR-10 for Assignment 4 Q1")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["vit_patch", "vit_hparam", "vit_aug", "vit_pos", "cvt", "all"],
    )
    parser.add_argument("--output_root", type=str, default="../training_curves")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    ensure_dir(args.output_root)

    if args.mode == "vit_patch":
        run_vit_patch_size_variation(args.output_root, epochs=args.epochs)
        return

    if args.mode == "vit_hparam":
        run_vit_hparam_search(args.output_root, epochs=args.epochs)
        return

    if args.mode == "vit_aug":
        best_cfg = maybe_load_best_hparams(args.output_root)
        if best_cfg is None:
            best_cfg = run_vit_hparam_search(args.output_root, epochs=args.epochs)
        run_vit_data_augmentations(args.output_root, best_cfg, epochs=args.epochs)
        return

    if args.mode == "vit_pos":
        best_cfg = maybe_load_best_hparams(args.output_root)
        if best_cfg is None:
            best_cfg = run_vit_hparam_search(args.output_root, epochs=args.epochs)
        run_vit_positional_embedding_ablation(args.output_root, best_cfg, epochs=args.epochs)
        return

    if args.mode == "cvt":
        best_cfg = maybe_load_best_hparams(args.output_root)
        if best_cfg is None:
            best_cfg = run_vit_hparam_search(args.output_root, epochs=args.epochs)
        run_cvt_experiments(args.output_root, best_cfg, epochs=args.epochs)
        return

    best_cfg = maybe_load_best_hparams(args.output_root)
    if best_cfg is None:
        best_cfg = run_vit_hparam_search(args.output_root, epochs=args.epochs)
    run_vit_patch_size_variation(args.output_root, epochs=args.epochs)
    run_vit_data_augmentations(args.output_root, best_cfg, epochs=args.epochs)
    run_vit_positional_embedding_ablation(args.output_root, best_cfg, epochs=args.epochs)
    run_cvt_experiments(args.output_root, best_cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
