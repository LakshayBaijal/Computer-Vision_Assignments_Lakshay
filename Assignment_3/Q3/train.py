from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from models import PointNetClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train simplified PointNet on ModelNet-10")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            logits = model(points)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def plot_curves(history: Dict[str, list], save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.data_root is not None:
        config["data"]["root_dir"] = args.data_root
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers

    set_seed(config["train"]["seed"])

    save_dir = Path(config["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = build_datasets(
        root_dir=config["data"]["root_dir"],
        num_points=config["data"]["num_points"],
        use_normals=config["data"].get("use_normals", False),
    )

    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    input_dim = 6 if config["data"].get("use_normals", False) else 3
    model = PointNetClassifier(
        num_classes=config["model"]["num_classes"],
        input_dim=input_dim,
        emb_dims=config["model"]["emb_dims"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"].get("label_smoothing", 0.0)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(5, config["train"]["epochs"] // 3), gamma=0.5)

    wandb_enabled = bool(config.get("wandb", {}).get("enabled", False))
    if wandb_enabled:
        import wandb

        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            name=config["wandb"].get("run_name"),
            config=config,
        )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
        for points, labels in progress:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            running_correct += (preds == labels).sum().item()
            running_samples += batch_size_now
            progress.set_postfix(loss=loss.item())

        train_loss = running_loss / max(1, running_samples)
        train_acc = running_correct / max(1, running_samples)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if wandb_enabled:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state": model.state_dict(),
                "class_names": train_ds.class_names,
                "config": config,
            }
            torch.save(checkpoint, save_dir / "best.pt")

    with open(save_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    import csv

    with open(save_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for idx in range(len(history["train_loss"])):
            writer.writerow(
                [
                    idx + 1,
                    history["train_loss"][idx],
                    history["val_loss"][idx],
                    history["train_acc"][idx],
                    history["val_acc"][idx],
                ]
            )

    plot_curves(history, save_dir / "losses.png")

    best_checkpoint = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    test_metrics = {"test_loss": test_loss, "test_accuracy": test_acc}
    with open(save_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    if wandb_enabled:
        wandb.summary["test/loss"] = test_loss
        wandb.summary["test/acc"] = test_acc
        wandb.finish()

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Test metrics: loss={test_loss:.4f}, acc={test_acc:.4f}")
    print(f"Outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
