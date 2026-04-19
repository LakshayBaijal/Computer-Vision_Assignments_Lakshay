import argparse
import json
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiTaskDataset
from metrics import compute_miou, compute_rmse
from models import MultiTaskUNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_wandb_run(config: Dict):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed. Continuing without wandb logging.")
        return None

    run = wandb.init(
        project=wandb_cfg.get("project", "q2-multitask"),
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.get("run_name", None),
        config=config,
    )
    return run


def evaluate(model, dataloader, device, seg_loss_fn, depth_loss_fn, lambda_seg, lambda_depth, num_classes):
    model.eval()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_depth_loss = 0.0
    total_miou = 0.0
    total_rmse = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            seg_logits, depth_pred = model(images)

            seg_loss = seg_loss_fn(seg_logits, masks)
            depth_loss = depth_loss_fn(depth_pred, depths)
            total_batch_loss = lambda_seg * seg_loss + lambda_depth * depth_loss

            miou = compute_miou(seg_logits, masks, num_classes=num_classes)
            rmse = compute_rmse(depth_pred, depths)

            total_loss += total_batch_loss.item()
            total_seg_loss += seg_loss.item()
            total_depth_loss += depth_loss.item()
            total_miou += miou
            total_rmse += rmse
            total_batches += 1

    return {
        "total_loss": total_loss / max(total_batches, 1),
        "seg_loss": total_seg_loss / max(total_batches, 1),
        "depth_loss": total_depth_loss / max(total_batches, 1),
        "miou": total_miou / max(total_batches, 1),
        "rmse": total_rmse / max(total_batches, 1),
    }


def save_training_curves(history: Dict[str, list], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_total_loss"]) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_total_loss"], label="Train Combined Loss")
    plt.plot(epochs, history["val_total_loss"], label="Val Combined Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Combined Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "combined_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_seg_loss"], label="Train Seg Loss")
    plt.plot(epochs, history["val_seg_loss"], label="Val Seg Loss")
    plt.plot(epochs, history["train_depth_loss"], label="Train Depth Loss")
    plt.plot(epochs, history["val_depth_loss"], label="Val Depth Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Individual Task Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "individual_loss_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["val_miou"], label="Validation mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "val_miou_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "val_rmse_curve.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train multi-task U-Net models for segmentation + depth")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--variant", type=str, default=None, choices=["vanilla", "noskip", "residual"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    config = load_config(config_path)

    if args.variant is not None:
        config["model"]["variant"] = args.variant
    if args.data_root is not None:
        config["dataset"]["root_dir"] = args.data_root
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    set_seed(config["experiment"]["seed"])

    q2_root = Path(__file__).resolve().parent
    output_root = q2_root / config["experiment"]["output_root"]
    run_name = f"{config['experiment']['name']}_{config['model']['variant']}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MultiTaskDataset(
        root_dir=config["dataset"]["root_dir"],
        split="train",
        image_size=config["dataset"]["image_size"],
        val_ratio=config["dataset"]["val_ratio"],
        seed=config["experiment"]["seed"],
        augment=True,
    )
    val_dataset = MultiTaskDataset(
        root_dir=config["dataset"]["root_dir"],
        split="val",
        image_size=config["dataset"]["image_size"],
        val_ratio=config["dataset"]["val_ratio"],
        seed=config["experiment"]["seed"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiTaskUNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        base_channels=config["model"]["base_channels"],
        variant=config["model"]["variant"],
    ).to(device)

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )

    lambda_seg = float(config["train"].get("lambda_seg", 1.0))
    lambda_depth = float(config["train"].get("lambda_depth", 1.0))

    scaler = GradScaler(enabled=bool(config["train"].get("amp", True) and torch.cuda.is_available()))

    history = {
        "train_total_loss": [],
        "train_seg_loss": [],
        "train_depth_loss": [],
        "val_total_loss": [],
        "val_seg_loss": [],
        "val_depth_loss": [],
        "val_miou": [],
        "val_rmse": [],
    }

    wandb_run = create_wandb_run(config)

    best_metric = float("inf")
    epochs = int(config["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()

        epoch_total_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_depth_loss = 0.0
        total_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=scaler.is_enabled()):
                seg_logits, depth_pred = model(images)
                seg_loss = seg_loss_fn(seg_logits, masks)
                depth_loss = depth_loss_fn(depth_pred, depths)
                total_loss = lambda_seg * seg_loss + lambda_depth * depth_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_total_loss += total_loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_depth_loss += depth_loss.item()
            total_batches += 1

            progress.set_postfix(
                total=f"{total_loss.item():.4f}",
                seg=f"{seg_loss.item():.4f}",
                depth=f"{depth_loss.item():.4f}",
            )

        train_total = epoch_total_loss / max(total_batches, 1)
        train_seg = epoch_seg_loss / max(total_batches, 1)
        train_depth = epoch_depth_loss / max(total_batches, 1)

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            seg_loss_fn=seg_loss_fn,
            depth_loss_fn=depth_loss_fn,
            lambda_seg=lambda_seg,
            lambda_depth=lambda_depth,
            num_classes=config["model"]["num_classes"],
        )

        history["train_total_loss"].append(train_total)
        history["train_seg_loss"].append(train_seg)
        history["train_depth_loss"].append(train_depth)

        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_seg_loss"].append(val_metrics["seg_loss"])
        history["val_depth_loss"].append(val_metrics["depth_loss"])
        history["val_miou"].append(val_metrics["miou"])
        history["val_rmse"].append(val_metrics["rmse"])

        print(
            f"Epoch {epoch}: "
            f"train_total={train_total:.4f}, train_seg={train_seg:.4f}, train_depth={train_depth:.4f}, "
            f"val_total={val_metrics['total_loss']:.4f}, val_mIoU={val_metrics['miou']:.4f}, val_RMSE={val_metrics['rmse']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/total_loss": train_total,
                    "train/seg_loss": train_seg,
                    "train/depth_loss": train_depth,
                    "val/total_loss": val_metrics["total_loss"],
                    "val/seg_loss": val_metrics["seg_loss"],
                    "val/depth_loss": val_metrics["depth_loss"],
                    "val/miou": val_metrics["miou"],
                    "val/rmse": val_metrics["rmse"],
                }
            )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "history": history,
        }
        torch.save(checkpoint_payload, run_dir / "last.pt")

        monitor_metric = val_metrics["total_loss"]
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            torch.save(checkpoint_payload, run_dir / "best.pt")

    save_training_curves(history, run_dir)

    history_path = run_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    if wandb_run is not None:
        try:
            import wandb

            wandb_run.log(
                {
                    "curves/combined_loss": wandb.Image(str(run_dir / "combined_loss_curve.png")),
                    "curves/individual_losses": wandb.Image(str(run_dir / "individual_loss_curves.png")),
                    "curves/val_miou": wandb.Image(str(run_dir / "val_miou_curve.png")),
                    "curves/val_rmse": wandb.Image(str(run_dir / "val_rmse_curve.png")),
                }
            )
        except Exception:
            pass
        wandb_run.finish()

    print(f"Training complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
