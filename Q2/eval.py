import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiTaskDataset
from metrics import compute_miou, compute_rmse
from models import MultiTaskUNet


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_qualitative_samples(model, test_loader, device, output_dir: Path, num_samples: int, num_classes: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    saved = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            seg_logits, depth_pred = model(images)
            pred_masks = torch.argmax(seg_logits, dim=1)

            for index in range(images.shape[0]):
                if saved >= num_samples:
                    return

                image_np = images[index].detach().cpu().permute(1, 2, 0).numpy()
                gt_mask_np = masks[index].detach().cpu().numpy()
                pred_mask_np = pred_masks[index].detach().cpu().numpy()
                gt_depth_np = depths[index].detach().cpu().squeeze(0).numpy()
                pred_depth_np = depth_pred[index].detach().cpu().squeeze(0).numpy()

                figure, axes = plt.subplots(1, 5, figsize=(18, 4))
                axes[0].imshow(image_np)
                axes[0].set_title("Input Image")
                axes[1].imshow(gt_mask_np, cmap="tab20", vmin=0, vmax=max(num_classes - 1, 1))
                axes[1].set_title("GT Segmentation")
                axes[2].imshow(pred_mask_np, cmap="tab20", vmin=0, vmax=max(num_classes - 1, 1))
                axes[2].set_title("Pred Segmentation")
                axes[3].imshow(gt_depth_np, cmap="viridis")
                axes[3].set_title("GT Depth")
                axes[4].imshow(pred_depth_np, cmap="viridis")
                axes[4].set_title("Pred Depth")

                for axis in axes:
                    axis.axis("off")

                figure.tight_layout()
                figure.savefig(output_dir / f"sample_{saved:02d}.png", dpi=200)
                plt.close(figure)
                saved += 1


def evaluate_test(model, test_loader, device, num_classes: int):
    model.eval()

    total_miou = 0.0
    total_rmse = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            seg_logits, depth_pred = model(images)
            batch_miou = compute_miou(seg_logits, masks, num_classes=num_classes)
            batch_rmse = compute_rmse(depth_pred, depths)

            total_miou += batch_miou
            total_rmse += batch_rmse
            total_batches += 1

    return {
        "test_miou": total_miou / max(total_batches, 1),
        "test_rmse": total_rmse / max(total_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained multi-task U-Net model on test set")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, default=None, choices=["vanilla", "noskip", "residual"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_visualizations", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    config = load_config(config_path)

    if args.variant is not None:
        config["model"]["variant"] = args.variant
    if args.data_root is not None:
        config["dataset"]["root_dir"] = args.data_root

    num_visualizations = args.num_visualizations
    if num_visualizations is None:
        num_visualizations = int(config["eval"].get("save_num_visualizations", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultiTaskUNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        base_channels=config["model"]["base_channels"],
        variant=config["model"]["variant"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    test_dataset = MultiTaskDataset(
        root_dir=config["dataset"]["root_dir"],
        split="test",
        image_size=config["dataset"]["image_size"],
        val_ratio=config["dataset"]["val_ratio"],
        seed=config["experiment"]["seed"],
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate_test(model, test_loader, device, num_classes=config["model"]["num_classes"])

    q2_root = Path(__file__).resolve().parent
    output_root = q2_root / config["experiment"]["output_root"]
    run_name = f"{config['experiment']['name']}_{config['model']['variant']}"
    eval_dir = output_root / run_name / "test_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_qualitative_samples(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=eval_dir / "qualitative_samples",
        num_samples=num_visualizations,
        num_classes=config["model"]["num_classes"],
    )

    print(f"Test mIoU: {metrics['test_miou']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Saved outputs to: {eval_dir}")


if __name__ == "__main__":
    main()
