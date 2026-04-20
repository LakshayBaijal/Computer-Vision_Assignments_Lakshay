import os
import json
import matplotlib.pyplot as plt

# Root directory for all metrics
PATCH_METRICS_DIR = os.path.join("training_curves", "1.1_ViT", "patch_size_variation")
PATCH_METRICS = [
    ("vit_patch_2_metrics.json", "ViT Patch Size 2"),
    ("vit_patch_4_metrics.json", "ViT Patch Size 4"),
    ("vit_patch_8_metrics.json", "ViT Patch Size 8"),
]

OUTPUT_DIR = os.path.join("training_curves", "1.1_ViT", "patch_size_variation", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_metrics(metrics_path, title_prefix):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    train_loss = metrics.get("train_loss", [])
    val_loss = metrics.get("val_loss", [])
    val_acc = metrics.get("val_acc", [])

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, f"{title_prefix.replace(' ', '_').lower()}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(val_acc, label="Validation Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} - Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(OUTPUT_DIR, f"{title_prefix.replace(' ', '_').lower()}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved: {loss_path} and {acc_path}")

if __name__ == "__main__":
    for fname, title in PATCH_METRICS:
        metrics_path = os.path.join(PATCH_METRICS_DIR, fname)
        if os.path.exists(metrics_path):
            plot_metrics(metrics_path, title)
        else:
            print(f"Metrics file not found: {metrics_path}")
