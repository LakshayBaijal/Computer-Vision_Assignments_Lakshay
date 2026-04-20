import os
import json
import matplotlib.pyplot as plt

def plot_metrics(metrics_path, out_dir, title_prefix):
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
    loss_path = os.path.join(out_dir, f"{title_prefix}_loss.png")
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
    acc_path = os.path.join(out_dir, f"{title_prefix}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved: {loss_path} and {acc_path}")

def scan_and_plot_all_vit():
    vit_root = os.path.join("training_curves", "1.1_ViT")
    for exp_type in os.listdir(vit_root):
        exp_path = os.path.join(vit_root, exp_type)
        if not os.path.isdir(exp_path):
            continue
        plot_dir = os.path.join(exp_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        for fname in os.listdir(exp_path):
            if fname.endswith("_metrics.json"):
                metrics_path = os.path.join(exp_path, fname)
                title_prefix = fname.replace("_metrics.json", "")
                plot_metrics(metrics_path, plot_dir, title_prefix)

def generate_vit_summary():
    vit_root = os.path.join("training_curves", "1.1_ViT")
    summary = {}
    for exp_type in os.listdir(vit_root):
        exp_path = os.path.join(vit_root, exp_type)
        if not os.path.isdir(exp_path):
            continue
        exp_summary = {}
        for fname in os.listdir(exp_path):
            if fname.endswith("_metrics.json"):
                metrics_path = os.path.join(exp_path, fname)
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                val_acc = metrics.get("val_acc", [])
                if val_acc:
                    best_acc = max(val_acc)
                    best_epoch = val_acc.index(best_acc) + 1
                    exp_summary[fname.replace("_metrics.json", "")] = {
                        "best_val_acc": best_acc,
                        "epoch": best_epoch
                    }
        if exp_summary:
            summary[exp_type] = exp_summary
    out_path = os.path.join(vit_root, "vit_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_path}")

if __name__ == "__main__":
    scan_and_plot_all_vit()
    generate_vit_summary()
