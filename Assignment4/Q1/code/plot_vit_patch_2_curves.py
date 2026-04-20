import json
import matplotlib.pyplot as plt
import os

# Define output directory for ViT curves
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Results", "Q1", "training_curves", "ViT"
)
os.makedirs(output_dir, exist_ok=True)

# Load metrics from vit_patch_2_metrics.json
with open("vit_patch_2_metrics.json", "r") as f:
    metrics = json.load(f)

train_loss = metrics.get("train_loss", [])
val_loss = metrics.get("val_loss", [])
val_acc = metrics.get("val_acc", [])

# Plot Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ViT Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_curve_path = os.path.join(output_dir, "vit_patch_2_loss_curve.png")
plt.savefig(loss_curve_path)
plt.close()

# Plot Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(val_acc, label="Validation Accuracy (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ViT Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
acc_curve_path = os.path.join(output_dir, "vit_patch_2_accuracy_curve.png")
plt.savefig(acc_curve_path)
plt.close()

print(f"Saved {loss_curve_path} and {acc_curve_path}")
