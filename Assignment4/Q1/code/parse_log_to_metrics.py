import re
import json


log_file = "vit_patch_2_log.txt"
metrics_file = "vit_patch_2_metrics.json"

train_loss, val_loss, val_acc = [], [], []

with open(log_file, "r") as f:
    for line in f:
        # Match lines like: [vit_patch_2] Epoch 1/50 | train_loss=1.8638 val_loss=1.7005 val_acc=37.73%
        m = re.search(r"train_loss=([\d.]+) val_loss=([\d.]+) val_acc=([\d.]+)%", line)
        if m:
            train_loss.append(float(m.group(1)))
            val_loss.append(float(m.group(2)))
            val_acc.append(float(m.group(3)))

metrics = {
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_acc": val_acc
}

with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved metrics to {metrics_file}")
