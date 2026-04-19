import torch


def compute_miou(seg_logits: torch.Tensor, target_mask: torch.Tensor, num_classes: int, ignore_index: int = None) -> float:
    pred_mask = torch.argmax(seg_logits, dim=1)
    ious = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_cls = pred_mask == cls
        target_cls = target_mask == cls

        intersection = torch.logical_and(pred_cls, target_cls).sum().item()
        union = torch.logical_or(pred_cls, target_cls).sum().item()

        if union > 0:
            ious.append(intersection / union)

    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def compute_rmse(depth_pred: torch.Tensor, depth_target: torch.Tensor) -> float:
    mse = torch.mean((depth_pred - depth_target) ** 2)
    rmse = torch.sqrt(mse)
    return float(rmse.item())

