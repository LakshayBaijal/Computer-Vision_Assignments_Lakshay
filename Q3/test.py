import torch

model.eval()
changed = 0
total = 0
logit_abs_diffs = []

with torch.no_grad():
    for pts, y in test_loader:
        pts = pts.to(device)          # [B, N, 3] or [B, 3, N] depending your code
        if pts.shape[1] != 3:         # convert to [B, 3, N] if needed
            pts = pts.transpose(1, 2)

        logits1 = model(pts)
        pred1 = logits1.argmax(dim=1)

        # real random permutation per sample
        B, C, N = pts.shape
        idx = torch.stack([torch.randperm(N, device=pts.device) for _ in range(B)], dim=0)
        pts_perm = torch.gather(pts, 2, idx.unsqueeze(1).expand(-1, C, -1))

        logits2 = model(pts_perm)
        pred2 = logits2.argmax(dim=1)

        changed += (pred1 != pred2).sum().item()
        total += B
        logit_abs_diffs.append((logits1 - logits2).abs().mean().item())

print("permutation_changed_pct =", 100.0 * changed / total)
print("mean_abs_logit_diff =", sum(logit_abs_diffs)/len(logit_abs_diffs))