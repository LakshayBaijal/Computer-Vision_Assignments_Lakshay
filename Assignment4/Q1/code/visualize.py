import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def denormalize_cifar(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=img.device).view(3, 1, 1)
    out = img * std + mean
    return out.clamp(0.0, 1.0)



def attention_rollout(attn_maps: List[torch.Tensor], discard_ratio: float = 0.0) -> torch.Tensor:
    if len(attn_maps) == 0:
        raise ValueError("attn_maps must be non-empty")

    batch_size = attn_maps[0].shape[0]
    num_tokens = attn_maps[0].shape[-1]
    device = attn_maps[0].device
    rollout = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    for attn in attn_maps:
        attn_mean = attn.mean(dim=1)

        if discard_ratio > 0:
            flat = attn_mean.reshape(batch_size, -1)
            _, indices = flat.topk(k=int(flat.shape[-1] * discard_ratio), dim=-1, largest=False)
            flat.scatter_(1, indices, 0.0)
            attn_mean = flat.reshape(batch_size, num_tokens, num_tokens)

        identity = torch.eye(num_tokens, device=device).unsqueeze(0)
        attn_aug = attn_mean + identity
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(attn_aug, rollout)

    return rollout



def save_vit_attention_maps(
    model,
    image: torch.Tensor,
    save_dir: str,
    file_prefix: str,
) -> None:
    ensure_dir(save_dir)
    model.eval()

    with torch.no_grad():
        logits, attn_maps = model(image.unsqueeze(0), return_attn=True)

    del logits

    last = attn_maps[-1][0]
    cls_to_patch = last[:, 0, 1:]
    num_heads, num_patches = cls_to_patch.shape
    grid_size = int(np.sqrt(num_patches))

    for head in range(num_heads):
        heat = cls_to_patch[head].reshape(grid_size, grid_size).cpu().numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(heat, cmap="inferno")
        plt.colorbar()
        plt.title(f"Last-layer CLS Attention | Head {head}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{file_prefix}_last_head_{head}.png"), dpi=180)
        plt.close()

    agg = cls_to_patch.mean(dim=0).reshape(grid_size, grid_size).cpu().numpy()
    plt.figure(figsize=(3, 3))
    plt.imshow(agg, cmap="inferno")
    plt.colorbar()
    plt.title("Last-layer CLS Attention | Mean Heads")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_prefix}_last_agg.png"), dpi=180)
    plt.close()

    layer_dir = os.path.join(save_dir, f"{file_prefix}_layers")
    ensure_dir(layer_dir)
    for layer_idx, attn in enumerate(attn_maps):
        layer_map = attn[0].mean(dim=0)[0, 1:]
        heat = layer_map.reshape(grid_size, grid_size).cpu().numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(heat, cmap="inferno")
        plt.colorbar()
        plt.title(f"Layer {layer_idx} CLS->Patch")
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, f"layer_{layer_idx}.png"), dpi=180)
        plt.close()



def save_vit_rollout_map(model, image: torch.Tensor, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    model.eval()

    with torch.no_grad():
        _, attn_maps = model(image.unsqueeze(0), return_attn=True)

    rollout = attention_rollout(attn_maps)[0]
    cls_map = rollout[0, 1:]
    grid_size = int(np.sqrt(cls_map.shape[0]))
    cls_map = cls_map.reshape(grid_size, grid_size)
    cls_map = cls_map.unsqueeze(0).unsqueeze(0)
    cls_map = F.interpolate(cls_map, size=(32, 32), mode="bilinear", align_corners=False)

    image_denorm = denormalize_cifar(image).permute(1, 2, 0).cpu().numpy()
    heat = cls_map.squeeze().cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(image_denorm)
    plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title("ViT Attention Rollout")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def save_positional_embedding_similarity(model, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    pos = model.get_token_positional_embedding()
    if pos is None:
        raise ValueError("Model has no positional embedding (pos_type='none')")

    emb = pos[0]
    sim = emb @ emb.T
    sim = sim.detach().cpu().numpy()

    plt.figure(figsize=(5, 4))
    plt.imshow(sim, cmap="viridis")
    plt.colorbar()
    plt.title("Positional Embedding Self-Similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def save_cvt_attention_maps(model, image: torch.Tensor, save_dir: str, file_prefix: str) -> None:
    ensure_dir(save_dir)
    model.eval()

    with torch.no_grad():
        _, all_attn, hw = model(image.unsqueeze(0), return_attn=True)

    final_attn = all_attn[-1][0]
    h, w = hw

    cls_to_spatial = final_attn[:, 0, 1:]
    num_heads = cls_to_spatial.shape[0]
    spatial_tokens = cls_to_spatial.shape[-1]
    expected_tokens = h * w

    if spatial_tokens != expected_tokens:
        # Fall back to attention-token grid if K/V were downsampled.
        side = int(np.sqrt(spatial_tokens))
        if side * side != spatial_tokens:
            raise ValueError(
                f"Cannot reshape {spatial_tokens} attention tokens into square map."
            )
        h, w = side, side

    for head in range(num_heads):
        heat = cls_to_spatial[head].reshape(h, w).cpu().numpy()
        heat = torch.tensor(heat).unsqueeze(0).unsqueeze(0)
        heat = F.interpolate(heat, size=(32, 32), mode="bilinear", align_corners=False).squeeze().numpy()

        plt.figure(figsize=(3, 3))
        plt.imshow(heat, cmap="inferno")
        plt.colorbar()
        plt.title(f"CvT Final CLS Attention | Head {head}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{file_prefix}_head_{head}.png"), dpi=180)
        plt.close()

    mean_map = cls_to_spatial.mean(dim=0).reshape(h, w)
    mean_map = mean_map.unsqueeze(0).unsqueeze(0)
    mean_map = F.interpolate(mean_map, size=(32, 32), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(mean_map, cmap="inferno")
    plt.colorbar()
    plt.title("CvT Final CLS Attention | Mean Heads")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_prefix}_agg.png"), dpi=180)
    plt.close()
