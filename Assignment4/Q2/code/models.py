from __future__ import annotations

import json
import os
from typing import Any, Dict

import clip
import torch
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imagenet_rn50(device: torch.device) -> torch.nn.Module:
    model = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model.to(device)


def load_clip_rn50(device: torch.device):
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    return model, preprocess


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def summarize_imagenet_rn50(model: torch.nn.Module) -> Dict[str, Any]:
    return {
        "model_name": "torchvision_resnet50_imagenet1k_v1",
        "total_params": _count_params(model),
        "stem": str(model.conv1),
        "layers": [str(model.layer1), str(model.layer2), str(model.layer3), str(model.layer4)],
        "pooling": str(model.avgpool),
        "head": str(model.fc),
        "num_classes": int(model.fc.out_features),
    }


def summarize_clip_rn50(model: torch.nn.Module) -> Dict[str, Any]:
    visual = model.visual
    return {
        "model_name": "openai_clip_rn50",
        "total_params_full_clip": _count_params(model),
        "total_params_visual": _count_params(visual),
        "visual_type": visual.__class__.__name__,
        "visual_stem": str(getattr(visual, "conv1", "N/A")),
        "visual_attention_pool": str(getattr(visual, "attnpool", "N/A")),
        "visual_output_dim": int(getattr(visual, "output_dim", -1)),
        "text_encoder_type": model.transformer.__class__.__name__,
        "text_width": int(model.transformer.width),
        "text_layers": int(model.transformer.layers),
        "text_vocab_size": int(model.vocab_size),
        "contrastive_logit_scale": float(model.logit_scale.exp().detach().cpu().item()),
    }


def save_architecture_report(path: str | None = None) -> Dict[str, Any]:
    device = get_device()
    imagenet_rn50 = load_imagenet_rn50(device)
    clip_model, _ = load_clip_rn50(device)

    report = {
        "device": str(device),
        "imagenet_rn50": summarize_imagenet_rn50(imagenet_rn50),
        "clip_rn50": summarize_clip_rn50(clip_model),
        "high_level_difference": (
            "Both use ResNet-style visual backbones, but CLIP RN50 is modified for "
            "contrastive image-text alignment (attention pooling and projection to a shared "
            "embedding space) and includes a separate text transformer encoder; "
            "ImageNet RN50 uses a 1000-way classification head."
        ),
    }

    out_path = (
        os.path.join(RESULTS_DIR, "model_architecture_report.json")
        if path is None
        else os.path.abspath(path)
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    report = save_architecture_report()
    print(json.dumps(report, indent=2))
