from typing import List, Tuple

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: List[int], use_residual: bool = False):
        super().__init__()
        block = ResidualBlock if use_residual else DoubleConv

        self.blocks = nn.ModuleList()
        current_channels = in_channels
        for out_channels in channels:
            self.blocks.append(block(current_channels, out_channels))
            current_channels = out_channels

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            if idx < len(self.blocks) - 1:
                x = self.pool(x)
        return x, skips


class DecoderWithSkips(nn.Module):
    def __init__(self, channels: List[int], use_residual: bool = False):
        super().__init__()
        block = ResidualBlock if use_residual else DoubleConv

        reversed_channels = list(reversed(channels))
        self.up_convs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            self.up_convs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.blocks.append(block(out_ch * 2, out_ch))

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skips[:-1]))
        for up_conv, block, skip in zip(self.up_convs, self.blocks, skips):
            x = up_conv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class DecoderNoSkips(nn.Module):
    def __init__(self, channels: List[int], use_residual: bool = False):
        super().__init__()
        block = ResidualBlock if use_residual else DoubleConv

        reversed_channels = list(reversed(channels))
        self.up_convs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            self.up_convs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.blocks.append(block(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for up_conv, block in zip(self.up_convs, self.blocks):
            x = up_conv(x)
            x = block(x)
        return x


class MultiTaskUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        variant: str = "vanilla",
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in {"vanilla", "noskip", "residual"}:
            raise ValueError(f"Unsupported variant: {variant}")

        use_residual = variant == "residual"
        channel_list = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.encoder = Encoder(in_channels, channel_list, use_residual=use_residual)

        if variant == "noskip":
            self.decoder = DecoderNoSkips(channel_list, use_residual=use_residual)
        else:
            self.decoder = DecoderWithSkips(channel_list, use_residual=use_residual)

        final_channels = base_channels
        self.segmentation_head = nn.Conv2d(final_channels, num_classes, kernel_size=1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels // 2, 1, kernel_size=1),
        )

        self.variant = variant

    def forward(self, x: torch.Tensor):
        bottleneck, skips = self.encoder(x)

        if self.variant == "noskip":
            features = self.decoder(bottleneck)
        else:
            features = self.decoder(bottleneck, skips)

        seg_logits = self.segmentation_head(features)
        depth_pred = self.depth_head(features)
        return seg_logits, depth_pred
