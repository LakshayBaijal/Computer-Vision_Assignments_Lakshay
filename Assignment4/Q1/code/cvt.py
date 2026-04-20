import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokens_to_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, num_tokens, channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError("Token count does not match height*width")
    return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.flatten(2).transpose(1, 2)


class ConvTokenEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, channels, height, width = x.shape
        tokens = _map_to_tokens(x)
        tokens = self.norm(tokens)
        return tokens, height, width


class ConvProjection(nn.Module):
    def __init__(self, dim: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, int, int]:
        feature_map = _tokens_to_map(tokens, height, width)
        out_map = self.pointwise(self.depthwise(feature_map))
        out_tokens = _map_to_tokens(out_map)
        new_height, new_width = out_map.shape[-2], out_map.shape[-1]
        return out_tokens, new_height, new_width


class ConvAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_stride: int = 2,
        use_conv_proj: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_conv_proj = use_conv_proj
        if use_conv_proj:
            self.q_proj = ConvProjection(dim, stride=1)
            self.k_proj = ConvProjection(dim, stride=kv_stride)
            self.v_proj = ConvProjection(dim, stride=kv_stride)
            self.cls_q = nn.Linear(dim, dim)
            self.cls_k = nn.Linear(dim, dim)
            self.cls_v = nn.Linear(dim, dim)
        else:
            self.q_linear = nn.Linear(dim, dim)
            self.k_linear = nn.Linear(dim, dim)
            self.v_linear = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
        cls_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_conv_proj:
            q_tokens, _, _ = self.q_proj(tokens, height, width)
            k_tokens, _, _ = self.k_proj(tokens, height, width)
            v_tokens, _, _ = self.v_proj(tokens, height, width)

            if cls_token is not None:
                q_cls = self.cls_q(cls_token)
                k_cls = self.cls_k(cls_token)
                v_cls = self.cls_v(cls_token)
                q_tokens = torch.cat([q_cls, q_tokens], dim=1)
                k_tokens = torch.cat([k_cls, k_tokens], dim=1)
                v_tokens = torch.cat([v_cls, v_tokens], dim=1)
        else:
            if cls_token is not None:
                tokens = torch.cat([cls_token, tokens], dim=1)
            q_tokens = self.q_linear(tokens)
            k_tokens = self.k_linear(tokens)
            v_tokens = self.v_linear(tokens)

        q = self._to_heads(q_tokens)
        k = self._to_heads(k_tokens)
        v = self._to_heads(v_tokens)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(q_tokens.shape[0], q_tokens.shape[1], self.dim)
        out = self.proj_dropout(self.out_proj(out))
        return out, attn


class CvTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        kv_stride: int = 2,
        use_conv_proj: bool = True,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(
            dim=dim,
            num_heads=num_heads,
            kv_stride=kv_stride,
            use_conv_proj=use_conv_proj,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
        cls_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if cls_token is None:
            attn_input = self.norm1(tokens)
            attn_out, attn_map = self.attn(attn_input, height, width, cls_token=None)
            tokens = tokens + attn_out
            tokens = tokens + self.mlp(self.norm2(tokens))
            return tokens, None, attn_map

        full_tokens = torch.cat([cls_token, tokens], dim=1)
        full_norm = self.norm1(full_tokens)
        cls_norm = full_norm[:, :1, :]
        tok_norm = full_norm[:, 1:, :]

        attn_out, attn_map = self.attn(tok_norm, height, width, cls_token=cls_norm)
        full_tokens = full_tokens + attn_out
        full_tokens = full_tokens + self.mlp(self.norm2(full_tokens))

        cls_out = full_tokens[:, :1, :]
        tok_out = full_tokens[:, 1:, :]
        return tok_out, cls_out, attn_map


class CvTStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        token_kernel: int,
        token_stride: int,
        token_padding: int,
        kv_stride: int,
        use_conv_proj: bool,
    ) -> None:
        super().__init__()
        self.token_embed = ConvTokenEmbedding(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=token_kernel,
            stride=token_stride,
            padding=token_padding,
        )
        self.blocks = nn.ModuleList(
            [
                CvTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    kv_stride=kv_stride,
                    use_conv_proj=use_conv_proj,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, int, int, Optional[torch.Tensor], List[torch.Tensor]]:
        tokens, height, width = self.token_embed(x)
        attn_maps: List[torch.Tensor] = []
        for block in self.blocks:
            tokens, cls_token, attn_map = block(tokens, height, width, cls_token=cls_token)
            if return_attn:
                attn_maps.append(attn_map)
        return tokens, height, width, cls_token, attn_maps


class CvT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
        use_pos_embed: bool = False,
        use_conv_proj: bool = True,
    ) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("This implementation is configured for CIFAR-10 image_size=32")

        self.stage1 = CvTStage(
            in_channels=in_channels,
            embed_dim=64,
            depth=1,
            num_heads=1,
            token_kernel=7,
            token_stride=2,
            token_padding=3,
            kv_stride=2,
            use_conv_proj=use_conv_proj,
        )
        self.stage2 = CvTStage(
            in_channels=64,
            embed_dim=128,
            depth=2,
            num_heads=2,
            token_kernel=3,
            token_stride=2,
            token_padding=1,
            kv_stride=2,
            use_conv_proj=use_conv_proj,
        )
        self.stage3 = CvTStage(
            in_channels=128,
            embed_dim=256,
            depth=2,
            num_heads=4,
            token_kernel=3,
            token_stride=2,
            token_padding=1,
            kv_stride=1,
            use_conv_proj=use_conv_proj,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 16, 256))
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.use_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        all_attn: List[torch.Tensor] = []

        tokens, h1, w1, _, attn1 = self.stage1(x, cls_token=None, return_attn=return_attn)
        if return_attn:
            all_attn.extend(attn1)
        x1 = _tokens_to_map(tokens, h1, w1)

        tokens, h2, w2, _, attn2 = self.stage2(x1, cls_token=None, return_attn=return_attn)
        if return_attn:
            all_attn.extend(attn2)
        x2 = _tokens_to_map(tokens, h2, w2)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens, h3, w3, cls_token, attn3 = self.stage3(x2, cls_token=cls_token, return_attn=return_attn)
        if return_attn:
            all_attn.extend(attn3)

        final_tokens = torch.cat([cls_token, tokens], dim=1)
        if self.use_pos_embed:
            final_tokens = final_tokens + self.pos_embed
        final_tokens = self.norm(final_tokens)
        logits = self.head(final_tokens[:, 0])

        if return_attn:
            return logits, all_attn, (h3, w3)
        return logits
