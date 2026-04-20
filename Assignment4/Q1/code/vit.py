import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sinusoidal_position_embedding(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        proj_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k, attn_dropout=attn_dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        out, attn = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        out = self.proj_dropout(self.out_proj(out))
        return out, attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            proj_dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pos_type: str = "1D",
    ) -> None:
        super().__init__()
        if pos_type not in {"none", "1D", "2D", "sinusoidal"}:
            raise ValueError("pos_type must be one of: none, 1D, 2D, sinusoidal")
        if pos_type == "2D" and dim % 2 != 0:
            raise ValueError("For 2D learned positional embeddings, dim must be even")

        self.pos_type = pos_type
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.dim = dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        if pos_type == "1D":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        elif pos_type == "2D":
            self.row_embed = nn.Parameter(torch.zeros(1, self.grid_size, dim // 2))
            self.col_embed = nn.Parameter(torch.zeros(1, self.grid_size, dim // 2))
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, dim))
        elif pos_type == "sinusoidal":
            self.register_buffer(
                "pos_embed",
                _build_sinusoidal_position_embedding(self.num_patches + 1, dim),
                persistent=False,
            )

        self.pos_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=dim,
                    num_heads=heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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

        if self.pos_type == "1D":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.pos_type == "2D":
            nn.init.trunc_normal_(self.row_embed, std=0.02)
            nn.init.trunc_normal_(self.col_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def _make_2d_pos_embed(self, batch_size: int) -> torch.Tensor:
        row = self.row_embed.unsqueeze(2).expand(1, self.grid_size, self.grid_size, -1)
        col = self.col_embed.unsqueeze(1).expand(1, self.grid_size, self.grid_size, -1)
        spatial = torch.cat([row, col], dim=-1).reshape(1, self.num_patches, self.dim)
        pos = torch.cat([self.cls_pos_embed, spatial], dim=1)
        return pos.expand(batch_size, -1, -1)

    def get_token_positional_embedding(self) -> Optional[torch.Tensor]:
        if self.pos_type == "none":
            return None
        if self.pos_type == "1D":
            return self.pos_embed.detach().clone()
        if self.pos_type == "2D":
            return self._make_2d_pos_embed(batch_size=1).detach().clone()
        return self.pos_embed.detach().clone()

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        tokens = self.patch_embed(x)
        batch_size = tokens.size(0)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        if self.pos_type == "1D":
            tokens = tokens + self.pos_embed
        elif self.pos_type == "2D":
            tokens = tokens + self._make_2d_pos_embed(batch_size)
        elif self.pos_type == "sinusoidal":
            tokens = tokens + self.pos_embed.to(tokens.device)

        tokens = self.pos_drop(tokens)

        attn_maps: List[torch.Tensor] = []
        for layer in self.layers:
            tokens, attn = layer(tokens)
            if return_attn:
                attn_maps.append(attn)

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])

        if return_attn:
            return logits, attn_maps
        return logits
