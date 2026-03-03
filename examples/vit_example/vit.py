"""Vanilla ViT — Standard Parametrization (SP) baseline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj(attn @ v)


class Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp1 = nn.Linear(dim, hidden, bias=bias)
        self.mlp2 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = F.gelu(self.mlp1(self.norm2(x)))
        x = x + self.mlp2(h)
        return x


class ViT(nn.Module):
    """Tiny ViT for CIFAR-10 (32x32 images, patch_size=4 -> 64 patches)."""

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 4,
        mlp_ratio: float = 2.0,
        patch_size: int = 4,
        n_classes: int = 10,
        bias: bool = False,
    ):
        super().__init__()
        patch_dim = patch_size * patch_size * 3  # 48 for patch_size=4
        n_patches = (32 // patch_size) ** 2      # 64 for patch_size=4

        self.patch_size = patch_size
        self.embed = nn.Linear(patch_dim, embed_dim, bias=bias)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, mlp_ratio, bias=bias) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes, bias=bias)

    def forward(self, x):
        # Patchify: (B, 3, 32, 32) -> (B, n_patches, patch_dim)
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, -1, p * p * C)

        x = self.embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])
