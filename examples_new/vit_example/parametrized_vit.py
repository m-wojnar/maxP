"""Parametrized ViT — muP (Maximal Update Parametrization)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from maxp_new import ParametrizedModule, Parametrization


class Attention(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.q = ParametrizedModule(
            nn.Linear(dim, dim, bias=bias), width_dim=dim, layer_type="hidden",
        )
        self.k = ParametrizedModule(
            nn.Linear(dim, dim, bias=bias), width_dim=dim, layer_type="hidden",
        )
        self.v = ParametrizedModule(
            nn.Linear(dim, dim, bias=bias), width_dim=dim, layer_type="hidden",
        )
        self.attn_score = ParametrizedModule(
            lambda q, k: q @ k.transpose(-2, -1),
            width_dim=dim,
            layer_type="readout",
        )
        self.proj = ParametrizedModule(
            nn.Linear(dim, dim, bias=bias), width_dim=dim, layer_type="hidden",
        )

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = self.attn_score(q, k).softmax(dim=-1)
        return self.proj(attn @ v)


class Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp1 = ParametrizedModule(
            nn.Linear(dim, hidden, bias=bias), width_dim=dim, layer_type="hidden",
        )
        self.mlp2 = ParametrizedModule(
            nn.Linear(hidden, dim, bias=bias), width_dim=dim, layer_type="hidden",
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = F.gelu(self.mlp1(self.norm2(x)))
        x = x + self.mlp2(h)
        return x


class ParametrizedViT(nn.Module):
    """Tiny ViT with ParametrizedModule wrappers for muP."""

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
        patch_dim = patch_size * patch_size * 3
        n_patches = (32 // patch_size) ** 2

        self.patch_size = patch_size
        self.embed = ParametrizedModule(
            nn.Linear(patch_dim, embed_dim, bias=bias),
            width_dim=embed_dim,
            layer_type="embedding",
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, mlp_ratio, bias=bias) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = ParametrizedModule(
            nn.Linear(embed_dim, n_classes, bias=bias),
            width_dim=embed_dim,
            layer_type="readout",
        )

    def forward(self, x):
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


def make_parametrized_vit(
    embed_dim: int = 128,
    n_layers: int = 4,
    mlp_ratio: float = 2.0,
    lr_prefactor: float = 0.01,
    optimizer_type: str = "adam",
) -> tuple[ParametrizedViT, Parametrization]:
    """Build a ParametrizedViT and apply muP parametrization."""
    model = ParametrizedViT(
        embed_dim=embed_dim,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
    )
    param = Parametrization(
        model,
        optimizer_type=optimizer_type,
        alignment="full",
        lr_prefactor=lr_prefactor,
    )
    return model, param
