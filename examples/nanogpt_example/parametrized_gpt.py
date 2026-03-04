"""Pre-LN GPT with every width-sensitive op wrapped in ParametrizedModule.

Same architecture as gpt.py, but ready for ABC parametrization:
every Linear, Embedding, and the QK^T dot product are wrapped so that
``Parametrization`` can auto-discover and control them.

Usage:
    python examples/nanogpt_example/parametrized_gpt.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from maxp import ParametrizedModule, Parametrization


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = ParametrizedModule(
            nn.Linear(d_model, 3 * d_model, bias=False),
            width_dim=d_model, layer_type="hidden")
        self.attn_score = ParametrizedModule(
            lambda q, k: q @ k.transpose(-2, -1),
            width_dim=self.head_dim, layer_type="readout")
        self.proj = ParametrizedModule(
            nn.Linear(d_model, d_model, bias=False),
            width_dim=d_model, layer_type="hidden")

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv.unbind(0)

        attn = self.attn_score(q, k)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = ParametrizedModule(
            nn.Linear(d_model, d_ff, bias=False),
            width_dim=d_model, layer_type="hidden")
        self.fc2 = ParametrizedModule(
            nn.Linear(d_ff, d_model, bias=False),
            width_dim=d_ff, layer_type="hidden")

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ParametrizedGPT(nn.Module):
    """Pre-LN GPT with all width-sensitive ops wrapped."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=512):
        super().__init__()
        self.tok_emb = ParametrizedModule(
            nn.Embedding(vocab_size, d_model),
            width_dim=d_model, layer_type="embedding")
        self.pos_emb = ParametrizedModule(
            nn.Embedding(max_seq_len, d_model),
            width_dim=d_model, layer_type="embedding")
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = ParametrizedModule(
            nn.Linear(d_model, vocab_size, bias=False),
            width_dim=d_model, layer_type="readout")

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def make_parametrized_gpt(vocab_size=256, d_model=128, n_heads=4, d_ff=512,
                          n_layers=4, max_seq_len=512, **param_kwargs):
    """Build ParametrizedGPT + apply Parametrization, return (model, param_groups)."""
    model = ParametrizedGPT(vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len)
    sample = torch.randint(0, vocab_size, (1, 8))
    defaults = dict(lr_prefactor=1.0, sample_input=sample)
    defaults.update(param_kwargs)
    param = Parametrization(model, **defaults)
    return model, param


if __name__ == "__main__":
    model, param = make_parametrized_gpt()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ParametrizedGPT: {n_params:,} params")
    print(f"Param groups: {len(param.param_groups)}")
    for g in param.param_groups:
        n_p = sum(p.numel() for p in g["params"])
        print(f"  {g['layer_name']:<45s}  lr={g['lr']:.6f}  params={n_p}")

    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    print(f"Input: {x.shape}, Output: {logits.shape}")
