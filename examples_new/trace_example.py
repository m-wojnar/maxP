"""Example: trace two model configs and classify all ops by width-scaling."""

import torch
import torch.nn as nn

import sys; sys.path.insert(0, ".")
from maxp_new.trace import classify


# --- Models ---

class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TinyTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_out):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SimpleAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, 4 * d_model)
        self.ffn_down = nn.Linear(4 * d_model, d_model)
        self.head = nn.Linear(d_model, d_out)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn_down(torch.relu(self.ffn_up(self.ln2(x))))
        return self.head(x.mean(dim=1))


# --- Classify ---
# Scale all width dims: d_model doubles, n_heads fixed → head_dim also doubles.
# d_out (num classes) stays fixed — it's not a width dimension.

small = TinyTransformer(d_model=32, n_heads=4, d_out=10)
large = TinyTransformer(d_model=64, n_heads=4, d_out=10)

ops = classify(small, large, torch.randn(1, 5, 32), torch.randn(1, 5, 64))

print(f"{'idx':<4} {'op':<12} {'type':<12} {'param?':<8} {'param_name':<25}")
print("-" * 65)
for c in ops:
    print(f"{c.index:<4} {c.op:<12} {c.layer_type:<12} {'yes' if c.parametrized else 'no':<8} {c.param_name or '':<25}")
