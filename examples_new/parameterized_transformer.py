"""Pre-LN Transformer with every width-sensitive op wrapped in ParametrizedModule.

Same architecture as transformer.py, but ready for ABC parametrization:
every Linear, Embedding, and the QK^T dot product are wrapped so that
``Parametrization`` can auto-discover and control them.

Usage:
    python examples_new/parameterized_transformer.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys; sys.path.insert(0, ".")
from maxp_new.utils import ParametrizedModule
from maxp_new.parametrization import Parametrization


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP: out = down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = ParametrizedModule(
            nn.Linear(d_model, d_ff, bias=False),
            width_dim=d_model, layer_type="hidden")
        self.up = ParametrizedModule(
            nn.Linear(d_model, d_ff, bias=False),
            width_dim=d_model, layer_type="hidden")
        self.down = ParametrizedModule(
            nn.Linear(d_ff, d_model, bias=False),
            width_dim=d_ff, layer_type="hidden")

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
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
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Pre-LN Transformer with all width-sensitive ops wrapped."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=512):
        super().__init__()
        self.tok_emb = ParametrizedModule(
            nn.Embedding(vocab_size, d_model),
            width_dim=d_model, layer_type="embedding")
        self.pos_emb = ParametrizedModule(
            nn.Embedding(max_seq_len, d_model),
            width_dim=d_model, layer_type="embedding")
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
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


# --- Training ---

def make_model(vocab_size=256, d_model=128, n_heads=4, d_ff=256, n_layers=4):
    return Transformer(vocab_size, d_model, n_heads, d_ff, n_layers)


def train(model, param_groups, steps=200, seq_len=64, batch_size=32):
    optimizer = torch.optim.AdamW(param_groups)
    vocab_size = model.tok_emb.inner.num_embeddings

    for step in range(steps):
        pattern_len = 4
        pattern = torch.randint(0, vocab_size, (batch_size, pattern_len))
        repeats = seq_len // pattern_len + 2
        seq = pattern.repeat(1, repeats)[:, :seq_len + 1]
        input_ids = seq[:, :-1]
        targets = seq[:, 1:]

        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  step {step:>4d}  loss {loss.item():.4f}")

    return loss.item()


if __name__ == "__main__":
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Apply ABC parametrization (discovers ParametrizedModules automatically)
    param = Parametrization(model, lr_prefactor=0.1)
    print(f"Parametrized {len(param.param_groups)} groups:")
    for g in param.param_groups:
        n_p = sum(p.numel() for p in g["params"])
        print(f"  {g['layer_name']:<45s}  lr={g['lr']:.6f}  params={n_p}")

    print("\nTraining on repeating-pattern task...")
    final_loss = train(model, param.param_groups)
    print(f"Final loss: {final_loss:.4f}")
