"""Standard pre-LN Transformer with gated MLP (SwiGLU).

Used as a test bed for tracing, classification, and coordinate checks.

Usage:
    python examples_new/parameterize_example/transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP: out = down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv.unbind(0)

        attn = q @ k.transpose(-2, -1)
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
    """Pre-LN Transformer with gated MLP for sequence-to-sequence tasks.

    Architecture: token embed -> N x (LN -> Attn -> LN -> GatedMLP) -> LN -> head
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# --- Training ---

def make_model(vocab_size=256, d_model=128, n_heads=4, d_ff=256, n_layers=4):
    model = Transformer(vocab_size, d_model, n_heads, d_ff, n_layers)
    return model


def train(model, steps=200, seq_len=64, batch_size=32, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    vocab_size = model.tok_emb.num_embeddings

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
    print("Training on repeating-pattern task...")
    final_loss = train(model)
    print(f"Final loss: {final_loss:.4f}")
