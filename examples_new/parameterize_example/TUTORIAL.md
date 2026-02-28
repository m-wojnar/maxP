# Parametrizing a Transformer — Tutorial

This walkthrough shows how to take an arbitrary PyTorch model and apply maxP's
ABC parametrization so that hyperparameters (especially learning rate) transfer
across model widths. The process has four steps:

1. Write your model (or start with an existing one)
2. Diagnose which ops are sensitive to width
3. Wrap those ops with `ParametrizedModule`
4. Apply `Parametrization` and verify with a coord check

We use a small pre-LN Transformer with SwiGLU as the running example.
All code lives in this directory.

## The Problem

Standard PyTorch initialization (`kaiming_uniform_`) doesn't keep activations
stable as you scale model width. A learning rate tuned at width 128 blows up or
vanishes at width 1024. This makes hyperparameter search expensive: you can't
tune on a small model and transfer to a large one.

ABC parametrization (Everett et al., 2024) fixes this by choosing per-layer:
- **a** — output multiplier exponent: `scale = width^(-a)`
- **b** — init variance exponent: `std = width^(-b)`
- **c** — learning rate exponent: `lr = lr_prefactor * width^(-c)`

maxP automates the choice of (a, b, c) via an LP solver, given the layer
structure of your model.

## Step 1: Write Your Model

See [`transformer.py`](transformer.py) — a standard pre-LN Transformer with
gated MLP (SwiGLU), multi-head attention, and a language modelling head.
Nothing special here; this is the model *before* any parametrization.

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers):
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks  = nn.ModuleList([...])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
```

## Step 2: Diagnose

Before wrapping anything, use `classify()` to find out which ops are sensitive
to which width dimension, and run a coord check to see the instabilities.

```bash
python examples_new/parameterize_example/run.py --plot
```

This:
1. Traces the model at two widths per axis (d_model, head_dim)
2. Classifies every matmul op as **embedding**, **hidden**, or **readout**
3. Sweeps widths, trains for a few steps, and measures `abs(activation).mean()`
4. Prints a table and saves a plot (`diagnose_vanilla_*.png`)

Look at the output table:

```
=== Axis: d_model ===
  idx  op         type       p?   label                                         d_model=64    d_model=128  ...
  ---- ---------- ---------- ---- --------------------------------------------- ------------- ------------
  0    embedding  embedding  Y    #0 tok_emb.weight                                  0.0002        0.0002  ...
  1    embedding  embedding  Y    #1 pos_emb.weight                                  0.0003        0.0003  ...
  2    linear     hidden     Y    #2 blocks.0.attn.qkv                               0.1432        0.1030  ...
  ...
```

- **embedding** ops: activation doesn't change with width — these are fine
- **hidden** ops: activation scales with width — these need wrapping
- **readout** ops: contraction scales but output is fixed — these need wrapping

The plots show lines drifting apart across widths at different training steps.
Flat lines = stable = good. Diverging lines = unstable = needs parametrization.

## Step 3: Wrap with ParametrizedModule

Compare [`transformer.py`](transformer.py) with
[`parameterized_transformer.py`](parameterized_transformer.py). Every
width-sensitive op is wrapped:

**Embeddings** — the embedding lookup itself:
```python
# Before:
self.tok_emb = nn.Embedding(vocab_size, d_model)

# After:
self.tok_emb = ParametrizedModule(
    nn.Embedding(vocab_size, d_model),
    width_dim=d_model, layer_type="embedding")
```

**Hidden linear layers** — MLP and attention projections:
```python
# Before:
self.gate = nn.Linear(d_model, d_ff, bias=False)

# After:
self.gate = ParametrizedModule(
    nn.Linear(d_model, d_ff, bias=False),
    width_dim=d_model, layer_type="hidden")
```

**Parameter-less ops** — like QK^T in attention:
```python
# Before (inline):
attn = q @ k.transpose(-2, -1)

# After (wrapped):
self.attn_score = ParametrizedModule(
    lambda q, k: q @ k.transpose(-2, -1),
    width_dim=self.head_dim, layer_type="readout")
# ...
attn = self.attn_score(q, k)
```

**Readout** — the final projection to vocab:
```python
# Before:
self.head = nn.Linear(d_model, vocab_size, bias=False)

# After:
self.head = ParametrizedModule(
    nn.Linear(d_model, vocab_size, bias=False),
    width_dim=d_model, layer_type="readout")
```

Key choices:
- **`width_dim`**: the fan-in dimension that scales with width (usually `d_model`
  for most layers, `d_ff` for the down-projection, `head_dim` for QK^T)
- **`layer_type`**: `"embedding"` for input layers, `"hidden"` for
  intermediate, `"readout"` for final outputs. This determines the (a, b)
  defaults and the LP constraint type.

**What NOT to wrap**: LayerNorm, activation functions, residual connections.
These don't have width-dependent matmuls, so they don't need parametrization.
Their parameters (if any) go into the `_other` param group at `lr_prefactor`.

## Step 4: Parametrize & Verify

Once the model is wrapped, applying parametrization is one line:

```python
from maxp_new import Parametrization

model = Transformer(vocab_size=256, d_model=128, n_heads=4, d_ff=256, n_layers=4)
sample = torch.randint(0, 256, (1, 8))
param = Parametrization(model, lr_prefactor=0.1, sample_input=sample)
optimizer = torch.optim.AdamW(param.param_groups)
```

`Parametrization` auto-discovers all `ParametrizedModule` instances and:
1. Sets output scales: `pm.scale = width_dim^(-a)`
2. Re-initialises weights: `std = width_dim^(-b)`
3. Solves for optimal `c` per op via the DAG LP solver
4. Builds `param_groups` with `lr = lr_prefactor * width_dim^(-c)`

### Chain solver vs DAG solver

When you pass `sample_input`, `Parametrization` traces a forward pass to discover
the actual PM-to-PM data flow — which ops feed into which, and whether they
combine via addition (residual) or multiplication (SwiGLU). It builds a DAG and
solves the LP with one `c` variable per weight-bearing op.

Without `sample_input`, it falls back to a chain solver that collapses all ops
of the same `layer_type` into a single `c` value. Both give the same results
under the default alignment presets, but the DAG solver is required for Phase 2
(per-op measured alignment).

### Visualizing the DAG

To verify the tracer parsed your model correctly:

```bash
python examples_new/visualize_dag.py --solve
python examples_new/visualize_dag.py --solve --n-layers 1  # simpler view
```

This shows the traced topology, merge types (`+` for add/residual, `*` for
multiply/SwiGLU), and the solved `c` values.

### Verify with coord check

```bash
python examples_new/parameterize_example/run.py --parametrized --plot
```

The plots (`diagnose_parametrized_*.png`) should show flat lines — activations
are O(1) across widths at every training step. This means the learning rate
found at width 64 will work at width 1024.

## Customization

### Override (a, b) per layer type

The defaults are muP: embedding(-0.5, 0.5), hidden(0.0, 0.5), readout(0.5, 0.5).
Override with `ab_overrides`:

```python
param = Parametrization(
    model,
    ab_overrides={"readout": (0.5, 0.0)},  # different b for readout
    lr_prefactor=0.1,
)
```

### Optimizer type

The LP solver needs to know the optimizer family (different gradient scaling):

```python
param = Parametrization(model, optimizer_type="sgd", lr_prefactor=0.01)
optimizer = torch.optim.SGD(param.param_groups)
```

### Alignment assumption

The `alignment` parameter controls how aggressively the solver maximises
learning rates:

- `"full"` (default): assumes full alignment — gives the largest stable LRs
- `"no"`: assumes no alignment — more conservative

```python
param = Parametrization(model, alignment="no", lr_prefactor=0.1)
```

## Reference

- [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872) — Everett et al., 2024
- [Dynamic alignment measurement](https://iejmac.github.io/2025/03/26/alignments.html)
- [`docs/parametrization_policy.md`](../../docs/parametrization_policy.md) — layer type classification rules
