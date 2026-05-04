# maxP

**maxP** is a PyTorch library for neural network parametrization implementing the abc-parametrization framework from [Everett et al., 2024](https://arxiv.org/abs/2407.05872) with dynamic alignment measurement from [this blog post](https://iejmac.github.io/2025/03/26/alignments.html).

Each layer `l` has three exponents controlling its width-scaling behavior:
- **`a_l`**: Output multiplier — layer output is scaled by `n^{-a_l}`
- **`b_l`**: Init variance — weights initialized as `N(0, n^{-2b_l})`
- **`c_l`**: Learning rate — `lr_l = lr_prefactor * n^{-c_l}`

The library solves a Linear Program (LP) to find optimal `c_l` values that maximize per-layer learning rates while maintaining numerical stability. Optionally, it measures actual alignment between initial and current weights/activations during training and re-solves the LP dynamically.

## Installation

```bash
git clone https://github.com/m-wojnar/maxP.git
cd maxP

# Core library only (CPU)
pip install -e .

# With dev dependencies (includes torchtitan for LLM experiments)
uv pip install -e ".[dev,cu128]"   # CUDA 12.8
uv pip install -e ".[dev,cu130]"   # CUDA 13.0
uv pip install -e ".[dev]"         # CPU / system torch
```

## Quick Start

```python
import torch
import torch.nn as nn
from maxp import ParametrizedModule, Parametrization

# 1. Wrap layers you want parametrized
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.emb  = ParametrizedModule(nn.Linear(784, width, bias=False),
                                       width_dim=width, layer_type="embedding")
        self.fc1  = ParametrizedModule(nn.Linear(width, width, bias=False),
                                       width_dim=width, layer_type="hidden")
        self.head = ParametrizedModule(nn.Linear(width, 10, bias=False),
                                       width_dim=width, layer_type="readout")

    def forward(self, x):
        return self.head(torch.relu(self.fc1(self.emb(x))))

model = MLP(width=256)

# 2. Apply parametrization (re-inits weights, solves LP, builds param groups)
param = Parametrization(model, lr_prefactor=1e-3, alignment="full")

# 3. Create optimizer from param_groups (each layer gets its own LR)
optimizer = torch.optim.AdamW(param.param_groups)

# 4. Capture initial state BEFORE training (needed for dynamic alignment)
X_init = next(iter(train_loader))[0]
param.capture_initial(X_init)

# 5. Training loop
for X, y in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    param.step(X, optimizer)  # measure alignment, re-solve LP, update LRs
```

## Design Overview

```
maxp/
├── module.py          # ParametrizedModule — marks layers for parametrization
├── parametrization.py # Parametrization — main entry point, Phase 1 + 2
├── alignment.py       # compute_alignment() — measures align_z0_dW, align_dZ_w0, align_dZ_dW
├── solver.py          # LP solver (Adam + SGD) — finds optimal c values
├── dag.py             # DAG builder — traces PM-to-PM data flow
├── trace.py           # Operation tracer — records matmul-like ops
└── diagnose.py        # Coord-check diagnostics — sweep widths, plot scaling
```

**Phase 1** (static, at init): discover PMs → reinit weights → solve LP → build param groups.

**Phase 2** (dynamic, per step): capture activations → measure alignment → re-solve LP → update per-layer LRs.


## ParametrizedModule

`ParametrizedModule` marks a layer for abc-parametrization. It wraps any `nn.Module` (or bare callable for parameter-free ops):

```python
# Standard linear layer
fc = ParametrizedModule(nn.Linear(n, n), width_dim=n, layer_type="hidden")

# Embedding layer
emb = ParametrizedModule(nn.Embedding(vocab, n), width_dim=n, layer_type="embedding")

# Attention QK^T — parameter-free, but needs its own scaling
qk = ParametrizedModule(lambda q, k: q @ k.T, width_dim=n // n_heads, layer_type="readout")

# Override any of (a, b, c) per layer
custom = ParametrizedModule(nn.Linear(n, n), width_dim=n, layer_type="hidden", a=0.0, b=0.5, c=1.5)
```

**Layer types and default (a, b) under muP:**

| `layer_type` | `a`    | `b`   | `a+b` | Role                    |
|--------------|--------|-------|-------|-------------------------|
| `"embedding"`| `-0.5` | `0.5` | `0.0` | Source (input embedding)|
| `"hidden"`   | `0.0`  | `0.5` | `0.5` | Interior hidden layers  |
| `"readout"`  | `0.5`  | `0.5` | `1.0` | Final output (logits)   |

## Parametrization

`Parametrization` is the main entry point. At construction it:
1. Discovers all `ParametrizedModule` instances in the model
2. Re-initializes their weights: `std = std_prefactor * width_dim^{-b}`
3. Sets output scales: `pm.scale = width_dim^{-a}`
4. Solves the LP to find per-layer `c` values
5. Builds `param_groups` with per-layer learning rates

```python
param = Parametrization(
    model,
    optimizer_type="adam",       # "adam" or "sgd" — needed for LP formulation
    alignment="full",            # "full" (worst-case) or "no" (no alignment)
    lr_prefactor=1e-3,           # base learning rate multiplier
    std_prefactor=1.0,           # weight init multiplier
    ab_overrides=None,           # dict: layer_type -> (a, b) to override defaults
    c_overrides=None,            # dict: layer_type -> c to pin c per layer type
    alignment_overrides=None,    # dict: name/suffix/type -> (align_z0_dW, align_dZ_w0, align_dZ_dW)
    sample_input=None,           # provide to trace actual data-flow DAG
    warmup_steps=0,              # steps before first dynamic LP re-solve
    solve_interval=1,            # re-solve every N steps
    sample_size=32,              # max batch size for alignment measurement
    c_ema=0.0,                   # EMA smoothing for c values (0 = instant)
    alignment_ema=0.0,           # EMA smoothing for measured alignment values
    resample_w0=False,           # re-sample w0 snapshot each solve
    use_training_activations=False,  # use activations from training forward pass
    solver=None,                 # custom PuLP solver (default: CBC)
    warm_start=False,            # warm-start LP from previous c solution
)

# Param groups for optimizer
optimizer = torch.optim.AdamW(param.param_groups)
```

### Alignment presets

The `alignment` argument controls the initial assumption about how correlated weights and activations are:

- `"full"` (default): worst-case assumption — assumes maximum alignment. Leads to smaller initial LRs that are safe regardless of actual alignment.
- `"no"`: assumes no alignment — leads to larger initial LRs, appropriate when you expect random-like behavior.

### DAG-based LP solving

By default, maxP builds a synthetic linear-chain graph from the order PMs are discovered. If you pass `sample_input`, it traces the actual data-flow graph:

```python
X_sample = torch.randn(1, 784)
param = Parametrization(model, sample_input=X_sample, lr_prefactor=1e-3)
```

This enables per-op `c` values for non-linear topologies (residuals, attention, SwiGLU, etc.) where different paths through the network can have different optimal learning rates.

## Dynamic Alignment (Phase 2)

After a few training steps, the actual alignment between initial weights/activations and their updates can differ significantly from the preset. Dynamic alignment measures this and re-solves the LP to track the true optimal LRs.

### Alignment metrics

For each layer computing `y = z @ W^T`, maxP decomposes the output into four terms:

```
y = z0 @ W0^T + z0 @ dW^T + dZ @ W0^T + dZ @ dW^T
```

And measures three alignment metrics (how efficiently each term scales with width):

| Metric         | Measures                         | Preset `"full"` | Preset `"no"`  |
|----------------|----------------------------------|-----------------|----------------|
| `align_z0_dW`  | alignment of `z0 @ dW^T` term    | `1.0`           | `0.5`          |
| `align_dZ_w0`  | alignment of `dZ @ W0^T` term    | `0.5`           | `0.5`          |
| `align_dZ_dW`  | alignment of `dZ @ dW^T` term    | `1.0`           | `0.5`          |

These are log-scale RMS alignment values. A value of `1.0` means fully aligned (outputs scale as if vectors were parallel); `0.5` means random (outputs scale as sqrt(fan_in)); values close to `0.0` mean anti-aligned.

### Training loop with dynamic alignment

```python
param = Parametrization(model, lr_prefactor=1e-3, warmup_steps=100, solve_interval=10)
optimizer = torch.optim.AdamW(param.param_groups)

# Required: capture initial (z0, W0) snapshots before training
param.capture_initial(X_init)

for step, (X, y) in enumerate(train_loader):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

    # After optimizer.step(): measure alignment, re-solve, update LRs
    param.step(X, optimizer)
```

`param.step()` does nothing during the first `warmup_steps` steps and only re-solves every `solve_interval` steps to reduce overhead.

### Pinning alignment per layer

Override alignment for specific layers (they skip dynamic measurement):

```python
param = Parametrization(
    model,
    alignment_overrides={
        "head":    (1.0, 0.5, 1.0),   # pin by exact name
        "fc":      (0.8, 0.5, 0.8),   # pin by name suffix (matches "blocks.0.ff.fc")
        "hidden":  (0.5, 0.5, 0.5),   # pin by layer_type
    },
)
```

### Adjusting lr_prefactor externally

`param.step()` always recomputes LRs from `param.lr_prefactor * width^{-c}`, so you can apply a global LR schedule by updating `lr_prefactor`:

```python
for step, (X, y) in enumerate(train_loader):
    # ... training step ...
    param.lr_prefactor = base_lr * lr_schedule(step)
    param.step(X, optimizer)
```

## Annotating a Model

### Choosing `layer_type`

The type determines the default `(a, b)` exponents and the role the layer plays in the LP graph. The key distinction is which dimensions scale with model width `n`:

- **`"embedding"`**: fan-in is **fixed** (does not scale with `n`), fan-out scales with `n`. Any projection from a fixed-size input space into the hidden dimension qualifies — token embeddings, positional embeddings, patch projections, etc. There can be multiple.
- **`"hidden"`**: both fan-in and fan-out scale with `n`. All weight matrices that live entirely within the hidden space: Q/K/V projections, attention output projection, FFN up/gate/down layers.
- **`"readout"`**: fan-in scales with `n`, fan-out is **fixed**. This includes the final unembedding layer, but also attention's QK^T product (which is a readout wrt sequence length — `d_head` scales with `n`, the sequence dimension does not). There can be multiple.

When in doubt: look at which dimensions of the weight matrix scale with model width. Fixed-in → `"embedding"`, fixed-out → `"readout"`, both scale → `"hidden"`.

### Choosing `width_dim`

`width_dim` is the fan-in of the operation — the dimension that is being contracted (summed over) in the matrix multiply, and the one that scales with model width. This is what sets `n` in the scaling exponents `n^{-a}`, `n^{-b}`, `n^{-c}`.

### Verifying with coordinate checks

After annotating your model, run a coordinate check to confirm that activation magnitudes are stable across widths before committing to a training run:

```python
from maxp import diagnose_axis, print_axis, plot_axis

widths = [64, 128, 256, 512]

def make_model(width):
    model = MyModel(width)
    param = Parametrization(model, lr_prefactor=1e-3)
    return model, param.param_groups

def make_input(width):
    return torch.randint(0, vocab_size, (4, seq_len))

all_ops, affected, act_stats = diagnose_axis(make_model, make_input, widths)
print_axis("width", all_ops, affected, act_stats, widths)
plot_axis("width", all_ops, affected, act_stats, widths, filename="coord_check.png")
```

`print_axis` shows how the RMS of each op's output scales with width. For a correctly parametrized model, activations should be roughly **constant** (slope ≈ 0 in log-log) at init and remain stable after a few training steps.

If an activation grows with width, `a` is too small for that layer — increase it. If it shrinks, `a` is too large. Adjust per-layer with the `a` override on `ParametrizedModule` and re-run until all slopes are near zero.

## LLM Experiments

The `experiments/lm/` directory contains a full LLaMA-3 pre-training pipeline built on [torchtitan](https://github.com/pytorch/torchtitan). It trains five scales (30M–3B parameters) with three methods:

| Method | Parametrization | Alignment | Schedule |
|---|---|---|---|
| `maxP` | µP + measured | online (dynamic re-solve) | emergent |
| `mup-full` | µP | full (worst-case preset) | constant |
| `mup-no` | µP | none (permissive preset) | constant |

Training steps are computed automatically as `20 × non-embed params / tokens-per-step`.

### Files

```
experiments/lm/
├── train.py               # Main training entry point (MaxPTrainer)
├── maxp_llama3.py         # LLaMA-3 scale configs (debug, s1–s5) + compute_steps
├── maxp_converter.py      # Post-optimizer-build hook; wires up Parametrization
├── launch_sweep.py        # Generate and submit SLURM jobs for full LR sweep
├── fineweb.py             # FineWeb-Edu dataset registration for torchtitan
├── download_hf_assets.py  # HuggingFace asset downloader (from torchtitan)
├── run.sh                 # Submit a sweep for one scale
├── run_debug.sh           # Quick single-GPU debug run
└── coord_check.py         # Coord check for the parametrized LLaMA-3 model
```

### Debug run (single GPU)

```bash
bash experiments/lm/run_debug.sh \
    --tokenizer /path/to/tokenizer \
    --c4-test /path/to/c4_test \
    --steps 200 --method maxP
```

### SLURM sweep

```bash
bash experiments/lm/run.sh s3   # submits 7 LRs × 3 methods × 2 seeds = 42 jobs
```

## Vision Experiments

The `experiments/vision/` directory contains ViT and MLP pre-training experiments using [timm](https://github.com/huggingface/pytorch-image-models) with streaming HuggingFace datasets. It trains four ViT scales and four MLP scales with the same three methods as the LLM experiments.

### Files

```
experiments/vision/
├── train.py           # Main training entry point (single-GPU, streaming HF data)
├── maxp_timm.py       # timm model registry + automatic ParametrizedModule wrappers
├── hf_vision_data.py  # HF streaming train/val pipeline + transforms
├── utils.py           # Shared helpers (LR logging, arg parsing, checkpointing)
├── launch_sweep.py    # Generate and submit SLURM jobs for full LR sweep
├── run.sh             # Submit a sweep for one scale
├── run_debug.sh       # Quick tiny-model debug run (CPU/GPU)
├── coord_check_vit.py # Coord-style diagnostic for ViT models
└── coord_check_mlp.py # Coord-style diagnostic for MLP models
```

Available scales: `debug`, `vit-s`, `vit-b`, `vit-l`, `mlp-s`, `mlp-m`, `mlp-b`, `mlp-l`.

### Debug run (CPU/GPU)

```bash
bash experiments/vision/run_debug.sh --steps 20
```

### SLURM sweep

```bash
bash experiments/vision/run.sh vit-s   # submits 9 LRs × 3 methods × 3 seeds = 81 jobs
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v --tb=short
```

Tests run on CPU only (no GPU required). The suite covers LP solver correctness (analytical values, optimality, perturbation tests), DAG tracing and merge detection, alignment computation, and end-to-end parametrization + dynamic step behavior.
