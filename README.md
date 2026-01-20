# maxP

**maxP** is a PyTorch learning rate scheduler that dynamically adjusts per-layer learning rates during training. It works by measuring the alignment between initial and current weights/activations, then solving a Linear Program (LP) to find optimal learning rate exponents that maximize training speed while maintaining numerical stability.

The scheduler assigns learning rate constraints based on **semantic roles**:

- **EMBEDDING**: Embedding layers, positional embeddings, scale parameters of the LayerNorm, or the first Linear layer if no embeddings exist
- **HIDDEN**: MLP layers, attention Q/K/V projections, attention output
- **READOUT**: The final output layer

For a detailed explanation of the theoretical foundations, see [this blog post](https://iejmac.github.io/2025/03/26/alignments.html).

## Installation

```bash
git clone https://github.com/m-wojnar/maxP.git
cd maxP
pip install .
```

## Quick Start

```python
import torch
import torch.nn as nn
from maxp import MaxPScheduler, create_param_groups, initialize_abc_weights

# 1. Create your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# 2. Initialize weights with ABC parametrization
initialize_abc_weights(model, parametrization="mup")

# 3. Create parameter groups with per-layer LRs
lr_prefactor = 0.001
param_groups = create_param_groups(
    model,
    lr_prefactor=lr_prefactor,
    parametrization="mup",
)

# 4. Create optimizer and scheduler
optimizer = torch.optim.AdamW(param_groups)
scheduler = MaxPScheduler(
    optimizer,
    model,
    parametrization="mup",
    lr_prefactor=lr_prefactor,
    solver_warmup_steps=100,
)

# 5. Capture initial state BEFORE training
X_init = next(iter(train_loader))[0]
scheduler.capture_initial(X_init)

# 6. Training loop
for X, y in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step(X)  # Pass input for alignment computation
```

## Advanced Usage

### Custom ABC Values

Define custom exponents per layer instead of using a named parametrization.

**Important**: Custom values must satisfy stability constraints based on the layer's semantic role:

- **EMBEDDING layer** (first Linear if no embeddings): `al + bl = 0.0`
- **HIDDEN layers**: `al + bl = 0.5` (stability at initialization)
- **READOUT layer** (last Linear): `al + bl >= 0.5`

For a 4-layer MLP (no embedding layers), the roles are: EMBEDDING, HIDDEN, HIDDEN, READOUT.

```python
from maxp import initialize_abc_weights, create_param_groups, MaxPScheduler

# Define custom exponents per layer (4-layer MLP: 1 EMBEDDING + 2 HIDDEN + 1 READOUT)
al = [-0.5, 0.0, 0.0, 0.5]  # Layer output multipliers
bl = [0.5, 0.5, 0.5, 0.5]   # Initialization variance exponents
cl = [0.5, 1.0, 1.0, 0.5]   # Initial learning rate exponents

# Verify constraints:
# EMBEDDING layer (0): al + bl = -0.5 + 0.5 = 0.0 ✓
# HIDDEN layers (1-2): al + bl = 0.0 + 0.5 = 0.5 ✓
# READOUT layer (3):   al + bl = 0.5 + 0.5 = 1.0 >= 0.5 ✓

initialize_abc_weights(model, al=al, bl=bl)
param_groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
optimizer = torch.optim.AdamW(param_groups)

scheduler = MaxPScheduler(
    optimizer, model,
    al=al, bl=bl,
    lr_prefactor=0.1,
)
```

### Memory-Efficient Mode

Avoid storing initial weights by resampling them when needed:

```python
scheduler = MaxPScheduler(
    optimizer, model,
    parametrization="mup",
    lr_prefactor=0.001,
    resample_w0=True,  # Don't store initial weights
)
```

### Feature Learning Constraint

Enforce the feature learning constraint (`r = 0` for the last hidden layer before readout) in the LP:

```python
optimizer = torch.optim.SGD(param_groups, lr=0.1)
scheduler = MaxPScheduler(
    optimizer, model,
    parametrization="ntk",
    lr_prefactor=0.1,
    feature_learning=True,  # Enforces r=0 for last HIDDEN layer
)
```

### Solve Interval

Reduce LP solving frequency for faster training:

```python
scheduler = MaxPScheduler(
    optimizer, model,
    parametrization="mup",
    lr_prefactor=0.001,
    solver_warmup_steps=100,
    solve_interval=10,  # Solve LP every 10 steps
)
```

### WSD (Warmup-Stable-Decay) Schedule

Built-in support for WSD learning rate schedules with independent LR warmup and LP solver warmup. During the decay phase, the LP solver stops and per-layer LRs are frozen at their last computed values.

```python
scheduler = MaxPScheduler(
    optimizer, model,
    parametrization="mup",
    lr_prefactor=0.001,
    solver_warmup_steps=100,       # LP solver starts after 100 steps
    wsd_warmup_steps=500,          # Linear LR warmup: 500 steps
    wsd_stable_steps=9000,         # Stable LR phase: 9000 steps
    wsd_decay_steps=500,           # Decay phase: 500 steps
    wsd_decay_type="cosine",       # "cosine" or "linear"
    wsd_min_factor=0.0,            # Decay to 0% of base LR
)
```

The WSD schedule consists of three phases:

1. **Warmup** (`wsd_warmup_steps`): Linear ramp from `wsd_min_factor × lr` to `lr`
2. **Stable** (`wsd_stable_steps`): Constant learning rate, LP solver actively adjusts per-layer LRs
3. **Decay** (`wsd_decay_steps`): LP solver stops, frozen per-layer LRs decay from `lr` to `wsd_min_factor × lr`

WSD is disabled by default (`wsd_decay_type="none"`). When enabled, both `wsd_stable_steps` and `wsd_decay_steps` are required.

### Chaining with Other LR Schedulers

Combine MaxPScheduler with standard PyTorch learning rate schedulers (e.g., cosine annealing, linear warmup) using `ChainedMaxPScheduler`:

```python
from maxp import MaxPScheduler, ChainedMaxPScheduler, create_param_groups, initialize_abc_weights

# Create MaxP scheduler
maxp_scheduler = MaxPScheduler(
    optimizer, model,
    parametrization="mup",
    lr_prefactor=0.1,
    solver_warmup_steps=100,
)

# Create standard PyTorch schedulers
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Chain them together
scheduler = ChainedMaxPScheduler(maxp_scheduler, [cosine_scheduler])

# Use like MaxPScheduler
scheduler.capture_initial(X_init)
for X, y in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    scheduler.step(X)
```

The chained scheduler works by:

1. Letting the external schedulers control the global base learning rate
2. MaxPScheduler handles per-layer LR ratios based on alignment measurements
3. On each step, the relative LR change from external schedulers is applied to MaxP's `lr_prefactor`

You can chain multiple schedulers together:

```python
# Warmup + cosine decay
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=100)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
scheduler = ChainedMaxPScheduler(maxp_scheduler, [warmup, cosine])
```

### Checkpointing

Save and restore scheduler state:

```python
# Save
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

# Load
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
```

### Accessing Scheduler State

```python
# Get current learning rates per layer
lrs = scheduler.get_last_lr()

# Get alignment values (alpha, omega, u)
alpha, omega, u = scheduler.get_alignment()

# Get layer names
layer_names = scheduler.get_layer_names()
```

## Running Training Examples

The `examples/` directory contains complete training scripts for MLP and ViT models on CIFAR-10.

### MLP Training

```bash
cd examples/mlp_training

# Train with maxP scheduler
python train.py --config config.yaml

# Train baseline (without maxP)
python train.py --config config_baseline.yaml
```

### ViT Training

```bash
cd examples/vit_training

# Train with maxP scheduler
python train.py --config config.yaml

# Train baseline
python train.py --config config_baseline.yaml
```

### Configuration

The config files control all training parameters. Key sections in `config.yaml`:

```yaml
model:
  input_dim: 3072
  hidden_dim: 512
  n_layers: 4
  output_dim: 10

optimizer:
  type: "adam"
  lr_prefactor: 0.001

maxp:
  use_maxp: true
  parametrization: "sp"
  alignment: "full"
  warmup_steps: 100
  solve_interval: 1
  alignment_norm: "rms"

logging:
  output_dir: "outputs/mlp_maxp"
```

Set `maxp.use_maxp: false` to disable the scheduler (as in `config_baseline.yaml`).

## Running Hyperparameter Sweeps

The sweep system allows you to run grid searches over hyperparameters with parallel workers.

### Sweep Configuration

Define your parameter grid in `sweep_config.yaml`:

```yaml
base_config: "config.yaml"
output_base: "outputs/mlp_sweep"
experiment_name: "mlp_lr_width_sweep"

# Parameter grids - all combinations will be explored
sweep:
  optimizer.lr_prefactor:
    - 0.3
    - 0.1
    - 0.03
    - 0.01
    - 0.003
    - 0.001

  model.hidden_dim:
    - 64
    - 128
    - 256

  model.n_layers:
    - 3
    - 4
    - 5

# Fixed overrides applied to all runs
overrides:
  n_steps: 2000
```

### Running Sweeps

```bash
cd examples/mlp_training

# Preview all configurations (dry run)
python sweep.py --config sweep_config.yaml --dry-run

# Run with a single worker
python sweep.py --config sweep_config.yaml

# Run with multiple parallel workers (4 GPUs)
./run_sweep.sh 4 sweep_config.yaml
```

The parallel runner (`run_sweep.sh`) automatically:
- Distributes jobs across available GPUs
- Sets thread limits to avoid oversubscription
- Handles graceful termination with Ctrl+C

## Visualizing Results

The `examples/visualize.py` script provides flexible visualization of training results.

### Basic Usage

```bash
cd examples

# Compare maxP vs baseline experiments
python visualize.py \
    --exp maxp=mlp_training/outputs/mlp_sweep \
    --exp baseline=mlp_training/outputs/mlp_baseline \
    --metric train_loss \
    --output plots/comparison.png
```

### Grouping and Selection

When you have multiple runs (e.g., from a sweep), the script groups them by configurable features and selects the best run from each group:

```bash
python visualize.py \
    --exp maxp=mlp_training/outputs/mlp_sweep \
    --group-by model.hidden_dim model.n_layers \
    --select-by loss \
    --metric train_loss \
    --smoothing 0.1 \
    --output plots/grid.png
```

### Available Options

| Option | Description |
|--------|-------------|
| `--exp name=path` | Add experiment (can be repeated) |
| `--group-by` | Config keys to group runs by (default: `model.hidden_dim model.n_layers`) |
| `--select-by` | How to select best run: `loss` or `accuracy` |
| `--metric` | Metric to plot: `train_loss`, `train_acc`, `test_loss`, `test_acc`, `lr_0`, `alpha_0`, `omega_0`, `u_0` |
| `--smoothing` | EMA smoothing alpha (lower = smoother, default: 0.05) |
| `--log-scale` | Use log scale for y-axis |
| `--single` | Create single plot instead of grid |
| `--per-layer` | Plot per-layer metrics: `lr`, `alpha`, `omega`, or `u` |
| `--output` | Output file path |
| `--title` | Custom plot title |
