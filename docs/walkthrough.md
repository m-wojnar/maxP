# maxp — Code Walkthrough

A full walkthrough of the `maxp` package: what each module does, how
they connect, and how to use them.

## Table of Contents

1. [Overview](#1-overview)
2. [Theory in 60 Seconds](#2-theory-in-60-seconds)
3. [Module: `module.py` — ParametrizedModule](#3-module-modulepy--parametrizedmodule)
4. [Module: `parametrization.py` — Parametrization](#4-module-parametrizationpy--parametrization)
5. [Module: `solver.py` — LP Solver](#5-module-solverpy--lp-solver)
6. [Module: `alignment.py` — Alignment Measurement](#6-module-alignmentpy--alignment-measurement)
7. [Module: `dag.py` — DAG Tracing & Solving](#7-module-dagpy--dag-tracing--solving)
8. [Module: `trace.py` — Forward-Pass Tracing](#8-module-tracepy--forward-pass-tracing)
9. [Module: `diagnose.py` — Coordinate Checks](#9-module-diagnosepy--coordinate-checks)
10. [Usage Guide: Static Parametrization](#10-usage-guide-static-parametrization)
11. [Usage Guide: Dynamic Alignment](#11-usage-guide-dynamic-alignment)
12. [Usage Guide: DAG Solver](#12-usage-guide-dag-solver)
13. [Examples Overview](#13-examples-overview)
14. [Test Suite](#14-test-suite)
15. [Architecture Diagram](#15-architecture-diagram)

---

## 1. Overview

`maxp` implements the **abc-parametrization** framework from
[Everett et al., 2024](https://arxiv.org/abs/2407.05872) and extends it
with **dynamic alignment measurement** from
[iejmac's blog post](https://iejmac.github.io/2025/03/26/alignments.html).

The core idea: for each layer *l* in a neural network, three exponents
control its width-scaling behaviour:

| Exponent | Controls | Formula |
|----------|----------|---------|
| **a_l** | Output multiplier | `scale = n^{-a}` |
| **b_l** | Initialization variance | `std = std_prefactor * n^{-b}` |
| **c_l** | Learning rate | `lr = lr_prefactor * n^{-c}` |

where *n* is the width dimension (fan-in) of the layer. The `(a, b)` values
are set by the parametrization choice (e.g. muP), and `c` is solved via
linear programming to maximise learning rates while maintaining stability.

### Package structure

```
maxp/
├── __init__.py          # Re-exports public API
├── module.py            # ParametrizedModule — marks layers for ABC treatment
├── parametrization.py   # Parametrization — main entry point, orchestrates everything
├── solver.py            # LP solver (chain + DAG variants, Adam + SGD)
├── alignment.py         # Compute (align_z0_dW, align_dZ_w0, align_dZ_dW) alignment metrics
├── dag.py               # Build PM-to-PM data-flow graph
├── trace.py             # Trace matmul ops via __torch_function__
└── diagnose.py          # Coordinate-check diagnostics (width sweep plots)
```

---

## 2. Theory in 60 Seconds

A single layer computes `y = scale * (z @ w^T)` where `z` is input
activations and `w` is the weight matrix.

During training, both `z` and `w` change. Decompose the output change:

```
y = z @ w^T
  = (z_0 + dz) @ (w_0 + dw)^T
  = z_0 @ w_0^T          (init term — stable by design)
  + z_0 @ dw^T           (weight-change term, alignment = alpha)
  + dz  @ w_0^T          (activation-change term, alignment = omega)
  + dz  @ dw^T           (cross term, alignment = u)
```

Each term scales differently with width. The **alignment** values
(align_z0_dW, align_dZ_w0, align_dZ_dW) quantify how efficiently matrix products concentrate
vs spread across dimensions:

- **alpha = 1.0** ("full alignment"): `z_0 @ dw^T` concentrates like a
  rank-1 product
- **alpha = 0.5** ("no alignment"): `z_0 @ dw^T` behaves like a random
  product

The LP solver uses these alignment values as constraints to find the
largest possible `c` values (strongest LRs) that keep every layer's
output stable at O(1) scale.

**Static mode**: assume alignment values upfront (e.g. "full" = muP
assumption: align_z0_dW=1, align_dZ_w0=0.5, align_dZ_dW=1). Solve once at init.

**Dynamic mode**: measure alignment during training from actual (z, w)
snapshots. Re-solve the LP periodically to adapt LRs.

---

## 3. Module: `module.py` — ParametrizedModule

**File**: `maxp/module.py` (~60 lines)

This is the annotation layer. You wrap each width-sensitive operation
in a `ParametrizedModule` to tell the system: "this op needs ABC
treatment."

```python
class ParametrizedModule(nn.Module):
    def __init__(self, module_or_fn, width_dim: int, layer_type: str = "hidden"):
        ...
```

### Key attributes

| Attribute | Set by | Purpose |
|-----------|--------|---------|
| `inner` | User | The wrapped `nn.Module`, or `None` for callables |
| `width_dim` | User | Fan-in dimension that scales with width |
| `layer_type` | User | `"embedding"`, `"hidden"`, or `"readout"` |
| `scale` | `Parametrization` | Output multiplier = `width_dim^{-a}` |
| `align_z0_dW, align_dZ_w0, align_dZ_dW` | `Parametrization` | Alignment values (from preset or measurement) |
| `_z0, _w0` | `Parametrization` | Initial snapshots for dynamic alignment |

### Two modes of wrapping

**1. Wrap an `nn.Module`** (has learnable parameters):
```python
ParametrizedModule(
    nn.Linear(d_in, d_out, bias=False),
    width_dim=d_out,
    layer_type="hidden",
)
```

**2. Wrap a bare callable** (no parameters, only needs scaling):
```python
ParametrizedModule(
    lambda q, k: q @ k.transpose(-2, -1),
    width_dim=dim,
    layer_type="readout",
)
```

### The `weight` property

```python
@property
def weight(self) -> torch.nn.Parameter | None:
```

Returns the primary weight parameter of the inner module, or `None` for
callables. This is how `Parametrization` distinguishes "has parameters
to set LR for" from "only needs output scaling."

### Forward pass

```python
def forward(self, *args, **kwargs):
    out = self.inner(*args, **kwargs)  # or self._fn(...)
    return self.scale * out
```

The scale multiplication is the `n^{-a}` output multiplier.

---

## 4. Module: `parametrization.py` — Parametrization

**File**: `maxp/parametrization.py` (~440 lines)

This is the **main entry point**. It orchestrates everything: walks the
model, solves for `c`, initialises weights, builds optimizer param groups,
and handles dynamic re-solving.

### Constructor flow

```python
param = Parametrization(model, lr_prefactor=0.01, optimizer_type="adam", alignment="full")
```

The `__init__` does five things in order:

1. **Discover** all `ParametrizedModule` instances via `model.named_modules()`.
2. **Look up (a, b)** for each PM from `_DEFAULT_AB` (muP defaults) or
   user-provided `ab_overrides`.
3. **Solve for c** via LP — either chain solver (default) or DAG solver
   (when `sample_input` is provided).
4. **Apply parametrization**: re-init weights (`std = std_pf * n^{-b}`),
   set output scale (`pm.scale = n^{-a}`), set initial alignment on each
   PM.
5. **Build param groups**: one group per PM with `lr = lr_pf * n^{-c}`,
   plus an `"_other"` group for non-PM parameters (LayerNorm, etc.) at
   the base `lr_prefactor`.

### Default (a, b) values — muP

```python
_DEFAULT_AB = {
    "embedding": (-0.5, 0.5),  # a+b=0,   scale=sqrt(n), std=1/sqrt(n)
    "hidden":    (0.0, 0.5),   # a+b=0.5, scale=1,       std=1/sqrt(n)
    "readout":   (0.5, 0.5),   # a+b=1.0, scale=1/sqrt(n), std=1/sqrt(n)
}
```

These satisfy the stability-at-initialization constraints from the paper:
`a_0 + b_0 = 0`, `a_l + b_l = 0.5` (hidden), `a_L + b_L >= 0.5` (readout).

To use SP instead, pass `ab_overrides`:
```python
SP_AB = {"embedding": (0.0, 0.0), "hidden": (0.0, 0.5), "readout": (0.0, 0.5)}
param = Parametrization(model, ab_overrides=SP_AB, ...)
```

### Alignment presets

```python
_ALIGNMENT_PRESETS = {
    "full": (1.0, 0.5, 1.0),  # muP assumption
    "no":   (0.5, 0.5, 0.5),  # conservative / no alignment
}
```

### The `param_groups` property

```python
optimizer = torch.optim.Adam(param.param_groups)
```

Returns `list[dict]` in the format PyTorch optimizers expect. Each dict
has:

| Key | Value |
|-----|-------|
| `"params"` | List of `nn.Parameter` |
| `"lr"` | `lr_prefactor * n^{-c}` |
| `"layer_name"` | PM name (e.g. `"layers.0"`) or `"_other"` |
| `"fan_in"` | Width dimension |
| `"c"` | The solved c exponent |
| `"maxp_managed"` | `True` for PM groups, `False` for `"_other"` |

### Phase 2: Dynamic alignment

Three methods support runtime re-solving:

**`capture_initial(sample_input)`** — call once before training:
- Runs a forward pass with hooks to capture input activations at each PM
- Stores `pm._z0` (activations) and `pm._w0` (weights) on each PM

**`step(sample_input, optimizer=None)`** — call after each `optimizer.step()`:
1. Increment step counter; skip if in warmup or not on solve_interval
2. Capture current activations via forward hooks
3. Compute `(align_z0_dW, align_dZ_w0, align_dZ_dW)` per PM using `compute_alignment()`
4. Re-solve LP with measured alignment
5. Update `lr` in param_groups (and sync to optimizer if provided)

**`_resolve_chain()` / `_resolve_dag()`** — internal re-solve methods.

### Static solvers (init-time)

**`_solve_c_chain_static()`** — filters to weight-bearing PMs, passes
their `(a, b)` and alignment preset to the chain LP solver. Returns
`{name: c}` dict.

**`_solve_c_dag_initial()`** — traces a DAG from `sample_input`, sets
alignment on graph nodes, solves the DAG LP. Returns `({name: c}, graph)`.

---

## 5. Module: `solver.py` — LP Solver

**File**: `maxp/solver.py` (~600 lines)

Solves for optimal learning rate exponents `c_l` using PuLP linear
programming. Two solver families: **chain** (sequential models) and
**DAG** (arbitrary graphs).

### Chain solver

```python
find_c(al, bl, align_z0_dW, align_dZ_w0, align_dZ_dW, optimizer_type="adam") -> (cl, rl)
```

Dispatches to `find_c_adam()` or `find_c_sgd()`.

**Inputs**: Lists of `(a, b, align_z0_dW, align_dZ_w0, align_dZ_dW)` per layer, one entry per
weight-bearing PM in chain order.

**Output**: `(cl, rl)` — optimal `c` exponents and stability residuals `r`.

**Objective**: Minimize `sum(c_l)` (i.e. maximize learning rates).

**Constraints** (Adam, layer *i*):
```
r[0] = a[0] + c[0]                          (first layer)
r[0] >= 0

r[i] = min(
    a[i] + c[i] - alpha[i],                 (weight-change stability)
    a[i] + c[i] + r[i-1] - u[i],            (cross-term stability)
    0.5 + r[i-1] - omega[i],                (activation-change stability)
)
r[i] >= 0

r[L] = min(
    a[L] + b[L] + r[L-1] - omega[L],        (readout activation stability)
    a[L] + c[L] - alpha[L],                  (readout weight stability)
    a[L] + c[L] + r[L-1] - u[L],            (readout cross stability)
)
r[L] >= 0
```

The `min()` is encoded via the big-M method (`_min2_lp`, `_min_lp`).

SGD constraints differ (they include `2*c[i]` terms because SGD updates
scale as `lr * gradient` rather than Adam's normalised updates).

### Graph-based solver

```python
find_c(graph: OpGraph, optimizer_type="adam") -> dict[str, (c, r)]
```

Dispatches to `find_c_adam()` or `find_c_sgd()`.

Same objective and constraint logic, but operates on an `OpGraph` where:
- Nodes have predecessors/successors (not just linear chain)
- Merge types (`MIN` for addition/residuals, `SUM` for elementwise multiply)
  determine how residuals combine at merge points

---

## 6. Module: `alignment.py` — Alignment Measurement

**File**: `maxp/alignment.py` (~184 lines)

Pure functions for computing alignment metrics. No coupling to
`ParametrizedModule` — takes raw tensors in, returns floats out.

### `compute_alignment(z0, w0, z, w, fan_in, )`

Given initial and current (activations, weights), computes:

```
alpha = log(rms(z_0 @ dw^T)) - log(rms(z_0) * rms(dw))
        ─────────────────────────────────────────────────
                         log(fan_in)
```

(and analogously for omega and u).

In RMS mode, this measures how much the matrix product concentrates
relative to the product of norms, normalised by `log(width)`. A value of
1.0 means maximal concentration (rank-1-like); 0.5 means random.

Returns `(align_z0_dW, align_dZ_w0, align_dZ_dW)` as sanitised floats (no inf/nan).

### `compute_alignment(z0, w0, z, w, fan_in)`

Batch wrapper: takes a list of `((z0, w0), (z, w))` tuples (one per PM)
and returns `(alpha_list, omega_list, u_list)`.

---

## 7. Module: `dag.py` — DAG Tracing & Solving

**File**: `maxp/dag.py` (~300 lines)

Builds a directed acyclic graph of PM-to-PM data flow. This is needed
for architectures where the chain assumption doesn't hold (e.g.
transformers with residual connections, SwiGLU gates, attention).

### Key types

**`MergeType`** (enum):
- `MIN` — addition / residual connections (output is dominated by the
  larger input, so residuals combine via min)
- `SUM` — elementwise multiplication (exponents add)

**`DagNode`** (dataclass):
```python
@dataclass
class DagNode:
    name: str
    a: float
    b: float
    layer_type: str
    has_weight: bool
    width_dim: int
    predecessors: list[str]
    successors: list[str]
    merge_type: MergeType = MergeType.MIN
    alpha: float = 1.0
    omega: float = 0.5
    u: float = 1.0
```

**`OpGraph`**:
- `nodes: dict[str, DagNode]`
- `topological_order()` — returns nodes in topological order
- `sources()` / `sinks()` — entry/exit nodes
- `validate()` — checks graph consistency

### `trace_pm_dag(model, sample_input, ab=None) -> OpGraph`

The main function. Runs a forward pass with hooks on all PMs to discover
data-flow edges. Uses `_DagBuilder` internally which:
1. Registers pre/post hooks on every PM
2. Tracks which PM's output feeds into which PM's input via tensor
   identity tracking
3. Detects merge points (additions → MIN, multiplications → SUM)
4. Builds the `OpGraph`

---

## 8. Module: `trace.py` — Forward-Pass Tracing

**File**: `maxp/trace.py` (~420 lines)

Low-level tracing infrastructure. Intercepts every matmul-like operation
during a forward pass using `__torch_function__`. Used by the coordinate
check tools (`diagnose.py`).

### Key concepts

**`_TracingTensor`**: A `torch.Tensor` subclass that overrides
`__torch_function__` to record matmul operations. When a matmul is
detected, it records the op name, input/output shapes, the associated
parameter name, and the source file location.

**`trace_forward(model, sample_input)`**: Wraps `sample_input` as a
`_TracingTensor` and runs a forward pass, collecting all `TracedOp`
entries.

**`classify(make_model_fn, make_input_fn, widths)`**: Traces at two
different widths, then classifies each op as `"embedding"`, `"hidden"`,
or `"readout"` based on how its dimensions scale with width.

**`measure_activations(model, sample_input, traced_ops)`**: Re-runs the
forward pass and measures `abs(output).mean()` for each traced op.

---

## 9. Module: `diagnose.py` — Coordinate Checks

**File**: `maxp/diagnose.py` (~200 lines)

High-level diagnostic tools for verifying that parametrization is working
correctly. These produce "coordinate check" plots: activation magnitudes
vs width on a log-log scale.

### `diagnose_axis(make_model_fn, make_input_fn, widths, n_steps, n_seeds)`

The main function. For each width:
1. Build model + input
2. Classify ops
3. Run `n_steps` training steps over `n_seeds` random seeds
4. Measure activation magnitudes at each step

Returns `(ops, affected_indices, act_stats)` — the classified ops, which
ones are affected by the width axis, and an array of activation statistics.

### `print_axis()` / `plot_axis()`

Display results as a table or matplotlib plot. Correct parametrization
shows flat lines (activation magnitudes independent of width).

---

## 10. Usage Guide: Static Parametrization

The simplest usage: annotate your model, create a `Parametrization`, use
the param groups.

### Step 1: Annotate the model

Wrap every width-sensitive linear layer in `ParametrizedModule`:

```python
import torch.nn as nn
from maxp import ParametrizedModule

class MyMLP(nn.Module):
    def __init__(self, d_in, d, d_out):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Linear(d_in, d, bias=False),
            width_dim=d, layer_type="embedding",
        )
        self.hidden = ParametrizedModule(
            nn.Linear(d, d, bias=False),
            width_dim=d, layer_type="hidden",
        )
        self.head = ParametrizedModule(
            nn.Linear(d, d_out, bias=False),
            width_dim=d, layer_type="readout",
        )

    def forward(self, x):
        x = self.emb(x).relu()
        x = self.hidden(x).relu()
        return self.head(x)
```

**Layer type rules**:
- `"embedding"`: first layer(s), maps from fixed input to width dimension
- `"hidden"`: middle layers, width-to-width
- `"readout"`: final layer(s), maps from width to fixed output

### Step 2: Create Parametrization

```python
from maxp import Parametrization

model = MyMLP(d_in=784, d=256, d_out=10)
param = Parametrization(
    model,
    lr_prefactor=0.01,      # base learning rate
    optimizer_type="adam",   # "adam" or "sgd"
    alignment="full",        # "full" (muP) or "no" (conservative)
)
```

This modifies the model **in-place**: re-initialises weights, sets output
scales on each PM.

### Step 3: Use param_groups with your optimizer

```python
optimizer = torch.optim.Adam(param.param_groups)

# Training loop — standard PyTorch
for xb, yb in dataloader:
    loss = F.cross_entropy(model(xb), yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

That's it for static parametrization. The learning rates are set once at
init and don't change.

### Using SP instead of muP

Override `(a, b)` values:

```python
SP_AB = {
    "embedding": (0.0, 0.0),
    "hidden":    (0.0, 0.5),
    "readout":   (0.0, 0.5),
}
param = Parametrization(model, ab_overrides=SP_AB, ...)
```

---

## 11. Usage Guide: Dynamic Alignment

Dynamic mode measures actual alignment during training and re-solves the
LP to adapt per-layer learning rates.

### Step 1: Same model annotation as static mode

### Step 2: Create Parametrization with dynamic params

```python
param = Parametrization(
    model,
    lr_prefactor=0.01,
    optimizer_type="sgd",
    alignment="full",        # initial assumption
    warmup_steps=50,         # don't re-solve during warmup
    solve_interval=1,        # re-solve every N steps
    sample_size=64,          # samples for alignment measurement
)
optimizer = torch.optim.SGD(param.param_groups, lr=0.01)
```

### Step 3: Capture initial state

```python
sample_X = X[:64]  # small batch for alignment measurement
param.capture_initial(sample_X)
```

This stores initial activations (`_z0`) and weights (`_w0`) on each PM.

### Step 4: Call `step()` after each optimizer step

```python
for xb, yb in dataloader:
    loss = F.cross_entropy(model(xb), yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Re-measure alignment and update LRs
    param.step(sample_X, optimizer)
```

The `step()` method:
1. Skips if still in warmup or not on solve_interval
2. Captures current activations via forward hooks
3. Computes `(align_z0_dW, align_dZ_w0, align_dZ_dW)` per PM
4. Re-solves the LP
5. Updates optimizer learning rates

### Inspecting per-layer alignment

After `step()`, alignment values live on each PM:

```python
for name, pm in param._pms:
    if pm.weight is not None:
        print(f"{name}: alpha={pm.align_z0_dW:.3f}, omega={pm.align_dZ_w0:.3f}, u={pm.align_dZ_dW:.3f}")
```

---

## 12. Usage Guide: DAG Solver

For architectures with residual connections, attention, or other non-chain
data flow, provide a `sample_input` to enable DAG solving:

```python
sample_input = torch.randn(1, 3, 32, 32)  # example input
param = Parametrization(
    model,
    lr_prefactor=0.01,
    sample_input=sample_input,  # triggers DAG tracing
)
```

This traces the PM-to-PM data-flow graph and solves a DAG LP instead of
the chain LP. Each PM gets its own `c` value based on its position in
the graph, handling merge points (residual additions, gated multiplications)
correctly.

You can inspect the graph:

```python
from maxp import trace_pm_dag

graph = trace_pm_dag(model, sample_input)
for node in graph.topological_order():
    print(f"{node.name}: a={node.a}, b={node.b}, preds={node.predecessors}")
```

---

## 13. Examples Overview

All examples live under `examples/`. Each example directory contains
model definitions and training scripts.

### `mlp_example/` — MLP baselines

| File | Purpose |
|------|---------|
| `mlp.py` | Vanilla MLP (no parametrization, baseline) |
| `parametrized_mlp.py` | MLP with `ParametrizedModule` wrappers |
| `train.py` | LR transfer demo: SP vs muP across widths |
| `mup_vs_conservative.py` | Conservative vs muP static alignment comparison |

### `vit_example/` — ViT baselines

| File | Purpose |
|------|---------|
| `vit.py` | Vanilla ViT (no parametrization) |
| `parametrized_vit.py` | ViT with `ParametrizedModule` wrappers |
| `train.py` | LR transfer demo: SP vs muP across widths |

### `dynamic_mlp_example/` — Dynamic alignment sweep (MLP)

| File | Purpose |
|------|---------|
| `parametrized_mlp.py` | Same as `mlp_example/parametrized_mlp.py` |
| `sweep.py` | 3-way LR sweep: SP vs muP vs maxP dynamic (SGD) |

### `dynamic_vit_example/` — Dynamic alignment sweep (ViT)

| File | Purpose |
|------|---------|
| `sweep.py` | 3-way LR sweep: SP vs muP vs maxP dynamic (SGD) |

### `parameterize_example/` — Transformer with SwiGLU

| File | Purpose |
|------|---------|
| `transformer.py` | Vanilla pre-LN Transformer (baseline) |
| `parameterized_transformer.py` | Parametrized version |
| `run.py` | Coordinate check + diagnosis |

### `repro_nanogpt_mup/` — nanoGPT-muP verification

| File | Purpose |
|------|---------|
| `verify_scaling.py` | Verifies our Parametrization reproduces nanoGPT-muP scaling |

### Standalone scripts

| File | Purpose |
|------|---------|
| `dag_solver_demo.py` | DAG solver properties: correctness, optimality, per-op values |
| `visualize_dag.py` | ASCII visualization of PM-to-PM data-flow graphs |

---

## 14. Test Suite

Tests live in `tests/`. Run with:

```bash
python -m pytest tests/ -v --tb=short
```

### New tests (maxp)

| File | Tests | What it covers |
|------|-------|---------------|
| `test_dag.py` | 22 | DAG tracing, DAG solver, Parametrization+DAG integration |
| `test_dag_solver_properties.py` | 19 | Analytical correctness, optimality, per-op differentiation |
| `test_step.py` | 47 | `capture_initial()`, `step()`, warmup, interval, optimizer sync |
| `test_alignment.py` | 5 | `compute_alignment()` edge cases and properties |
| `test_optimizations_integration.py` | 10 | End-to-end optimization integration |

---

## 15. Architecture Diagram

```
User's Model
    │
    ├─ ParametrizedModule("embedding", width_dim=d)
    │   └─ nn.Linear(d_in, d)
    │
    ├─ ParametrizedModule("hidden", width_dim=d)
    │   └─ nn.Linear(d, d)
    │
    └─ ParametrizedModule("readout", width_dim=d)
        └─ nn.Linear(d, d_out)

                │
                ▼

        Parametrization(model, ...)
                │
    ┌───────────┼───────────────┐
    │           │               │
    ▼           ▼               ▼
 Look up     Solve LP       Apply to model
 (a, b)    for c values    ├─ Re-init weights (std * n^{-b})
 per type    via solver     ├─ Set scale (n^{-a})
                            └─ Build param_groups (lr * n^{-c})
                │
                ▼
    optimizer = Adam(param.param_groups)

    ┌─── Training Loop ────────────────────┐
    │  loss.backward()                      │
    │  optimizer.step()                     │
    │                                       │
    │  param.step(sample_X, optimizer)      │  ◄── Dynamic mode only
    │    ├─ Capture current (z, w)          │
    │    ├─ compute_alignment() per PM      │
    │    ├─ Re-solve LP                     │
    │    └─ Update optimizer LRs            │
    └───────────────────────────────────────┘
```

### Data flow for DAG-aware solving

```
model + sample_input
        │
        ▼
  trace_pm_dag()  ──►  OpGraph
        │                 │
        │          ┌──────┴──────┐
        │          │  DagNode    │
        │          │  ├─ a, b    │
        │          │  ├─ alpha   │
        │          │  ├─ omega   │
        │          │  ├─ u       │
        │          │  └─ merge   │
        │          └─────────────┘
        │                 │
        ▼                 ▼
  find_c()  ──►  {name: (c, r)}
```
