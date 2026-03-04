# DESIGN.md — maxP v2

**Status**: DRAFT — for discussion

---

## Goals

1. **Generalize beyond MLPs**: Support arbitrary architectures by letting the user specify
   which layers to parametrize and what role each has.
2. **Clean API**: Parametrization setup mutates the model and returns param groups for the optimizer.
3. **Composable with PyTorch ecosystem**: Work alongside standard LR schedulers, not replace them.
4. **Robust tests**: Verify correctness against known parametrization results, not just API shape.
5. **Phased implementation**: Start with static parametrizations (no dynamic alignment/solver),
   build robust tests, then add dynamic `.step()`.

---

## Target API

### Phase 1: Static Parametrizations

```python
import maxp

model = MyModel(...)

param = maxp.Parametrization(model, config)
# This:
#   1. Finds specified layers in the model
#   2. Re-initializes their weights according to b_l
#   3. Wraps them to apply layer multipliers according to a_l
#   4. Computes per-layer learning rates according to c_l
#   5. Exposes .param_groups for the optimizer

optimizer = AdamW(param.param_groups)
lr_sched = CosineAnnealingLR(optimizer, T_max=1000)

for batch in dataloader:
    loss = loss_fn(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
    lr_sched.step()
    optimizer.zero_grad()
```

### Phase 2: Dynamic Alignment (future)

```python
param = maxp.Parametrization(model, config)
optimizer = AdamW(param.param_groups)
lr_sched = CosineAnnealingLR(optimizer, T_max=1000)

for batch in dataloader:
    loss = loss_fn(model(batch.x), batch.y)
    loss.backward()
    lr_sched.step()
    param.step()      # measure alignment, solve LP, adjust per-layer LRs
    optimizer.step()
    optimizer.zero_grad()
```

---

## Layer Specification

The user specifies which layers to parametrize and what role each has. No magic
heuristics, no hardcoded skip logic. Any module type can be parametrized — Linear,
Conv2d, LayerNorm, whatever has parameters and produces output that should be scaled.

### Layer Roles

Each parametrized layer gets a **role** that determines its constraint type in the LP:

| Role | Meaning | LP Constraint |
|------|---------|---------------|
| `embedding` | Input doesn't change during training | `r_l = a_l + c_l` |
| `hidden` | Both activations and weights change | `r_l = min(a_l+c_l-alpha, a_l+c_l+r_{l-1}-U, 0.5+r_{l-1}-omega)` |
| `readout` | Final output / classification layer | Stable-logits variant |

### Config specifies layers via patterns

```python
config = maxp.Config(
    parametrization="mup",
    lr_prefactor=1e-3,
    layers={
        # glob pattern on module name → role
        "embed": "embedding",
        "blocks.*.ffn.up": "hidden",
        "blocks.*.ffn.down": "hidden",
        "blocks.*.attn.qkv": "hidden",
        "blocks.*.attn.proj": "hidden",
        "head": "readout",
    },
)
```

Layers not listed are **not parametrized** — they go into the optimizer with default
lr_prefactor, no ABC treatment. This gives the user full control over exactly which
modules get ABC scaling and which don't.

### MLP convenience

For simple MLPs where all Linear layers should be parametrized in order
(first=embedding, middle=hidden, last=readout):

```python
config = maxp.Config(
    parametrization="mup",
    lr_prefactor=1e-3,
    # layers=None → auto-detect: find all nn.Linear in definition order,
    # assign first=embedding, last=readout, rest=hidden
)
```

Auto-detection is opt-in (the default when `layers` is omitted), not a hidden heuristic.

---

## What Parametrization.__init__ Does

1. **Resolves layers**: Walks model modules, matches against `layers` patterns (or
   auto-detects). Produces an ordered list of (name, module, role) tuples.
   Any module type is valid — not limited to nn.Linear.

2. **Resolves ABC values**: From named parametrization or custom al/bl/cl.
   Each matched layer gets its own (a_l, b_l, c_l). For named parametrizations,
   values are derived from the role. For custom, the user provides one triplet per
   matched layer.

3. **Initializes weights**: For each parametrized layer, re-init with
   `std = std_prefactor * n^{-b_l}` where n = fan_in.

4. **Applies multipliers**: Wraps each parametrized layer so its output is scaled by
   `n^{-a_l}`. (How: replace the module in the model with a wrapper that scales output.)
   Works for any module type since it just wraps the forward pass.

5. **Builds param_groups**: One group per parametrized layer with
   `lr = lr_prefactor * n^{-c_l}`, plus one group for all other parameters with
   `lr = lr_prefactor`.

---

## Config

```python
@dataclass
class Config:
    # --- Parametrization ---
    # Named (mutually exclusive with custom al/bl/cl)
    parametrization: str | None = None   # "mup", "sp", "ntk", "mfp"
    optimizer_type: str | None = None    # "adam" or "sgd" — needed for ABC table lookup
    alignment: str | None = None         # "full" or "no" — alignment assumption for ABC tables

    # Custom (mutually exclusive with named parametrization)
    al: list[float] | None = None
    bl: list[float] | None = None
    cl: list[float] | None = None

    # --- Learning rate ---
    lr_prefactor: float = 1e-3

    # --- Layer specification ---
    # Dict of glob pattern → role ("embedding", "hidden", "readout")
    # None = auto-detect (all Linear layers, first=embedding, last=readout, rest=hidden)
    layers: dict[str, str] | None = None

    # --- Weight init ---
    std_prefactor: float = 1.0
    apply_multipliers: bool = True
```

Note: solver/alignment/tracer config omitted — that's Phase 2.

---

## Module Structure

```
maxp/
    __init__.py          # Public API: Parametrization, Config
    parametrization.py   # Parametrization class — main entry point
    tables.py            # Named parametrization ABC tables (mup, sp, ntk, mfp)
    solver.py            # LP solver (Phase 2)
    alignment.py         # Alignment computation (Phase 2)
    tracer.py            # Hook-based layer state capture (Phase 2)
```

Phase 1 only needs: `__init__.py`, `parametrization.py`, `tables.py`.

---

## Named Parametrization Tables

The ABC values for named parametrizations depend on:
- Which parametrization (mup, sp, ntk, mfp)
- Which optimizer (adam, sgd)
- Which alignment assumption (full, no)
- Layer role (embedding, hidden, readout)
- Number of layers (for some parametrizations)

These are pure data — lookup tables derived from the paper. Stored in `tables.py`.

---

## Tests (Phase 1)

### 1. ABC Tables
- Verify values match the paper (Table 1 from Everett et al.) for all combinations
- Verify consistency: n_layers=2 edge case, n_layers=100
- Verify stability-at-initialization conditions hold: a_0+b_0=0, a_l+b_l=0.5 (hidden), etc.

### 2. Weight Initialization
- Variance matches n^{-2b_l} for each layer
- Works with different model architectures (not just Sequential)
- Biases reset to zero
- Only specified layers are re-initialized

### 3. Layer Multipliers
- Output is scaled by n^{-a_l}
- Forward pass produces correct values
- Gradient flows through correctly

### 4. Param Groups
- One group per parametrized layer with correct lr = lr_prefactor * n^{-c_l}
- Non-parametrized params get default lr
- Groups contain correct parameters

### 5. Layer Matching
- Glob patterns match expected modules
- Auto-detection assigns correct roles
- Unmatched layers excluded from parametrization
- Error on empty match (no layers found)

### 6. Integration / Correctness
- For named parametrizations under full alignment, the static c_l values should equal
  the known muP/SP/NTK/MFP learning rate exponents
- Width transfer: train at width 64, find good lr_prefactor, verify it works at width 512
- Compare against standalone muP implementation (if available)

### 7. Solver (Phase 2)
- With full alignment inputs, recover named parametrization c_l values
- Constraint satisfaction: r_l >= 0
- Relaxed alignment → larger LRs (smaller c_l)
- Edge cases, infeasibility

---

## Open Questions

1. **Multiplier wrapper generality**: Current code has ScaledLinear (Linear-specific).
   Since we now support any module type, we need a generic ScaledModule wrapper that
   scales the output of any module. This is straightforward — just wrap forward() and
   multiply output by scale. Anything tricky here?

2. **fan_in for non-Linear layers**: For nn.Linear, fan_in = in_features and the scaling
   formulas are well-defined. For Conv2d, fan_in = in_channels * kernel_h * kernel_w.
   For LayerNorm, what is fan_in? We may need the user to specify width `n` explicitly
   when parametrizing non-standard layer types.

3. **Constraint chain ordering (Phase 2)**: When layers are specified via patterns, the
   LP chains r_l values sequentially. We'll use module definition order. The user controls
   which layers are included and their roles, so the ordering follows naturally from the
   model definition. Not relevant for Phase 1 (static c_l, no LP).
