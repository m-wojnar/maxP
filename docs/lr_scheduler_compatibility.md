# LR Scheduler Compatibility Analysis

## Problem

`Parametrization.step()` overwrites LRs with absolute values:

```python
group["lr"] = self.lr_prefactor * (fan_in ** (-c))
```

Any external PyTorch LR scheduler adjustment gets blown away on the next re-solve.

## Conceptual Decomposition

MaxP controls **per-layer LR ratios** (`n^{-c}`). A scheduler controls the **global LR envelope**. These compose as a product:

```
effective_lr(t) = schedule_multiplier(t) * lr_prefactor * fan_in^(-c_l)
```

The two concerns are orthogonal — maxP picks the ratios, the scheduler picks the overall scale over time.

## Old Implementation (`maxp/scheduler.py`)

The old package solved this two ways:

### 1. Built-in WSD (Warmup-Stable-Decay)

`MaxPScheduler` has `_get_wsd_multiplier()` returning a scalar envelope:

```python
lr = lr_prefactor * (fan_in ** (-c)) * wsd_mult
```

- Warmup: linear ramp from `wsd_min_factor` to 1.0
- Stable: multiplier = 1.0 (LP solver active)
- Decay: LP solver stops, LRs frozen, only decay multiplier applied

### 2. ChainedMaxPScheduler (for external PyTorch schedulers)

Wraps `MaxPScheduler` + arbitrary `torch.optim.lr_scheduler` instances:

```python
# Each step:
lr_before = optimizer.param_groups[managed_idx]["lr"]
for sched in external_schedulers:
    sched.step()
lr_after = optimizer.param_groups[managed_idx]["lr"]

# Apply relative change to prefactor
maxp_scheduler.lr_prefactor *= (lr_after / lr_before)

# Then maxP computes per-layer LRs with updated prefactor
maxp_scheduler.step(X)
```

This lets CosineAnnealingLR, LinearLR, etc. control the envelope while maxP handles per-layer ratios.

## Plan for `maxp_new`

Simplest approach: add a `schedule_multiplier` to `Parametrization.step()` or adopt the `ChainedMaxPScheduler` pattern. The key change in `step()`:

```python
# Current:
group["lr"] = self.lr_prefactor * (fan_in ** (-c))

# With scheduler support:
group["lr"] = self.lr_prefactor * (fan_in ** (-c)) * multiplier
```

Where `multiplier` comes from either:
- A built-in WSD schedule (like the old code)
- An external scheduler via the chaining pattern
- A user-supplied callback

The old `maxp/scheduler.py` is the reference implementation — it already handles WSD, chaining, state_dict/load_state_dict, and decay-phase LR freezing.
