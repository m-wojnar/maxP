# Width-Scaling Parameterization Policy

**Source**: Everett et al., "Scaling Exponents Across Parameterizations and Optimizers" (ICML 2024, arXiv:2407.05872)

## Core Idea

When scaling model width `n`, every operation that **contracts (sums) over a width-scaled dimension** introduces a sqrt(n) factor in the activation scale that must be compensated for by the initialization, parameter multiplier, and learning rate. The parameterization is a prescription for how to set these three quantities per-layer to maintain stable activations and at-most-stable logits as width grows.

## Classification Rule

Given two model configs that differ only in width `n`, trace the forward pass and classify every **parameterized operation** (i.e., operation involving learnable weights) by asking:

> **Does computing each output element require summing over a dimension that scales with `n`?**

### Three layer types:

| Type | Contraction over width-scaled dim? | Examples in Transformer | Stability constraint |
|------|-----------------------------------|------------------------|---------------------|
| **Embedding** | No | Token embedding (sums over vocab, fixed), positional embedding, LayerNorm scale (elementwise) | `a + b = 0` |
| **Hidden** | Yes, on both input and output side (both fan-in and fan-out scale with `n`) | QKV projections, attention output projection, MLP up/down projections | `a + b = 1/2` |
| **Readout** | Yes on input side (fan-in scales), but output dim is fixed and these are the final logit-producing weights | LM head / unembedding | `a + b >= 1/2` |

### Operations that are NOT parameterized (no learnable weights involved in width-scaling):
- Elementwise nonlinearities (GeLU, SiLU, etc.) -- bounded derivative, negligible in infinite-width limit
- Residual additions -- elementwise
- Softmax -- operates over sequence dim, not width
- Attention QK^T -- contracts over head dim `h`, which is typically fixed (not width-scaled)
- Any contraction over a fixed (non-width-scaled) dimension

These operations require no parameterization adjustment. Only matmuls (or equivalent contractions) over width-scaled dimensions need to be classified.

## Parameterization Triples {a, b, c}

For each classified layer, three quantities are prescribed as powers of `n`:

| Symbol | Meaning | Definition |
|--------|---------|------------|
| `a` | Parameter multiplier exponent | Forward pass computes `n^{-a} * W * input` |
| `b` | Initialization std exponent | `W ~ N(0, n^{-2b})` |
| `c` | Learning rate exponent | `eta proportional to n^{-c}` |

## Four Standard Parameterizations

These define `(a, b)` per layer type. The learning rate exponent `c` depends on the optimizer and alignment assumption.

|  | Embedding (a, b) | Hidden (a, b) | Readout (a, b) |
|--|-------------------|---------------|----------------|
| **Standard (STP)** | (0, 0) | (0, 1/2) | (0, 1/2) |
| **NTK** | (0, 0) | (1/2, 0) | (1/2, 0) |
| **muP** | (-1/2, 1/2) | (0, 1/2) | (1/2, 0) |
| **Mean-Field (MFP)** | (0, 0) | (1/2, 0) | (1, 0) |

Note: STP and NTK are equivalent, and muP and MFP are equivalent, under infinite precision (they form equivalence classes). The key distinction is the readout layer: STP/NTK have `a+b = 1/2` (constant-scale logits at init), muP/MFP have `a+b = 1` (logits scale as `1/sqrt(n)` at init).

## Recommended Learning Rate Exponents

### For Adam (recommended: Standard Param + Full Alignment):

| Layer Type | LR scaling |
|-----------|------------|
| Embedding | `O(1)` -- no decay with width |
| Hidden | `O(1/n)` -- decays linearly with width |
| Readout | `O(1/n)` -- decays linearly with width |

This is the paper's top recommendation for Adam. It outperforms muP at all scales tested (up to 26.8B params).

### For Adam + Parameter Scaling / Adafactor (recommended: No Alignment):

| Layer Type | LR scaling |
|-----------|------------|
| Embedding | `O(1)` |
| Hidden | `O(1)` |
| Readout | `O(1)` |

Global LR works because parameter scaling automatically compensates for width.

### Full reference: See Table 1 in the paper for all combos of {STP, NTK, muP, MFP} x {SGD, Adam, Adafactor} x {Full Align, No Align}.

## Per-Layer Constant Multipliers

Beyond the exponent, each layer type gets a **constant multiplier** gamma:

```
eta_l = beta_n * gamma_l * (n / b)^{-c_l}
```

where `b` is a base model width (e.g., 1024), `beta_n` is the base LR (swept per model size), and `gamma_l` is tuned at small scale and transferred. These constants are **essential** for good performance -- they can be tuned cheaply at the base width and reused across all scales.

## Epsilon in Adam

The epsilon hyperparameter breaks scale invariance. At large width, gradients shrink and can fall below epsilon, causing underflow. Three mitigations:
1. Use a smaller constant epsilon (e.g., 1e-15 instead of 1e-8)
2. Scale epsilon per-layer: `eps_l = base_eps * (n/b)^{-g_l}` where `g_l` is the gradient scale exponent
3. **Adam-atan2** (recommended): Replace `m / (sqrt(v) + eps)` with `atan2(m, sqrt(v))` -- eliminates epsilon entirely, one-line change, scale-invariant

## Compiler Design Notes

To auto-parameterize a model:

1. **Trace the forward pass** between two configs with different widths.
2. **Identify all learnable parameters** and the operations they participate in.
3. **For each parameterized matmul**, determine:
   - What is the contraction dimension? Does it scale with width?
   - If no: **skip** (no parameterization needed, or classify as embedding-type)
   - If yes: Does the output dimension also scale with width?
     - Yes -> **hidden**
     - No (fixed output, produces logits) -> **readout**
4. **For elementwise learnable params** (LayerNorm scale/bias): classify as **embedding**.
5. **Apply the chosen parameterization** (a, b, c) triple to each classified layer.
6. **Assign per-layer LR groups** based on classification.

### Key invariant to verify:
At the base width `b`, the parameterization should be a no-op (all scaling factors equal 1). The parameterization only modifies behavior when scaling away from the base width.

### Edge cases to handle:
- **Tied weights** (e.g., embedding = readout^T): These serve dual roles. The paper uses untied weights. If tying, you need to decide which parameterization wins or handle gradient contributions separately.
- **Attention head dim**: If head dim `h` is fixed and only num_heads `H` scales, the QK^T contraction over `h` is O(1). But if head dim scales with width, you'd need to parameterize that too.
- **MoE routing**: Not covered by this paper but the same principle applies -- classify by whether the contraction dimension scales.
- **Shared dimensions**: If two dimensions co-scale (e.g., D and F=4D), both are "width-scaled" for classification purposes.