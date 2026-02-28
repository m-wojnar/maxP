# Research Context

## Base Paper

**"Scaling Exponents Across Parameterizations and Optimizers"** (Everett et al., 2024)
arXiv: 2407.05872

Establishes the **abc-parametrization framework**: each layer l in a network has three exponents
(a_l, b_l, c_l) that control how quantities scale with width n:

- **a_l**: layer output multiplier — output scaled by n^{-a_l}
- **b_l**: initialization variance — weights ~ N(0, n^{-2b_l})
- **c_l**: learning rate — lr_l = lr_prefactor * n^{-c_l}

The paper derives **stability constraints** — conditions on (a, b, c) that ensure activations
remain O(1) throughout training — and formulates finding optimal c_l as a Linear Program:

- **Objective**: minimize sum(c_l) (= maximize learning rates, since lr ~ n^{-c})
- **Constraints**: per-layer stability, chained through the network

The paper recovers known parametrizations (muP, SP, NTK, MFP) as special cases of this
framework, and shows that all of them (not just muP) can achieve hyperparameter transfer.

**Key limitation**: The stability constraints assume **full alignment** — worst-case correlations
between weights and activations. This is conservative: it assumes weight updates become maximally
correlated with activations during training, which rarely happens in practice.

Full alignment values: alpha=1.0, omega=0.5, U=1.0 for all layers.

---

## Blog Post / Our Extension

**"Exploring Tensor Alignments in Neural Networks"** (Kilian & Wojnar, 2025)
https://iejmac.github.io/2025/03/26/alignments.html
Code: https://github.com/iejMac/paramR/tree/main/research

### Key Contribution

The full alignment assumption is overly conservative. We can **measure** actual alignment during
training and solve with those values, getting larger learning rates while maintaining stability.

### What Alignments Are

When layer l computes y = z @ W^T, the output decomposes as:

```
y = z_0 @ w_0^T + z_0 @ dW^T + dz @ w_0^T + dz @ dW^T
     (init)       (weight Δ)    (act Δ)      (cross Δ)
```

The three alignment metrics measure correlation between the cross-terms:

- **alpha_l**: alignment of z_0 @ dW^T — how correlated initial activations are with weight changes
- **omega_l**: alignment of dz @ w_0^T — how correlated activation changes are with initial weights
- **U_l**: alignment of dz @ dW^T — how correlated activation changes are with weight changes

These are measured as exponents of width: a value of 0 means no alignment (CLT scaling),
full alignment values (alpha=1, omega=0.5, U=1) represent maximum correlation.

### LP Constraints (Adam)

Variables: c_l (LR exponents), r_l (stability residuals)

**First layer (embedding)**:
```
r_0 = a_0 + c_0
r_0 >= 0
```

**Hidden layers (l = 1 to L-2)**:
```
r_l = min(a_l + c_l - alpha_l,
          a_l + c_l + r_{l-1} - U_l,
          0.5 + r_{l-1} - omega_l)
r_l >= 0
```

**Output layer (l = L-1)**:
```
r_{L-1} = min(a_{L-1} + b_{L-1} + r_{L-2} - omega_{L-1},
              a_{L-1} + c_{L-1} - alpha_{L-1},
              a_{L-1} + c_{L-1} + r_{L-2} - U_{L-1})
r_{L-1} >= 0
```

### Key Results

- Measured alignments are significantly lower than full-alignment assumption
- Dynamic LP solving consistently outperforms or matches baselines
- Works across parametrizations (SP, muP, MFP, NTK) and optimizers (SGD, Adam)
- SP + Adam showed the most significant improvements

### Limitations Identified

1. **Alignment trajectory dependence**: measuring from a baseline run and applying to a new run
   can fail because different LRs change the alignment trajectory
2. **Sum objective causes tradeoffs**: solver can decrease one layer's LR to increase others
3. **MLP-only**: the constraint structure (embed/hidden/readout) is hardcoded for MLPs
4. **SGD less robust** than Adam for alignment-based scheduling

---

## paramR vs maxP Comparison

paramR is the research prototype; maxP is the library extraction.

Key differences relevant to the refactor:
- paramR: tightly coupled to MLP model class, closure-based scheduler
- maxP: model-agnostic via nn.Linear scanning, proper scheduler class
- Both use identical LP formulation
- maxP added: WSD support, ChainedMaxPScheduler, checkpointing, two alignment norm modes
- paramR has a metric registry system that maxP dropped
- paramR's alignment uses ratio-style; maxP defaults to log-scale (configurable)

---

## What Needs to Change

1. **Solver generalization**: Currently hardcoded to embed/hidden/readout structure.
   Need to support arbitrary layer topologies.
2. **API simplification**: Three-step setup (init weights, create param groups, create scheduler)
   is error-prone. Should be a single entry point.
3. **Architecture support**: Need to map LP semantics onto transformers, convnets, etc.
4. **Tests**: Need tests that verify against known parametrization results, not just API smoke tests.
