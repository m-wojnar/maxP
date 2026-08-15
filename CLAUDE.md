# CLAUDE.md — maxP

maxP is a PyTorch library for **abc-parametrization** (Everett et al. 2024, arXiv:2407.05872), extended with **measured dynamic alignment** (https://iejmac.github.io/2025/03/26/alignments.html). Each layer `l` has exponents `(a_l, b_l, c_l)`: output `×n^{-a}`, init variance `n^{-2b}`, learning rate `lr_prefactor·n^{-c}`. Three modes: **maxP** (dynamic, the default — measure alignment during training, re-solve the LP, update per-layer LRs), **maxP-meas** (measure once via `measure_only`, freeze the table through `alignment_overrides`, train statically), and static µP presets (`alignment="full"`/`"no"`). LP solving covers Adam / SGD / Adafactor.

## Layout

```
maxp/                  # the library
  module.py            # ParametrizedModule — wraps a layer (output scale + per-layer init/LR)
  parametrization.py   # Parametrization — entry point: wrap → solve c → optimizer param_groups
  solver.py            # LP solver for the c-exponents (Adam / SGD / Adafactor)
  alignment.py  dag.py  trace.py  diagnose.py   # measure / classify-DAG / coord-checks
tests/                 # pytest, CPU-only
examples/              # small standalone demos (MLP / ViT / nanoGPT)
experiments/
  lm/                  # torchtitan LLaMA-3 pipeline (SLURM, server)
  vision/              # timm ViT pipeline           (SLURM, server)
docs/
  walkthrough.md              # per-module code map (start here)
  parametrization_policy.md   # layer-classification / abc theory
  research_context.md         # paper + blog theory summary
  verify.md / verify.py       # component-by-component correctness evidence
```

## Dev

```bash
source .venv/bin/activate
pip install -e .[dev]
python -m pytest tests/ -v --tb=short      # CPU-only, no GPU; CI on py3.10–3.14
```

Experiments run on GPU via SLURM (server), never locally.
