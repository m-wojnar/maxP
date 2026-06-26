# CLAUDE.md — maxP

maxP is a PyTorch library for **abc-parametrization** (Everett et al. 2024, arXiv:2407.05872), extended with **measured dynamic alignment** (https://iejmac.github.io/2025/03/26/alignments.html). Each layer `l` has exponents `(a_l, b_l, c_l)`: output `×n^{-a}`, init variance `n^{-2b}`, learning rate `lr_prefactor·n^{-c}`. Static parametrization and dynamic alignment/LP solving are both implemented; current focus is width-ladder muTransfer experiments (LM + vision).

## Layout

```
maxp/                  # the library
  module.py            # ParametrizedModule — wraps a layer (output scale + per-layer init/LR)
  parametrization.py   # Parametrization — entry point: wrap → solve c → optimizer param_groups
  solver.py            # LP solver for the c-exponents
  alignment.py  dag.py  trace.py  diagnose.py   # measure / classify-DAG / coord-checks
tests/                 # pytest, CPU-only
examples/              # small standalone demos (MLP / ViT / nanoGPT)
experiments/
  lm/                  # torchtitan LLaMA-3 width-ladder  (SLURM, server)
  vision/              # timm ViT width-ladder            (SLURM, server)
docs/
  walkthrough.md              # per-module code map (start here)
  parametrization_policy.md   # layer-classification / abc theory
  research_context.md         # paper + blog theory summary
  verify.md / verify.py       # component-by-component correctness evidence
  width_ladder_results.md     # LM results
  vision_experiment_plan.md   # vision plan
```

## Dev

```bash
source .venv/bin/activate
pip install -e .[dev]
python -m pytest tests/ -v --tb=short      # CPU-only, no GPU; CI on py3.10–3.14
```

Experiments run on GPU via SLURM (server), never locally.
