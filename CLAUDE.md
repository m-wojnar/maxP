# CLAUDE.md — maxP project instructions

## What This Project Is

maxP is a PyTorch library for neural network parametrization. It implements the
abc-parametrization framework from "Scaling Exponents Across Parameterizations and
Optimizers" (Everett et al., 2024, arXiv:2407.05872) and extends it with dynamic
alignment measurement from https://iejmac.github.io/2025/03/26/alignments.html.

Core idea: each layer l has exponents (a_l, b_l, c_l) controlling output multiplier
(n^{-a}), init variance (n^{-2b}), and learning rate (lr_prefactor * n^{-c}).

## Status

Active rewrite in progress. Working in two phases:
- **Phase 1** (current): Static parametrizations with robust tests
- **Phase 2**: Dynamic alignment measurement and LP solving

See `docs/DESIGN.md` for full design. All working docs live in `docs/`.

## Environment

```bash
source .venv/bin/activate        # Python 3.13 via Homebrew
pip install -e .[dev]
python -m pytest tests/          # run tests (CPU only, no GPU)
```

## Key Dependencies

- torch >= 2.8, numpy >= 2.0, pulp >= 3.0
- Dev: pytest, pytest-cov, matplotlib, pandas, pyyaml, torchvision

## Project Layout

```
maxp/              # Main package
tests/             # pytest suite
examples/          # Training scripts (MLP, ViT on CIFAR-10)
docs/              # Design docs and notes
  DESIGN.md        # Target design
  research_context.md  # Paper/blog theory summary
  current_state_of_main.md  # Pre-refactor snapshot
```

## Conventions

- Tests must run on CPU (no GPU assumed during development)
- CI runs on Python 3.10-3.14
- pytest with `-v --tb=short`
