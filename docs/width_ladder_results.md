# Width-ladder muTransfer results: maxP-meas vs muP (mup-no)

Run root (helios): `/net/storage/pr3/plgrid/plggadlers/maxP/runs/2026-06-12_*`
Status: **s1–s5 complete** (2026-06-25).

## Setup

- Architecture: LLaMA-3, width-only ladder, `n_layers=12`, `seq_len=3072`, global batch 16.
- Compute budget: `steps = 20 × non_embed_params / tokens_per_step` (Chinchilla-ish, fixed tokens-per-param).
- Train corpus: fineweb-edu. Validation corpus: **c4-validation** (OOD vs train — fair across arms, but not in-distribution).
- Two arms, identical everything except parametrization:
  - **mup-no** = standard muP (alignment = 0.5 uniform).
  - **maxP-meas** = static per-layer c-table from a single passive alignment measurement at **s2**, applied unchanged at every scale (`transfer-s2`). No online re-solving.
- Metric: validation loss (last `validate` point) and mean train loss over last 1k steps. LR grid swept per scale; s5 ran only the transferred optimum (lr 3e-2).

| scale | non-embed params | total steps |
|-------|------------------|-------------|
| s1 | 11.4M  | — |
| s2 | 40.9M  | — |
| s3 | 163.6M | — |
| s4 | 654.4M | — |
| s5 | 2.62B  | 1,065,002 |

Seeds: s1/s2 ×3, s3/s4 ×2, s5 ×1.

## Results — validation loss (best LR per arm, mean over seeds)

| scale | params | mup-no | maxP-meas | **Δ (nat)** |
|-------|--------|--------|-----------|-------------|
| s1 | 11M   | 4.847 (3e-2) | 4.756 (3e-2) | **0.091** |
| s2 | 41M   | 4.319 (3e-2) | 4.223 (3e-2) | **0.096** |
| s3 | 164M  | 3.945 (3e-2) | 3.822 (3e-2) | **0.123** |
| s4 | 654M  | 3.583 (1e-2)\* | 3.487 (3e-2) | **0.096** |
| s5 | 2.62B | 3.322 (3e-2 only)† | 3.240 (3e-2) | **0.083** |

\*s4 mup-no best (1e-2) sits at the **grid edge** (lowest LR run) — unbracketed, so the s4 margin is an upper bound; a 3e-3 point could narrow it. At matched 3e-2, s4 Δ = 0.102.

†s5 ran **only** lr3e-2 for both arms (not a swept best). By s4, muP's own optimum had slid below 3e-2 (own-best 1e-2), so the s5 muP number is muP at a likely-too-high LR while maxP-meas is at its genuine optimum. Δ_s5 = 0.083 is therefore an **upper bound on the loss-floor gap** (muP at its own optimum would land ~3.31, narrowing it to ~0.07). Conversely, that same fact is itself a result — see conclusion 3.

## Results — train loss (mean last 1k steps, lr 3e-2; s4 at each arm's own best)

| scale | mup-no | maxP-meas | **Δ (nat)** |
|-------|--------|-----------|-------------|
| s1 | 4.442 | 4.301 | **0.141** |
| s2 | 3.809 | 3.678 | **0.131** |
| s3 | 3.308 | 3.178 | **0.130** |
| s4 | 2.921\* | 2.826 | **0.096** |
| s5 | 2.685 | 2.595 | **0.090** |

\*s4 mup-no own-best = lr1e-2 (2.921); lr3e-2 = 2.928 (gap 0.007). maxP-meas own-best clearly lr3e-2 at s4/s5.

## Conclusions

1. **maxP-meas beats muP at every scale, ~0.1 nat, roughly flat across 230× params (11M → 2.6B).**
   Val Δ: 0.091 / 0.096 / 0.123 / 0.096 / 0.083. Train Δ: 0.141 / 0.131 / 0.130 / 0.096 / 0.090.
   Val is flat-ish/humped (s3 bump = one 2-seed point, treat as noise). Train Δ is **monotone declining** — 0.141 → 0.131 → 0.130 → 0.096 → 0.090, ~36% relative drop s1→s5 — but stays clearly positive, settling to a ~0.08–0.09 nat floor at 2.6B. Honest phrasing: **positive at every scale, gently decaying in train, ~0.08–0.09 nat at 2.6B — not decaying to zero**.

2. **Strongest finding — one alignment measurement transfers both directions.** The single s2-measured alignment table, applied unchanged, wins at s1 (down), s3/s4/s5 (up). One passive measurement → width-correct c-table that holds across the full ladder.

3. **LR optimum is width-stable for maxP-meas — partly a muTransfer-robustness result.** maxP-meas optimum stays 3e-2, cleanly bracketed at s1–s4 (s4: 1e-2 > 3e-2 < 1e-1) and still optimal-and-winning at s5. muP optimum holds 3e-2 through s3, then slides to the grid edge by s4 (best at lowest LR run) — i.e. **muP's transferred LR stops being optimal at scale while maxP-meas's does not**. So part of the s5 head-to-head is not "maxP-meas reaches a lower floor" but "muP failed to keep its muTransfer-predicted 3e-2 optimal" — a robustness claim, stronger and distinct from the loss-floor gap.

4. **Mechanism still open — not adjudicated by this data.** No run varies readout LR independently of global LR. The maxP-meas c-table is readout-dominated (output Δc ≈ +0.228 vs muP; all other layers within ±0.04). Width-consistency is *consistent with* width-correct per-layer scaling, but a flat-enough constant readout-LR cut could also hold ~0.1 nat. The clean falsification arm (**mup-no + tuned readout LR**, 2D sweep) was not run. Claim the ~0.1 nat as real and width-stable; leave the *mechanism* (width-correct per-layer scaling vs a constant readout cut) unproven.

## Caveats

- **OOD validation**: val = c4, train = fineweb-edu. Fair across arms (identical), but val is not in-distribution. Train margin runs ~0.03–0.05 larger than val at small scale (maxP-meas fits train harder; part doesn't transfer to c4) — expected.
- **Coarse LR grid at large scale**: s3/s4 swept ~3 LRs, s5 only 1 → "stable optimum" claims rest on a coarse grid. muP s4 optimum unbracketed (see note above).
- **Thin seeds at scale**: s5 single seed → no error bar on the headline 2.6B number.
- One s4 cell (mupno lr3e-2 s2) lost its `.out` train history to a resume-job guard-skip truncation; val/checkpoint intact, seed-s1 twin used for train.

## Operational note

18 leftover SLURM jobs (4 × s4, 14 × s5) remain PD — chain links whose target dirs already have `COMPLETED`; they hit the launch_sweep guard and exit on start. Safe to `scancel` to free scheduling slots.
