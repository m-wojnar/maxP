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

## Is the preserved LR optimum (mup-no ↔ maxP-meas) a guarantee or luck?

Both arms share the **same** prefactor optimum at the tuning scales — `mup-no s1=s2=3e-2`, `maxP-meas s1=s2=3e-2` (pipeline `report.txt`, `[consistent]`), and it `[TRANSFERS]` to s3. So switching mup-no → maxP-meas re-scales per-layer LRs but does **not** move the global-LR argmin. Is that guaranteed? **Neither a free theorem nor luck — a conditional structural result, and the condition is empirically met here.**

**Reduction.** At fixed tuning width `n`, the c-table only changes the width-*exponent*, so it multiplies each layer's LR by a **constant** `k_l = n^(−Δc_l)`. The question is purely: rescale per-layer LRs `{η·m_l} → {η·m_l·k_l}`, does the optimal global `η*` stay put?

**In general — no guarantee.** Rescaling *relative* per-layer LRs deforms the loss-vs-η landscape; `η*` can move. No theorem makes it invariant under arbitrary per-layer rescaling.

**Why it held here — two semi-principled facts:**

1. **The correction is readout-dominated** — `Δc_readout ≈ +0.228`, every other layer within `±0.04` → `k_l ≈ 1` except the readout. Predicted, not accidental: muP's `α=0.5` alignment assumption is approximately right for the **bulk hidden layers** (muP's basis) and wrong mainly at the **boundary** (readout), so measured corrections concentrate there.
2. **Readout LR is weakly coupled to `η*`** — the usable-LR window (hence the argmin) is set by the hidden stack's feature-learning dynamics; the readout is one top linear layer, and its c is solver-**pinned** by the output-stability constraint. Rescaling it lowers the loss **floor** but barely moves the **location** of the optimum.

**Deeper near-principle.** Both arms are maximal-update-class parametrizations. The optimal global LR is an intrinsic property of the optimization problem (arch + optimizer + data) that muP factors the width-scaling out of; two *correct* MUP-family parametrizations target the same joint-"all-layers-maximal" `η`, differing only in **how** correct they are → they differ in the **loss floor**, not the **argmin**. The residual gap between them *is* mup-no's alignment error — small and concentrated → small `η*` shift.

**Evidence the optimum is decisive (not a coarse-grid near-tie).** Per-LR loss at s2 (`.out`, seed s1):

| LR | mup-no | maxP-meas |
|----|--------|-----------|
| 1e-2 | 3.988 | 4.040 |
| **3e-2** | **3.786** | **3.675** |
| 1e-1 | 3.945 | 3.740 |

3e-2 wins by ~0.1–0.2 nat over both neighbours in **both** arms — well above seed noise.

**Where it breaks (paper-honest boundary):**

- If a regime makes muP's alignment assumption badly wrong across **many bulk layers** (large Δc spread through depth, not just readout), `η*` **will** move → re-tuning is mandatory.
- "Preserved" is only **within one grid step** (×3 spacing); a sub-3× drift is real-but-invisible.
- Assumes both stay in the stable feature-learning class.

**Implication.** The small-scale maxP-meas re-tune (Stage C) is **not ceremony** — it is the *test* that the readout-concentrated + decoupled precondition holds. It is **not** a priori safe to skip for a new arch/dataset (e.g. **vision** — its c-table may not be readout-dominated; verify before trusting a single tune). Correct claim for the paper: *"the LR optimum is preserved because the measured correction is readout-concentrated and the readout LR is decoupled from the global optimum"* — **not** *"maxP-meas preserves the LR optimum"* unconditionally.

## Caveats

- **OOD validation**: val = c4, train = fineweb-edu. Fair across arms (identical), but val is not in-distribution. Train margin runs ~0.03–0.05 larger than val at small scale (maxP-meas fits train harder; part doesn't transfer to c4) — expected.
- **Coarse LR grid at large scale**: s3/s4 swept ~3 LRs, s5 only 1 → "stable optimum" claims rest on a coarse grid. muP s4 optimum unbracketed (see note above).
- **Thin seeds at scale**: s5 single seed → no error bar on the headline 2.6B number.
- One s4 cell (mupno lr3e-2 s2) lost its `.out` train history to a resume-job guard-skip truncation; val/checkpoint intact, seed-s1 twin used for train.

## Operational note

18 leftover SLURM jobs (4 × s4, 14 × s5) remain PD — chain links whose target dirs already have `COMPLETED`; they hit the launch_sweep guard and exit on start. Safe to `scancel` to free scheduling slots.
