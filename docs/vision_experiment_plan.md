# Vision muTransfer experiment plan (ViT) — maxP-meas vs muP

Goal: confirm the maxP-meas result on a **second modality**. Replicate the Llama3
LM pipeline on a Vision Transformer width ladder: a single passive alignment
measurement → static per-layer c-table → does it (a) beat standard muP and (b)
keep the LR optimum width-stable, transferring the table both down and up the
ladder.

This plan is the acceptance gate. **No training runs launch until accepted.**

## Scope (confirmed)

- **ViT only.** ViT is the transformer analog of the LM experiment. MLP deferred.
- **Two arms only:** `mup-no` (standard muP baseline) and `maxP-meas` (static
  c-table from a measured alignment table). Dynamic `maxP` and `mup-full` dropped.
- **Dataset: imagenet-12k** (`timm/imagenet-12k-wds`, 11821 classes). The maxP-meas
  win is readout-dominated, so a large head (= num_classes) is required or the
  effect is muted — 11821 classes satisfies this.
- **Fixed epoch budget** across all widths (same #epochs = same #samples at every
  scale; textbook muTransfer setup): a few real epochs of imagenet-12k. Tests
  width-stability of the optimal LR. Batch size held constant across all widths.

## Width ladder (width-only, mirrors LM s1–s5)

Width axis = `embed_dim`. **`head_dim = 64` and `depth = 12` held constant**;
`num_heads = embed_dim / 64`. `drop_path = dropout = 0` at every scale (so the
only thing varying is width — regularization that scales with width would
confound the transfer test). Built via timm geometry overrides on
`vit_base_patch16_224`, 224px / patch16 (196 tokens).

**Biases kept (original ViT).** Linear biases stay at timm ViT defaults
(qkv/proj/fc1/fc2/head all biased); LayerNorm affine kept. (The LM arm is bias-free
because torchtitan `Linear` defaults to `bias=False`; here we keep the stock ViT.)
Each bias lives *inside* its Linear's `ParametrizedModule` (`…qkv.inner.bias`), so
it shares that layer's per-layer LR (`lr_prefactor · width_dim^(−c)`) — identical
to how the original (pre-refactor) ViT wrapper already handled them. Both arms wrap
biases identically, so the only between-arm difference is still the per-layer `c`;
the transfer comparison is unaffected. Empirically verified: the parametrized
coord check stays width-stable across embed 128→1024 *with* biases (same numbers as
bias-free), so the muP scaling is intact. `pos_embed` / `cls_token` are additive
O(1) terms in the `_other` group (fixed-std init + constant LR — conventional muP
for additive embeddings).

| scale | embed_dim | heads | params (11821 cls) | optim state (AdamW, 16 B/param) | seeds | LR grid |
|-------|-----------|-------|--------------------|--------------------------------|-------|---------|
| s1 | 256  | 4  | 12.8M  | 0.2 GB | 3 | full (7) |
| s2 | 512  | 8  | 44.4M  | 0.7 GB | 3 | full (7) — **measure base** |
| s3 | 1024 | 16 | 164.3M | 2.6 GB | 2 | narrow (3) |
| s4 | 2048 | 32 | 630.5M | 10.1 GB | 2 | narrow (3) |
| s5 | 4096 | 64 | 2.47B  | 39.5 GB | 1 | transfer LR only (1) |

Seed counts and LR-grid narrowing match the LM ladder. Full grid =
`[3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]`. **No scale hardcodes its LR set** —
every scale defaults to the full grid; the coordinators narrow s3/s4 to a 3-point
bracket (argmin ± 1 grid step) and s5 to the single transferred optimum *at
runtime*, by releasing per-arm sentinel files. So the "narrow (3)" / "transfer (1)"
column above is the effective set the gates release, not a value baked into the
config — the optimum is always data-driven, never assumed.

## Single-GPU feasibility (no FSDP)

The vision trainer is **single-GPU** (`model.to(device)` + `torch.compile`, no
DDP/FSDP). Each model must fit one GH200 (96 GB HBM).

- **Memory: fits at every scale.** Worst case s5: 39.5 GB optimizer state +
  activations (grad-checkpointing enabled) leaves comfortable headroom under 96 GB.
- **The s5 risk is wall-time, made survivable by checkpoint chaining.** A 2.47B
  ViT on one GPU with no sharding and an epoch budget (~95k steps) won't finish in
  one 48 h wall. Handled by **resume chains** (s4 = 2 links, s5 = 4 links): each
  link checkpoints every 1000 steps, the next link resumes from the latest
  checkpoint via `--dependency=afterany`, and once `final_metrics.json` exists the
  remaining links no-op. So a wall-time TIMEOUT is recoverable; at most ~1000 steps
  are lost per interruption. Still worth measuring sec/step on the server to size
  the chain length, but it is no longer a single-shot gate. (s1–s3 run in one job,
  chain = 1.)

## Automated pipeline (DAG, mirrors LM `pipeline.sh`)

`bash experiments/vision/pipeline.sh` (login node, `DRY=1` to preview) submits the
whole DAG upfront — submit-only, exits in seconds. Decisions unknowable at submit
time are enforced by runtime sentinel files (`launch_sweep --require`), so the
later stages are pre-submitted and self-gate when their coordinator releases them.

- **Stage A** — s1 + s2 `mup-no --measure-only` full LR sweeps: the alignment
  sources. Logs per-layer `(z0·dW, dZ·w0, dZ·dW)` to `metrics.json` (LRs never
  changed; measurement is passive).
- **Coord B** (afterany A) — pick best mup-no LR per scale (`pipeline_analyze
  best-lr`, lowest val loss), **export** the alignment table from that LR's runs,
  write gate report (G1 seed stability, G2 source-LR stability, G3 s1-vs-s2).
- **Stage C** (afterany B, gated on `s2_align.json`) — s1 + s2 `maxP-meas` sweeps:
  prefactor re-tune under the measured table.
- **Coord C** (afterany C) — best maxP-meas LR per scale, s1-vs-s2 consistency,
  releases the 3-point s3 LR bracket (argmin ± 1 grid step) per arm via sentinels.
- **Stage D** (afterany C, per-LR gated) — s3 full-grid candidates both arms; only
  the released bracket runs, the rest exit in ~1 s.
- **Coord D** (afterany D) — transfer proof (s3 argmin == transferred LR?) + **E1
  verdict** (transfer ≤ baseline + tol → continue, else stop); releases s4/s5
  sentinels. E1 fail ⇒ s4/s5 self-skip.
- **Stage E** (afterany D, per-LR gated) — s4 (chain 2) + s5 (chain 4) candidates,
  chained checkpoint-resume so they complete across wall-times.

**Export keying — deliberate divergence from LM:** table keyed by **exact PM name**
(depth fixed ⇒ block names identical across the ladder ⇒ transfers verbatim;
`patch_embed` excluded). LM pooled per op-type suffix; vision keys per exact block
(forced by the `proj` leaf collision, slightly more faithful). Does not affect the
within-vision comparison.

**Manual fallback:** `run.sh measure|export|sweep <scale>` runs the same phases by
hand without the DAG/gating, for debugging.

## Training settings

- Optimizer AdamW (β=0.9/0.999, weight_decay 0.05), bf16 autocast, grad clip 1.0.
- Cosine LR schedule, warmup 5% of steps.
- Label smoothing 0.1, batch size 512 (constant across all widths), **4 epochs**
  of imagenet-12k (fixed sample budget; ≈95k steps/run). grad-checkpointing on,
  compile on.
- **Checkpointing (all scales):** every 1000 steps, keep latest 1 (+ a `final.pt`),
  atomic write, model + optimizer + RNG saved. `final_metrics.json` is the
  completion sentinel. **Resume chains** for s4 (2 links) and s5 (4 links) via
  `--dependency=afterany`; `--resume` loads the latest checkpoint and the run
  continues to the same optimizer-step budget. **Resume-boundary honesty:** the
  train loader now shuffles (deterministic per-pass reseed), but within-pass
  position is not restored — on resume the interrupted pass restarts, so up to one
  pass (epoch) of *reshuffled* data is re-seen and, because the step budget is
  fixed, an equal tail of the final pass is dropped. Bounded, roughly symmetric
  across arms, optimizer-step count exact. Moot for s1–s3 (chain=1); a small
  bounded perturbation per interruption for s4/s5.
- Metric: validation top-1 / top-5 / loss on the imagenet-12k validation split
  (in-distribution, unlike the LM's OOD c4 val). Validation cadence
  `--val-interval 500 --val-steps 50` (~190 val points over a 4-epoch run, ~10%
  compute overhead; the original `--val-steps 500` default would have burned ~⅓ of
  every run on eval — fixed). `--val-interval` must be a multiple of
  `--log-interval` (20); 500 satisfies this.
- wandb runs **online** (matches the LM arm, which also sets no `WANDB_MODE`);
  `pipeline_analyze` reads `metrics.json`, not wandb, so the decision path never
  depends on wandb connectivity.
- LR per layer = `lr_prefactor · width_dim^(−c)`; the two arms differ only in the
  per-layer `c` (mup-no: uniform 0.5 alignment; maxP-meas: measured table).

## Verification already done (local, CPU)

- **coord_check_vit PASSES** — every activation width-stable across embed
  128→1024 (qkv 0.0125, attn.proj 0.0009, fc1 0.798, fc2 ~0.52, head ~0.79, all
  flat). muP parametrization is correct.
- **Width-only geometry confirmed** — head_dim 64 and depth 12 constant at all
  five scales (verified by build+forward).
- **Bug found & fixed: train loader never shuffled** — `make_loader` set
  `drop_last` from `is_train` but never `shuffle`, so imagenet-12k was walked in
  identical order every epoch (no SGD shuffling). Now `shuffle=is_train` with a
  per-pass-reseeded generator. (An obsolete-assumption defect, exactly what to hunt.)
- **measure → export → apply smoke** — measure-only logs alignment; export builds
  a 9-layer table (block PMs + head; `patch_embed` excluded); maxP-meas loads it
  and solves a distinct per-layer c-table (head a≈0.85, readout-dominant as in LM).
- **launch_sweep dry-run** — correct job counts (s4 = 2 methods × 3 LRs × 2 seeds
  = 12), `--method maxP-meas --alignment-table … --epochs … --batch-size 512`
  wired into job.sh, and the "maxP-meas requires --alignment-table" guard fires.
  Idempotent resubmit via a `final_metrics.json` completion guard.

- **checkpoint round-trip** — save → load restores weights bit-exact (max diff
  0.0), optimizer moments + step + RNG + progress (step/epoch/samples) restored;
  `latest_checkpoint` selects the highest step; chained dry-run emits the correct
  `afterany` links (s4 = 2, s5 = 4; s1–s3 carry no `--resume`).
- **full pipeline `DRY=1`** — all 10 stage-sweeps submit, coordinators B/C/D
  generate, dependencies chain (afterany), s3/s4/s5 job.sh carry the runtime
  sentinel guard, s4/s5 emit chained links. `pipeline_analyze` smoke: `best-lr`
  picks the min-val-loss LR, `e1` returns PASS/exit 0, `gates` reports G1/G2.

**Not yet exercised (gates before the full sweep):** the smoke validated the
*library* path but rebuilt the loop in a scratch script — `train.py`'s `main()`
has **never run end-to-end** (arg parsing, the in-main table load,
`build_dataloaders`, `evaluate`, wandb/tb logging, and a real
checkpoint→interrupt→resume cycle through the training loop), and `--dry-run`
only writes `job.sh`. So the integrated path is unproven.

## Code changes made (experiments/vision)

- `maxp_timm.py` — replaced the obsolete named-model ladder (vit-s/b/l, depth
  drifted to 24 at large, dropout/drop-path scaling with width) with a clean
  width-only s1–s5 ladder; `create_model` passes embed_dim/depth/num_heads.
  Biases kept at stock timm ViT defaults (land in `_other`, constant LR).
- `train.py` — added `maxP-meas`, `--measure-only`, `--alignment-table`,
  `--c-ema`; dropped dynamic `maxP`/`mup-full`; `dynamic = measure_only`;
  alignment overrides + measure_only wired into `Parametrization`. Implemented
  `--resume` (was parsed but ignored): loads latest checkpoint, restores
  weights/optimizer/RNG/progress, completes to a step budget. Checkpointing on by
  default (every 1000 steps, keep latest 1, atomic save).
- `export_alignment.py` (new) — exact-name table from `metrics.json` (avoids the
  tensorboard-on-aarch64 problem from the LM run).
- `coord_check_vit.py` — width-only widths + `alignment="no"` (was the wrong
  `"full"`); now the correctness gate.
- `launch_sweep.py` — two-arm ladder, fixed `--epochs` budget, constant batch
  size, completion guard, method-specific args (`--alignment-table` /
  `--measure-only`), and **resume chaining** (`--chain`; s4 = 2, s5 = 4 links via
  `--dependency=afterany`, with sbatch-failure abort and job-id capture). No
  `--run-date`. `utils.py` — atomic checkpoint save/prune + `load_checkpoint` /
  `latest_checkpoint`.
- `hf_vision_data.py` — **fixed missing train shuffle** (`shuffle=is_train` +
  optional generator for reproducible per-pass order).
- `pipeline.sh` (new) — full submit-only DAG (stages A–E + coordinators B/C/D,
  sentinel gating), mirroring `experiments/lm/pipeline.sh`.
- `pipeline_analyze.py` (new) — coordinator decision helpers (`best-lr`, `e1`,
  `gates`) reading `metrics.json` (no tensorboard dependency).
- `launch_sweep.py` — added the DAG primitives: `--tag`, `--job-ids-file`,
  `--dependency` (first-link DAG edge), `--require` (runtime sentinel guards with
  `{lr}` templating), plus the measure-only+chain guard.
- `run.sh` / `run_debug.sh` — updated to the measure/export/sweep phases.

## Open items / decisions for you

1. **s5 wall-time** — the hard gate (see Single-GPU feasibility); validate
   sec/step on the server before committing s5; fallbacks = fewer epochs / smaller
   batch / enable checkpointing / cap at s4.
2. **Budget magnitude** — 4 epochs × batch 512 are proposed defaults. More epochs
   = better absolute accuracy but more compute. Adjust if you want longer/shorter.
3. **Measure base = s2** (as in LM). Could measure at s1 instead (cheaper); s2
   chosen to match the LM protocol.

## Not done until accepted — launch gates

No SLURM jobs submitted. On acceptance:
1. scp changed files to the server.
2. **GATE #1 — dataset loads map-style + is pre-staged.** `timm/imagenet-12k-wds`
   is WebDataset; confirm `load_dataset(..., streaming=False)` returns a map-style
   `Dataset` (`hasattr(ds, "__len__")`, `ds[0]["jpg"]` decodes) — if it returns an
   `IterableDataset`, `len`/indexing/`shuffle=True` all break and the data layer
   needs a rewrite. Also **pre-stage the dataset into `HF_HOME`** (the non-streaming
   prepare is a ~1 TB download + Arrow conversion); do it once before launch so the
   84 stage-A/C jobs don't each trigger or race the prepare (and so the train.py
   smoke doesn't hang for hours on a cold cache).
3. **Disk budget.** Checkpoints are fp32 model+Adam: s5 ≈ 30 GB, s4 ≈ 7.5 GB each.
   `keep_latest_k` is 1 at every scale (set by launch_sweep; atomic save means one
   complete checkpoint always survives an interrupt), but concurrent s4/s5
   runs + the ~1 TB dataset can still total hundreds of GB — check the PLG quota
   before launch (a disk-full stalls the run; the atomic save protects the prior
   checkpoint).
4. **Real end-to-end run** (not just `launch_sweep --dry-run`): execute `train.py`
   once per path on a tiny config (debug/s1, ~50 steps) and confirm by eye —
   measure-only populates `align/*` rows in `metrics.json`; maxP-meas loads
   `s2_align.json`, applies per-layer `c`, and steps.
5. **Resume verification** (mirrors the LM `verify_resume.sh`): run a short job,
   interrupt it mid-training, relaunch with `--resume`, and confirm it loads the
   latest checkpoint and trains through to `final_metrics.json`. Validates the
   full checkpoint→interrupt→resume cycle that s4/s5 chains depend on.
6. **s5 throughput projection** — ~100 s5 steps, sec/step → project the full run,
   size the chain length (s4 = 2, s5 = 4). Measure with the production
   `--val-interval 500 --val-steps 50` (validation is ~10% overhead at those
   settings; do not project off a config with heavier eval).
7. Then launch phase 1 (measure → export → sweep).

**Post-run completeness guard:** before treating any results as final, check that
**every** run dir has `final_metrics.json`. If a chain was too short (last link
hit the wall mid-train, no successor), the run is silently incomplete — nothing
alerts. Re-running `launch_sweep` for that scale (with `--chain`) appends links
and the guard no-ops the already-complete runs.
