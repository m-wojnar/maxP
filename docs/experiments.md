# maxP Experiment Plan

Consolidated plan for the empirical section of the paper. Every decision in
this doc is locked unless overridden in a later conversation.

## 1. Thesis the experiments defend

maxP is an adaptive per-layer learning-rate scheduler. The empirical claim is
**LR-robustness**: maxP widens the near-optimal base-LR basin by roughly
1--2 orders of magnitude relative to µP+AdamW, while matching µP at its
tuned optimum. Secondary claims: (a) maxP preserves µTransfer (one
base-LR transfers across scale without retuning); (b) per-layer LR
trajectories spontaneously reproduce a warmup-stable-decay shape; (c) the
scheduler is practical at LLM scale (overhead <1% at `solve_interval=100`).

The empirical story has three headline plots:

- **Fig. 1 (LR basin):** val-equivalent training loss vs. `lr_prefactor` at one
  middle scale, one curve per method.
- **Fig. 2 (transfer across scale):** best small-scale LR, applied zero-shot to
  larger scales, loss-vs-scale per method.
- **Fig. 3 (transfer-basin heatmap):** 2-D (scale × LR) loss-gap-to-best map;
  maxP has a wide green band, µP has a narrow green diagonal.

All three come from one experimental matrix (§4); no extra runs needed.

## 2. Codebases and architectures

| Modality | Type | Codebase | Architecture |
|---|---|---|---|
| Text | Transformer | **torchtitan** (PyTorch/Meta) | LLaMA-3 (RMSNorm, SwiGLU, RoPE, GQA) |
| Image | Transformer | **timm** (HuggingFace) | ViT-S/16, ViT-B/16, ViT-L/16 |
| Image | Non-transformer (optional) | **timm** | ConvNeXt-V2-T |

No Mamba / RWKV in scope. No multimodal. No toy/MLP sanity experiments. No S6 (10B).

### Scale ladder (text, LLaMA-3)

| Scale | Non-embed params | d_model | n_layers | n_heads | n_kv_heads | seq len |
|---|---|---|---|---|---|---|
| S1 | ~30M  | 512  | 6  | 8  | 4 | 2048 |
| S2 | ~100M | 768  | 10 | 12 | 4 | 2048 |
| S3 | ~300M | 1024 | 16 | 16 | 4 | 2048 |
| S4 | ~1B   | 2048 | 20 | 16 | 4 | 2048 |
| S5 | ~3B   | 2560 | 32 | 32 | 8 | 2048 |

Vocabulary: LLaMA-3 tokenizer (128k). GQA with n_kv_heads ≤ n_heads/2.
Geometric spread across scales is ~3.3× per step (30M → 3B, two decades in five rungs).

### Image scales

| Tag | Model | Params | Input | Patches | Role |
|---|---|---|---|---|---|
| V-S | ViT-S/16 | ~22M  | 224² | 196 | main sweep (LR-basin) |
| V-B | ViT-B/16 | ~86M  | 224² | 196 | transfer test (mid) |
| V-L | ViT-L/16 | ~305M | 224² | 196 | transfer test (large) |
| V-CNX (optional) | ConvNeXt-V2-T | ~29M | 224² | — | non-transformer check |

Rationale for V-L: fills the 86M–1B gap the user flagged. ViT-L/16 is the
canonical "large" ViT in timm and gives a ~14× param spread (22M → 305M) for
the vision µTransfer claim — parallel to the 30M → 3B spread on the LM side.
ViT-H/14 (~632M) is tempting but adds ~2.5× compute per run for marginal
additional spread; we keep it as a stretch goal if S5 LM underruns its budget.

### maxP layer annotations (same for both codebases)

- `patch_embed.proj`, token embedding: `layer_type="embedding"`.
- All internal linears (qkv, attn-proj, ffn fc1/fc2, SwiGLU w1/w2/w3):
  `layer_type="hidden"`.
- Classification head / LM head: `layer_type="readout"`.
- Attention `Q @ K^T` (parameter-free): `layer_type="readout"` with
  `width_dim = d_head`.

Use `Parametrization(..., sample_input=x)` at init so the DAG solver traces
residual-stream merges correctly. Verify with `maxp.diagnose_axis` coord-check
before each real run.

## 3. Datasets

### Text — FineWeb-Edu

- Source: HuggingFace `HuggingFaceFW/fineweb-edu`, 1.3T tokens, streaming.
- Tokenizer: LLaMA-3 (128k vocab). Store as uint32 parquet shards.
- Preprocessing protocol (run once, upfront):
  1. Stream from HF with a fixed shuffle seed.
  2. Tokenize the first 100B tokens (cumulative need across S1–S5 at 20×
     params is 0.6 + 2 + 6 + 20 + 60 ≈ 89B; +11B headroom for reruns).
  3. Write sharded parquet to local disk. Size on disk: ~300 GB.
- Every run reads its first 10N tokens from a *seeded offset* — no overlap
  within a run, identical slice across methods at the same scale.

### Image — ImageNet-21k (Winter 2021)

- Source: official ImageNet Fall/Winter 21k release (~14M images, 21,841 classes).
- Preprocessing protocol (run once, upfront):
  1. Decode all JPEGs to 224×224 center-crop-and-rescale.
  2. Re-encode as JPEG quality 90, shard as WebDataset `.tar` files.
  3. Size on disk: ~280 GB.
- Augmentation during pretraining: **minimal** (just random crop to 224 + hflip).
  No RandAugment, no MixUp. Rationale: keeping "1 epoch = 1 view" clean.
  Argue this in §Method.

### Single-epoch policy

All pretraining runs see each example exactly once. Training loss is therefore
a legitimate proxy for held-out loss; we report training loss only for the
LR-basin and scaling plots. Validation/downstream evals (§7) are on
**held-out benchmark sets**, not multiple epochs over training data.

## 4. Experiment matrix

### 4.1 LR grid (shared across all methods, all scales)

`lr_prefactor ∈ {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1}`
— 7 points, half-decade (√10) spacing.

Every method sees the same raw `lr_prefactor` values. This is the whole
point: under maxP the numeric value of `lr_prefactor` should become
nearly irrelevant.

### 4.2 Method set

| Tag | Parametrization | Alignment | Schedule | Stages |
|---|---|---|---|---|
| mup-full | µP | full | constant | all scales |
| mup-no | µP | no | constant | all scales |
| **maxP** | µP + measured | online (dynamic re-solve) | emergent | all scales |
| sf-adamw | n/a | n/a | Schedule-Free AdamW | S3 only |
| muon | n/a | n/a | constant | S3 only (cuttable) |

### 4.3 Seeds per (method, scale)

| Scale | Seeds | Runs per method | Methods | Total runs |
|---|---|---|---|---|
| S1 | 3 | 21 | 5 | 105 |
| S2 | 2 | 14 | 5 | 70 |
| S3 | 2 | 14 | 5 + Muon | 84 |
| S4 | 1 | 7 | 3 | 21 |
| S5 | 1 | 1 (single-LR transfer test) | 3 | 3 |

### 4.4 Vision experiment matrix

| Model | LRs | Methods | Seeds | Runs |
|---|---|---|---|---|
| ViT-S/16 (IN-21k, 1 epoch) | 5 (drop the 2 extremes) | 5 | 1 | 25 |
| ViT-B/16 (IN-21k, 1 epoch, transfer test) | 1 per method | 3 | 1 | 3 |
| ViT-L/16 (IN-21k, 1 epoch, transfer test) | 1 per method | 3 | 1 | 3 |
| ConvNeXt-V2-T (optional) | 5 | 3 | 1 | 15 |

The vision µTransfer claim is: tune `lr_prefactor` on ViT-S, apply zero-shot
to ViT-B and ViT-L, compare basin-hit rate to µP. Two transfer rungs (not one)
let us plot a transfer *trajectory* in vision to match the LM 5-rung ladder.

### 4.5 Ablations (all at S3, 1 seed unless noted)

Each ablation toggles one knob vs. the maxP default. All share the S3
backbone (~1 h per run at 300M × 3B tokens).

| Ablation | Values | Runs |
|---|---|---|
| Alignment norm | RMS (default), spectral | 2 (7 LRs each → 14) |
| LP objective | min Σc (default), min max c, min Σc² | 3 (7 LRs → 21) |
| solve_interval | 1, 10, 100 (default), 1000 | 4 (3 LRs each → 12) |
| `resample_w0` | False (default), True | 2 (3 LRs → 6) |
| `warm_start` | False (default), True | 2 (3 LRs → 6) |
| `use_training_activations` | False (default), True | 2 (3 LRs → 6) |
| `c_ema` / `alignment_ema` | 0 (default), 0.5, 0.9 | 3 × 3 LRs → 9 |
| `sample_size` | 8, 32 (default), 128 | 3 × 3 LRs → 9 |

Ablation total: ~83 runs × 1 h ≈ **83 h node-time** (~660 GPU-h).

### 4.6 Overhead study (at S3 and S4)

Report wall-clock and peak memory at `solve_interval ∈ {1, 10, 100, 1000}`
and `{resample_w0, warm_start, use_training_activations} ∈ {0, 1}³` (eight
combinations). Single LR, single seed. Done only at a minimal grid:

- S3: 4 (solve_interval) × 4 (a curated subset of the 8 boolean combos)
  = 16 runs × 1 h = **16 h node-time**.
- S4: 4 (solve_interval) only × 8 h = **32 h node-time** (the boolean
  optimizations' wall-clock impact transfers from S3, so we only vary the
  one knob that scales nontrivially with model size at S4).

Total overhead study: **~48 h node-time** (~384 GPU-h).

## 5. Training protocol

### Common to all runs

- Optimizer: AdamW, β₁=0.9, β₂=0.95 (LLaMA default), weight decay 0.1.
- Precision: BF16 mixed, FP32 master weights + Adam states.
- Gradient clipping: 1.0 (global norm).
- Warmup: 2000 steps of linear `lr_prefactor` ramp from 0 to target for all methods.
- No dropout.
- Batch size: fixed tokens-per-step ≈ 0.5M tokens at all LM scales (seq 2048 × global batch 256).
- Optimizer ε: 1e-8.
- No weight tying at readout (LLaMA-3 default).

### maxP-specific hyperparameters (fixed across scales)

Default config is **fully un-optimized** — every knob that trades correctness
for speed/memory is off. Ablations (§4.5) and overhead study (§4.6) measure
the cost/benefit of each optimization separately.

```python
Parametrization(
    model,
    optimizer_type="adam",
    alignment="full",                    # initial preset (used during warmup)
    warmup_steps=2000,                   # first re-solve at step 2000
    solve_interval=100,                  # re-solve every 100 steps
    sample_size=32,                      # 32 rows per alignment measurement
    c_ema=0.0,                           # OFF — no EMA on c
    alignment_ema=0.0,                   # OFF — no EMA on alignment
    resample_w0=False,                   # OFF — keep full W0 snapshot in memory
    use_training_activations=False,      # OFF — use dedicated fwd pass each solve
    warm_start=False,                    # OFF — solve LP from scratch each time
    sample_input=calibration_batch,      # enable DAG tracing (not an optimization)
)
```

The only "non-default" knobs left on are `solve_interval=100`,
`warmup_steps=2000`, and `sample_size=32` — these are *amortization settings*
(how often / how cheaply we measure), not correctness-affecting optimizations.

### Parallelism and memory

| Scale | Parallelism | Micro-batch | Grad accum | Activation ckpt |
|---|---|---|---|---|
| S1–S3 | DDP (single-node) | 32 | 1–2 | off |
| S4 | FSDP2, full shard | 16 | 2 | selective |
| S5 | FSDP2 + TP=2, full shard | 8 | 4 | selective |
| V-S, V-B | DDP | 512 (vision) | 1 | off |

torchtitan's built-in selective activation checkpointing is sufficient at S4–S5.

### Compile and attention kernels

- `torch.compile(mode="reduce-overhead")` on all models. Fall back to eager if
  maxP's dynamic param-group updates trip the compiler — test at S1 first.
- Flash-attention-2 or 3 via SDPA (torchtitan default).

## 6. Evaluation protocol

### Primary metric — training loss on held-out single-epoch data

Last 200 training steps' mean loss = headline number. With single-epoch
training, no example is ever seen twice, so this is a valid generalization
proxy. State this in §Method of the paper.

### LM downstream evals (at end of training, S3–S5 only)

Single pass through each benchmark using lm-eval-harness. No training data
contamination.

- **HellaSwag** (commonsense)
- **ARC-Easy** + **ARC-Challenge** (science QA)
- **PIQA** (physical reasoning)
- **MMLU** (5-shot, academic knowledge)

Cost: ~5 GPU-hours total across all runs.

### Vision downstream eval

- **ImageNet-1k top-1 accuracy** via linear probe on frozen representations
  (1000-class, 1.28M train, 50k val). Cost: negligible.
- Also report **zero-shot 21k top-1** on the held-out IN-21k validation split.

## 7. Compute budget (8×GH200, 10k GPU-hour ceiling)

Per-node wall-clock converted to GPU-hours via × 8. Wall-clock per run is
projected from FLOPs ≈ 6ND at ~2 PFLOPs sustained (≈40% MFU on 8×GH200),
padded for data loading and checkpoint I/O at small scales where that
dominates.

| Block | Runs | Time/run | Node-h | GPU-h |
|---|---|---|---|---|
| LM S1 (30M × 0.6B tok)  | 105 | 6 min  | 11    | 84    |
| LM S2 (100M × 2B tok)   | 70  | 16 min | 19    | 150   |
| LM S3 (300M × 6B tok)   | 84  | 2 h    | 168   | 1,344 |
| LM S4 (1B × 20B tok)    | 21  | 16 h   | 336   | 2,688 |
| LM S5 (3B × 60B tok, transfer) | 3 | 160 h | 480 | 3,840 |
| Vision ViT-S (IN-21k 1 ep)   | 25 | 6 h  | 150 | 1,200 |
| Vision ViT-B (transfer)      | 3  | 20 h | 60  | 480   |
| Vision ViT-L (transfer)      | 3  | 40 h | 120 | 960   |
| Ablations (S3)               | 83 | 2 h  | 166 | 1,328 |
| Overhead study (S3)          | 16 | 2 h  | 32  | 256   |
| Overhead study (S4)          | 4  | 16 h | 64  | 512   |
| Downstream evals (LM)        | ~20 | 15 min | 5  | 40   |
| Downstream evals (vision)    | 6  | 30 min | 3  | 24   |
| ConvNeXt-V2-T (optional)     | 15 | 4 h  | 60  | 480   |
| **Subtotal (excl. optional ConvNeXt)** | | | **1,836** | **14,686** |
| Slop / reruns buffer (~10%)  |    |      | 184 | 1,469 |
| **Total**                    |    |      | **~2,020** | **~16,155** |

Fits tightly against the 10k GPU-h ceiling. ConvNeXt-V2-T (480 GPU-h) is the
first thing cut if anything slips; see the cut list in §11.

## 8. Storage budget (2 TB disk)

| Item | Size |
|---|---|
| FineWeb-Edu tokenized (LLaMA-3, 100B tokens uint32) | 300 GB |
| ImageNet-21k preprocessed (224², JPEG q=90) | 280 GB |
| Final BF16 checkpoints (all runs) | 130 GB |
| Resumable training states (S5 only, 2 per run) | ~60 GB |
| Loss curves + per-layer LR trajectories (JSON/CSV) | 10 GB |
| Downstream eval dumps | 5 GB |
| Ablation and overhead checkpoints | 25 GB |
| Logs, TensorBoard, WandB cache | 20 GB |
| Debug / exploratory artifacts | 50 GB |
| **Total** | **~880 GB** |

Fits in **~44% of disk**. Enough headroom for a second dataset (e.g., DCLM)
or higher-res ImageNet-21k re-encoding if needed. Checkpoint accounting:
sum over runs of `2 × params` bytes (BF16) — S1 6 GB, S2 14 GB, S3 50 GB,
S4 42 GB, S5 18 GB → ~130 GB total.

## 9. Timeline (indicative, ~8 weeks full-time)

| Week | Phase | Deliverable |
|---|---|---|
| 1 | Integration | torchtitan wrapper: `wrap_llama.py`; S1 runs pass coord-check |
| 2 | Integration | timm wrapper: `wrap_vit.py`, `wrap_convnext.py`; V-S runs pass coord-check; data pipelines built |
| 3 | Small-scale sweep | S1–S2 full sweep; maxP passes debug checks; fix any bugs |
| 4 | Ablations + S3 sweep | S3 full sweep (main battleground for ablations); ablations complete (~83 h of S3 runs) |
| 5 | Overhead study + S4 sweep | Overhead numbers locked; S4 runs complete (168 h node-time, ~7 days) |
| 6 | S5 + vision sweep | S5 transfer-test runs (serialize, ~10 days wall); ViT-S sweep on a second node if available |
| 7 | Vision transfer + downstream evals | ViT-B and ViT-L transfer tests; lm-eval-harness run; IN-1k linear probe |
| 8 | Buffer for reruns, plot making, paper drafting | Headline plots (Figs 1–3), Table 1 (design space), ablation table |

S5 runs serialize on a single 8-GPU node (3 runs × ~80 h ≈ 10 days wall). If a
second node becomes available temporarily, parallelize these.

## 10. Logging and artifact management

### Per-run log schema (written to `runs/<date>_<scale>_<method>_<lr>_<seed>/`)

- `config.json` — full config including maxP hyperparameters.
- `metrics.jsonl` — one line per logging step: `{step, train_loss, lr_prefactor, per_layer_lr: {name: η_l}, per_layer_c: {name: c_l}, alignment: {name: (α_z0dW, α_dZw0, α_dZdW)}, wall_time, gpu_mem_peak}`.
- `final_ckpt.pt` — BF16 weights only (plus EMA of weights if used).
- `final_eval.json` — downstream benchmark scores (filled in after pretraining).
- `profile.json` — wall-clock and memory stats (for overhead study).

### Analysis pipeline

- One pandas script (`analysis/collect.py`) globs all `runs/*/metrics.jsonl` and
  produces a single merged dataframe.
- Plots are generated from the dataframe only — no per-plot rerun needed.
- Headline-plot code lives in `analysis/fig1_basin.py`, `fig2_transfer.py`,
  `fig3_heatmap.py`; each takes the dataframe and writes `figs/*.pdf`.

### Reproducibility artifact

- Release the `runs/` dataframe (CSV + parquet) alongside code.
- Config files and training scripts are in-repo.
- Tokenized FineWeb-Edu shard → too large for release; publish the seed and
  preprocessing script so a reviewer can rebuild bit-for-bit.

## 11. Risk register and contingency cuts

Ordered by drop-priority if compute slips. Cut from the top until slack returns.

| # | What to cut | Impact on paper | GPU-h saved |
|---|---|---|---|
| 1 | ConvNeXt-V2-T (already optional) | lose non-transformer vision claim | 480 |
| 2 | Muon baseline at S3 | one-sentence weakening of §Related Work | ~120 |
| 3 | Schedule-Free AdamW at S3 | weakens Table 1; respond in rebuttal | ~160 |
| 4 | ViT-L/16 transfer | transfer ladder collapses to ViT-S → ViT-B only | 960 |
| 5 | S4 → 3 LRs instead of 7 | narrower S4 basin plot | ~770 |
| 6 | S5 → 2 methods (mup-full, maxP only) | lose mup-no at largest scale | 640 |
| 7 | Downstream evals | lose benchmark numbers | 64 |
| 8 | Ablation: `sample_size` sweep | minor ablation loss | ~70 |

### Watchlist (things that could go wrong and how to respond)

- **`torch.compile` trips on maxP's dynamic param-group updates.**
  Fallback: eager mode at S4–S5, accept ~15% throughput hit.
- **LP solver is slow at LLaMA-scale DAG (100+ nodes).**
  Mitigation: use CBC with warm-start (already configured); if still slow,
  batch `c` updates over multiple `solve_interval` steps.
- **FSDP2 + maxP param-group sync bug.**
  Verify at S4 first, fix before attempting S5. Budget 2 days for this.
- **Alignment measurements become numerically unstable at FP8 / very small
  batch.** Use BF16 for the alignment forward pass even if training is FP8;
  bump `sample_size` to 64 if seen.
- **Running out of budget at S5.** Drop S5 entirely; largest scale becomes S4
  (1B). Paper still has 4 scales with ~33× spread; thesis survives.

## 12. Decisions locked in this doc

1. **Codebases:** torchtitan (text), timm (image). No MLP / toy experiments.
2. **Architecture:** LLaMA-3 for text; ViT-S/B/L for image; ConvNeXt-V2-T optional.
3. **Scale ladder (text):** S1=30M, S2=100M, S3=300M, S4=1B, S5=3B — 5 scales,
   ~3.3× geometric, two decades total. No S6=10B.
4. **Tokens:** 20× params; single-epoch; training-loss reporting.
5. **Datasets:** FineWeb-Edu (LLaMA-3 tokenizer, 50B tokens, 150 GB); ImageNet-21k at 224² (280 GB).
6. **LR grid:** 7 half-decade-spaced `lr_prefactor` values, shared across methods.
7. **Seeds:** 3/2/2/1/1 at S1–S5; 1 for vision.
8. **Baselines:** mup-full, mup-no, maxP at all scales, sf-adamw, muon at S3.
9. **maxP defaults (un-optimized):** `solve_interval=100`, `warmup_steps=2000`,
   `sample_size=32`, `c_ema=0.0`, `alignment_ema=0.0`, `resample_w0=False`,
   `warm_start=False`, `use_training_activations=False`. Optimizations live
   in the ablation section, not in the default config.
10. **Parallelism:** DDP at S1–S3, FSDP2 at S4, FSDP2+TP=2 at S5.
11. **Downstream evals:** HellaSwag, ARC-E/C, PIQA, MMLU for LM at S3–S5; IN-1k linear probe + IN-21k val for vision.
12. **Resumable S5 checkpoints:** yes, ~60 GB.
13. **Compute:** ~7.8k GPU-h planned + ~2.2k slop buffer = 10k cap.
14. **Storage:** ~730 GB of 2 TB.
15. **Ablations:** metric (RMS vs spectral), LP objective (3 variants),
    `solve_interval`, `resample_w0`, `warm_start`, `use_training_activations`,
    EMAs, `sample_size` — all at S3.

## 13. Open items deferred to later

- Exact LLaMA-3 config per scale (n_kv_heads ratio, ffn_dim_multiplier) — pick
  from torchtitan's config library, confirm at S1 before scaling up.
- WandB vs TensorBoard vs CSV-only for live dashboards — pick during week 1.
- Exact global batch size sweeping — fixed at 256 seqs × 2048 toks = 524,288
  tokens/step for now; revisit if S5 loss curves look bad.
- Whether to include a non-transformer *text* result (Mamba / RWKV) — deferred;
  cite as future work in the paper.
