"""Downstream evals on training checkpoints.

Boots the model through the EXACT training path — train.build_trainer_config +
MaxPTrainer (wrappers, output scales, dims all identical to the run that wrote
the checkpoint) — loads the DCP checkpoint via torchtitan's own CheckpointManager,
then runs the fixed scoring suite from eval_tasks.py.

Single GPU only. Writes eval JSON next to the checkpoint; never writes tfevents
(metrics are disabled below) so harvested run dirs stay clean.

Usage (inside a SLURM GPU shell, venv active):

    torchrun --nproc_per_node=1 experiments/lm/eval_downstream.py \
        --run-dir /path/to/runs_phaseX/<run> --scale s4 \
        --tasks hellaswag,arc_easy,piqa,lambada,wikitext

Notes:
- The parametrization method does NOT matter for eval: per-layer LRs never
  enter the forward pass, and output scales are structural (layer_type + dims).
  All arms are therefore booted as `mup-no`; weights come from the checkpoint.
- The training dataloader is constructed at boot but never stepped; eval
  datasets are fetched by eval_tasks (prefetch once on a node with network
  access).
- Runs on 1 GPU by default. Under torchrun with more ranks the FSDP-sharded
  model forwards the same batches on every rank (correct, redundant) and only
  rank 0 writes the JSON — sharded-data eval is not implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
import torch.distributed as dist

from torchtitan.components.checkpoint import MODEL as model_key
from torchtitan.components.tokenizer import HuggingFaceTokenizer

import eval_tasks as T
import train as train_mod


def find_checkpoint_step(run_dir: str, step: int) -> int:
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"no checkpoint/ under {run_dir}")
    steps = sorted(int(m.group(1)) for d in os.listdir(ckpt_dir)
                   if (m := re.fullmatch(r"step-(\d+)", d)))
    if not steps:
        raise FileNotFoundError(f"no step-* checkpoints under {ckpt_dir}")
    if step == -1:
        return steps[-1]
    if step not in steps:
        raise FileNotFoundError(f"step {step} not in {steps}")
    return step


def build_model(args, step: int):
    """Boot MaxPTrainer on the run dir and load the checkpoint at `step`."""
    train_argv = [
        "--scale", args.scale,
        "--method", "mup-no",
        "--lr", "3e-2",
        "--seed", "1",
        "--output-dir", args.run_dir,
        "--checkpoint-interval", "5000",
        "--hf-assets-path", args.hf_assets_path,
    ]
    old_argv = sys.argv
    sys.argv = ["train.py"] + train_argv
    try:
        targs = train_mod.parse_args()
    finally:
        sys.argv = old_argv

    cfg = train_mod.build_trainer_config(targs)
    # Eval-mode overrides: never write metrics into the run dir, never compile
    # (variable warmup cost; scoring uses fixed shapes but keep boot simple).
    cfg.metrics.enable_tensorboard = False
    cfg.metrics.enable_wandb = False
    cfg.compile.enable = False
    # Disable the validator by flag, NOT by setting the field to None: torchtitan's
    # Trainer.__init__ dereferences `config.validator.enable` unconditionally,
    # so a None here is an AttributeError at boot.
    cfg.validator.enable = False

    trainer = train_mod.MaxPTrainer(cfg)

    # Load MODEL WEIGHTS ONLY. Two reasons, both load-bearing:
    #
    #  1. s3/s4 checkpoints were written by 4-GPU runs and eval runs on 1 GPU.
    #     DCP reshards model tensors fine, but the *dataloader* state asserts
    #     equality of the dp world size and aborts the load:
    #       torchtitan/components/dataloader.py:161
    #       "dp_degree is inconsistent before and after checkpoint,
    #        dataloader resharding is not supported yet."
    #     Eval never steps the dataloader, so that state is pure baggage.
    #  2. It also skips optimizer/lr-scheduler state, which at s4 is the bulk of
    #     the checkpoint's memory.
    #
    # `exclude_from_loading` is the checkpointer's own filter, applied in
    # _states_to_load(); everything except the model key is excluded, computed
    # from the live states dict so no key name is hard-coded (the checkpointer
    # raises ValueError for an excluded key that is not present).
    ckpt = trainer.checkpointer
    if model_key not in ckpt.states:
        raise RuntimeError(
            f"expected a '{model_key}' entry in checkpointer.states, got {sorted(ckpt.states)}")
    ckpt.exclude_from_loading = [k for k in ckpt.states if k != model_key]

    ok = ckpt.load(step=step)
    if not ok:
        raise RuntimeError(f"checkpointer.load(step={step}) found nothing in {args.run_dir}")

    model = trainer.model_parts[0]
    model.eval()
    return model, trainer


def main() -> None:
    p = argparse.ArgumentParser(description="Downstream evals on a training checkpoint")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--step", type=int, default=-1, help="-1 = latest checkpoint")
    p.add_argument("--tasks", default="hellaswag,arc_easy,piqa,lambada,wikitext")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mc-seq-len", type=int, default=1024)
    p.add_argument("--wikitext-seq-len", type=int, default=3072)
    p.add_argument("--limit", type=int, default=None, help="items per task (smoke tests)")
    p.add_argument("--out", default=None)
    p.add_argument("--hf-assets-path",
                   default=os.path.join(os.path.dirname(__file__), "assets/hf/Llama-3.1-8B"))
    args = p.parse_args()

    rank0 = int(os.environ.get("RANK", "0")) == 0
    os.environ.setdefault("WANDB_MODE", "disabled")

    step = find_checkpoint_step(args.run_dir, args.step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = HuggingFaceTokenizer(tokenizer_path=args.hf_assets_path)

    model, trainer = build_model(args, step)
    try:

        results: dict = {"run_dir": os.path.abspath(args.run_dir), "scale": args.scale,
                         "step": step, "limit": args.limit,
                         "templates": "eval_tasks.py fixed templates — relative use only",
                         "tasks": {}}
        bs, L = args.batch_size, args.mc_seq_len

        for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
            t0 = time.time()
            if task in ("hellaswag", "arc_easy", "piqa"):
                docs = {"hellaswag": T.build_hellaswag,
                        "arc_easy": T.build_arc_easy,
                        "piqa": T.build_piqa}[task](args.limit)
                scores = T.score_requests(model, tok, T.mc_doc_pairs(docs),
                                          device=device, batch_size=bs, seq_len=L)
                res = T.multiple_choice_metrics(docs, scores)
            elif task == "lambada":
                pairs = T.build_lambada(args.limit)
                scores = T.score_requests(model, tok, pairs,
                                          device=device, batch_size=bs, seq_len=L)
                res = T.lambada_metrics(scores)
            elif task == "wikitext":
                ids = T.build_wikitext_ids(tok, args.limit)
                pad = getattr(tok, "eos_id", None) or 0
                res = T.perplexity_over_tokens(model, ids, device=device,
                                               seq_len=args.wikitext_seq_len,
                                               batch_size=4, pad_id=pad)
            else:
                raise ValueError(f"unknown task '{task}'")
            res["seconds"] = round(time.time() - t0, 1)
            results["tasks"][task] = res
            if rank0:
                print(f"[eval] {task}: {res}")

        if rank0:
            out = args.out or os.path.join(args.run_dir, f"eval_downstream_step{step}.json")
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[eval] wrote {out}")
    finally:
        # Close the trainer (checkpointer may own a worker process) and the
        # process group; leaving either open can hang the process at exit.
        trainer.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
