"""MaxP LLaMA-3 pre-training entry point.

Uses torchtitan's Trainer with MaxPConverter and per-layer LRs from Parametrization.

Usage (single GPU, debug smoke test):

    python experiments/lm/train.py --scale debug --steps 20

Usage (real training, 8 GPUs):

    torchrun --nproc_per_node=8 experiments/lm/train.py \\
        --scale s3 --method maxP --lr 1e-3 --steps 10000 \\
        --dataset HuggingFaceFW/fineweb-edu \\
        --output-dir ./outputs/s3_maxP_lr1e-3_s1
"""

from __future__ import annotations

import argparse
import os

import torch
torch._dynamo.config.recompile_limit = 100

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config.configs import ActivationCheckpointConfig, CommConfig, CompileConfig, DebugConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer

from maxp_converter import MaxPConverter
from maxp_llama3 import maxp_model_registry, compute_steps, SCALE_CONFIGS
import fineweb  # noqa: F401 — registers fineweb-edu in torchtitan's DATASETS dict


class MaxPTrainer(Trainer):
    """Trainer that calls param.step() for dynamic maxP and logs per-layer metrics."""

    def train_step(self, data_iterator):
        # Pre-fetch a batch to capture real tokens for alignment measurement
        # before the optimizer step modifies weights.
        needs_capture = [m for m in self.model_parts
                         if getattr(m, "_is_dynamic", False)
                         and not getattr(m, "_maxp_ready", False)]

        if needs_capture:
            batch = next(data_iterator)
            tokens = batch[0]["input"].detach()
            for model in needs_capture:
                model._maxp_sample_x = tokens
                model._maxp_param.capture_initial(tokens)
                model._maxp_ready = True

        super().train_step(data_iterator)

        # Dynamic maxP: re-solve LP and sync LRs after each optimizer+scheduler step.
        # lr_prefactor is read from the "_other" group whose lr = lr_prefactor * wsd_factor,
        # so _sync_lrs computes per-layer lr = lr_prefactor * wsd_factor * n^(-c).
        for model, opt in zip(self.model_parts, self.optimizers.optimizers):
            if not getattr(model, "_is_dynamic", False) or not getattr(model, "_maxp_ready", False):
                continue
            param = model._maxp_param
            for g in opt.param_groups:
                if g.get("layer_name") == "_other":
                    param.lr_prefactor = g["lr"]
                    break
            param.step(model._maxp_sample_x, opt)
            model._maxp_align = {
                name: (pm.align_z0_dW, pm.align_dZ_w0, pm.align_dZ_dW)
                for name, pm in param._pms
                if pm.weight is not None and pm.align_z0_dW is not None
            }

        if not self.metrics_processor.should_log(self.step):
            return
    
        extra: dict = {}
        for model in self.model_parts:
            for name, vals in getattr(model, "_maxp_align", {}).items():
                z0_dw, dz_w0, dz_dw = vals
                extra[f"align/z0_dW/{name}"] = z0_dw
                extra[f"align/dZ_w0/{name}"] = dz_w0
                extra[f"align/dZ_dW/{name}"] = dz_dw
        for opt in self.optimizers.optimizers:
            for g in opt.param_groups:
                ln = g.get("layer_name")
                if ln:
                    extra[f"lr/{ln}"] = g["lr"]
        if extra:
            self.metrics_processor.logger.log(extra, self.step)


def _local_batch_size(global_batch_size: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert global_batch_size % world_size == 0, (
        f"--batch-size {global_batch_size} not divisible by WORLD_SIZE {world_size}"
    )
    return global_batch_size // world_size


def _resolve_steps(args: argparse.Namespace) -> int:
    if args.steps is not None:
        return args.steps
    if args.scale == "debug":
        return 20
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return compute_steps(args.scale, args.seq_len, _local_batch_size(args.batch_size), world_size)


def build_trainer_config(args: argparse.Namespace) -> Trainer.Config:
    is_debug = args.scale == "debug"
    steps = _resolve_steps(args)
    local_batch_size = _local_batch_size(args.batch_size)

    model_spec = maxp_model_registry(
        scale=args.scale,
        method=args.method,
        attn_backend="sdpa",
        vocab_size=args.vocab_size,
    )

    dataloader_config = HuggingFaceTextDataLoader.Config(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    return Trainer.Config(
        model_spec=model_spec,
        hf_assets_path=args.hf_assets_path,
        dump_folder=args.output_dir,
        model_converters=ModelConvertersContainer.Config(
            converters=[MaxPConverter.Config(
                method=args.method,
                lr_prefactor=args.lr,
                alignment_warmup=args.alignment_warmup,
                solve_interval=args.solve_interval,
                sample_size=args.sample_size,
                c_ema=args.c_ema,
                measure_only=args.measure_only,
                alignment_table=args.alignment_table,
                indep_wd=args.indep_wd,
            )],
        ),
        optimizer=OptimizersContainer.Config(
            lr=args.lr,
            weight_decay=args.weight_decay,
            implementation="foreach" if is_debug else "fused",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10 if is_debug else int(0.1 * steps),
            decay_ratio=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=args.seq_len,
            steps=steps,
            dtype="bfloat16",
        ),
        dataloader=dataloader_config,
        metrics=MetricsProcessor.Config(
            log_freq=10 if is_debug else 50,
            enable_tensorboard=not is_debug,
            enable_wandb=not is_debug,
        ),
        checkpoint=CheckpointManager.Config(
            enable=args.checkpoint_interval is not None,
            interval=args.checkpoint_interval,
            last_save_model_only=False,
            keep_latest_k=2,
            async_mode="async",
        ),
        compile=CompileConfig(enable=not is_debug),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        comm=CommConfig(init_timeout_seconds=600, train_timeout_seconds=1800),
        debug=DebugConfig(seed=args.seed),
        validator=Validator.Config(
            enable=not is_debug, 
            freq=500, 
            steps=50, 
            dataloader=HuggingFaceTextDataLoader.Config(
                dataset=args.val_dataset, 
                infinite=False
            )
        ),
    )


def parse_args() -> argparse.Namespace:
    default_hf_path = os.path.join(os.path.dirname(__file__), "assets/hf/Llama-3.1-8B")
    p = argparse.ArgumentParser(
        description="MaxP LLaMA-3 pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scale", choices=list(SCALE_CONFIGS), default="s3",
                   help="Model scale")
    p.add_argument("--method", choices=["maxP", "mup-no", "maxP-meas"], default="maxP",
                   help="maxP variant")
    p.add_argument("--measure-only", action="store_true",
                   help="Measure + log alignment without changing LRs (mup-no source runs)")
    p.add_argument("--alignment-table", default=None,
                   help="JSON alignment table from export_alignment.py (maxP-meas only)")
    p.add_argument("--indep-wd", action="store_true",
                   help="Independent (fully decoupled) weight decay: keep the AdamW decay "
                        "step lr*wd*p identical across param groups and widths (it follows "
                        "the LR schedule but not the per-layer n^{-c} multiplier). "
                        "Maintained by the library across LR re-solves; works with every "
                        "method.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="LR prefactor")
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="AdamW weight decay (torchtitan default 0.1). Note the decay step is "
                        "lr*wd*p, so per-layer LRs rescale per-layer decay unless --indep-wd.")
    p.add_argument("--alignment-warmup", type=int, default=100,
                   help="Steps before first LP re-solve (maxP only)")
    p.add_argument("--solve-interval", type=int, default=200,
                   help="Re-solve LP every N steps (maxP only)")
    p.add_argument("--sample-size", type=int, default=8,
                   help="Sequences for alignment measurement (maxP only)")
    p.add_argument("--c-ema", type=float, default=0.0,
                   help="EMA smoothing for c values (maxP only)")
    p.add_argument("--seq-len", type=int, default=3072,
                   help="Sequence length")
    p.add_argument("--steps", type=int, default=None,
                   help="Training steps (default: auto-computed as 20 × non-embed params / tokens-per-step)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Global batch size (divided by WORLD_SIZE to get per-GPU)")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed")
    p.add_argument("--output-dir", default="./outputs",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--dataset", default="fineweb-edu-10bt",
                   help="HuggingFace dataset name or local path")
    p.add_argument("--vocab-size", type=int, default=128256,
                   help="Model vocab (embeddings + head). Default matches the LLaMA-3 tokenizer.")
    p.add_argument("--dataset-path", default=None,
                   help="Override dataset path (e.g. absolute path to c4_test on disk)")
    p.add_argument("--val-dataset", default="c4_validation",
                   help="Validation dataset name (default c4_validation)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader num_workers for prefetching")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Batches prefetched per DataLoader worker")
    p.add_argument("--hf-assets-path", default=default_hf_path,
                   help="Path to HF tokenizer assets (local copy)")
    p.add_argument("--checkpoint-interval", type=int, default=None,
                   help="Save checkpoint every N steps")
    return p.parse_args()


def main() -> None:
    init_logger()

    args = parse_args()
    config = build_trainer_config(args)
    trainer = MaxPTrainer(config)
    try:
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if int(os.environ.get("RANK", "0")) == 0:
            # Sentinel for chained SLURM jobs: lets follow-up links skip
            # without spinning up torchrun (see launch_sweep.py).
            open(os.path.join(args.output_dir, "COMPLETED"), "w").close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
