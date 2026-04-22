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

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    TrainingConfig,
)
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
                         if getattr(m, "_maxp_param", None) is not None
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
            param = getattr(model, "_maxp_param", None)
            if param is None or not getattr(model, "_maxp_ready", False):
                continue
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
            for name, (z0_dw, dz_w0, dz_dw) in getattr(model, "_maxp_align", {}).items():
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


def _resolve_steps(args: argparse.Namespace) -> int:
    if args.steps is not None:
        return args.steps
    if args.scale == "debug":
        return 20
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return compute_steps(args.scale, args.seq_len, args.batch_size, world_size)


def build_trainer_config(args: argparse.Namespace) -> Trainer.Config:
    is_debug = args.scale == "debug"
    steps = _resolve_steps(args)

    model_spec = maxp_model_registry(
        scale=args.scale,
        method=args.method,
        attn_backend="sdpa",
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
            )],
        ),
        optimizer=OptimizersContainer.Config(
            lr=args.lr,
            # fused requires CUDA; foreach works on CPU and GPU
            implementation="foreach" if is_debug else "fused",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10 if is_debug else 2000,
            decay_ratio=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2 if is_debug else args.batch_size,
            seq_len=args.seq_len,
            steps=steps,
            dtype="bfloat16",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10 if is_debug else 50,
            enable_tensorboard=not is_debug,
            enable_wandb=not is_debug,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        compile=CompileConfig(enable=not is_debug),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        debug=DebugConfig(seed=args.seed),
        validator=Validator.Config(
            enable=not is_debug,
            freq=500,
            steps=50,
        ),
    )


def parse_args() -> argparse.Namespace:
    default_hf_path = os.path.join(os.path.dirname(__file__), "assets/hf/Llama-3.1-8B")
    default_dataset_path = os.path.join(os.path.dirname(__file__), "assets/c4_test")
    p = argparse.ArgumentParser(
        description="MaxP LLaMA-3 pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scale", choices=list(SCALE_CONFIGS), default="s3",
                   help="Model scale")
    p.add_argument("--method", choices=["maxP", "mup-full", "mup-no"], default="maxP", 
                   help="maxP variant")
    p.add_argument("--lr", type=float, default=1e-3, 
                   help="LR prefactor")
    p.add_argument("--alignment-warmup", type=int, default=10,
                   help="Steps before first LP re-solve (maxP only)")
    p.add_argument("--solve-interval", type=int, default=100,
                   help="Re-solve LP every N steps (maxP only)")
    p.add_argument("--sample-size", type=int, default=32,
                   help="Sequences for alignment measurement (maxP only)")
    p.add_argument("--c-ema", type=float, default=0.0,
                   help="EMA smoothing for c values (maxP only)")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Sequence length")
    p.add_argument("--steps", type=int, default=None,
                   help="Training steps (default: auto-computed as 20 × non-embed params / tokens-per-step)")
    p.add_argument("--batch-size", type=int, default=8, 
                   help="Local batch size per GPU")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed")
    p.add_argument("--output-dir", default="./outputs",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--dataset", default="c4_test",
                   help="HuggingFace dataset name or local path")
    p.add_argument("--dataset-path", default=default_dataset_path,
                   help="Override dataset path (e.g. absolute path to c4_test on disk)")
    p.add_argument("--hf-assets-path", default=default_hf_path,
                   help="Path to HF tokenizer assets (local copy)")
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
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
