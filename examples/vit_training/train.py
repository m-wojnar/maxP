#!/usr/bin/env python3
"""
Vision Transformer (ViT) Training on CIFAR-10 with optional maxP scheduler.

Usage:
    python train.py --config config.yaml
    python train.py --config config_baseline.yaml
"""

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from maxp import MaxPScheduler, create_param_groups, get_linear_layers, initialize_abc_weights


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, bias: bool) -> None:
        super(FeedForwardBlock, self).__init__()
        self.dense1 = nn.Linear(dim, int(dim * mlp_ratio), bias=bias)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(int(dim * mlp_ratio), dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, mup_scaling: bool, bias: bool) -> None:
        super(AttentionBlock, self).__init__()
        self.k = nn.Linear(dim, dim, bias=bias)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        
        if mup_scaling:
            self.scale = 1 / dim
        else:
            self.scale = 1 / math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, mup_scaling: bool, bias: bool) -> None:
        super(TransformerBlock, self).__init__()
        self.attn = AttentionBlock(dim, mup_scaling, bias)
        self.ffn = FeedForwardBlock(dim, mlp_ratio, bias)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, n_layers: int, mup_scaling: bool, bias: bool) -> None:
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, mlp_ratio, mup_scaling, bias) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class Patchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super(Patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, -1, self.patch_size * self.patch_size * C)
        return x


class ViT(nn.Module):
    def __init__(
        self, 
        image_size: int, 
        patch_size: int, 
        n_layers: int, 
        embed_dim: int, 
        mlp_ratio: float, 
        n_classes: int, 
        mup_scaling: bool,
        bias: bool,
    ) -> None:
        super(ViT, self).__init__()
        self.patchify = Patchify(patch_size)
        self.embed = nn.Linear(patch_size * patch_size * 3, embed_dim, bias=bias)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, embed_dim))
        self.transformer = Transformer(embed_dim, mlp_ratio, n_layers, mup_scaling, bias)
        self.readout = nn.Linear(embed_dim, n_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, *_ = x.shape

        x = self.patchify(x)
        x = self.embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed

        x = self.transformer(x)
        x = self.readout(x[:, 0])

        return x


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_optimizer(model: nn.Module, config: dict, param_groups=None) -> torch.optim.Optimizer:
    opt_type = config["optimizer"]["type"].lower()
    lr = config["optimizer"]["lr_prefactor"]
    wd = config["optimizer"].get("weight_decay", 0.0)

    params = param_groups if param_groups is not None else model.parameters()

    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def get_data_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, test_loader


def evaluate(model: nn.Module, iter: Iterator, n_steps: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(n_steps):
            X, y = next(iter)
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def infinite_iter(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train(config: dict) -> None:
    # Setup
    torch.manual_seed(config["seed"])
    device = get_device(config["device"])
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Build model
    model = ViT(
        image_size=config["model"]["image_size"],
        patch_size=config["model"]["patch_size"],
        n_layers=config["model"]["n_layers"],
        embed_dim=config["model"]["embed_dim"],
        mlp_ratio=config["model"]["mlp_ratio"],
        n_classes=config["model"]["output_dim"],
        mup_scaling=config["maxp"]["use_maxp"] and config["maxp"]["parametrization"] == "mup",
        bias=config["model"]["bias"],
    )
    
    # Count Linear layers for initialization
    linear_layers = get_linear_layers(model)
    n_linear = len(linear_layers)
    print(f"Found {n_linear} Linear layers (excluding attention)")

    use_maxp = config["maxp"]["use_maxp"]
    parametrization = config["maxp"]["parametrization"]
    alignment = config["maxp"]["alignment"]
    opt_type = "adam" if "adam" in config["optimizer"]["type"].lower() else "sgd"

    # Initialize weights with ABC parametrization
    initialize_abc_weights(
        model,
        parametrization=parametrization,
        optimizer=opt_type,
        alignment=alignment,
    )

    if use_maxp:
        # Create param groups with per-layer LRs
        lr_prefactor = config["optimizer"]["lr_prefactor"]
        param_groups = create_param_groups(
            model,
            lr_prefactor=lr_prefactor,
            parametrization=parametrization,
            optimizer=opt_type,
            alignment=alignment,
        )
        optimizer = get_optimizer(model, config, param_groups)

        # Create maxP scheduler
        scheduler = MaxPScheduler(
            optimizer,
            model,
            parametrization=parametrization,
            alignment_assumption=alignment,
            lr_prefactor=lr_prefactor,
            warmup_steps=config["maxp"]["warmup_steps"],
            solve_interval=config["maxp"]["solve_interval"],
            alignment_norm=config["maxp"]["alignment_norm"],
        )
        print(f"Using maxP scheduler with {parametrization} {alignment} parametrization")
    else:
        optimizer = get_optimizer(model, config)
        scheduler = None
        print("Using baseline training (no maxP)")

    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Apply torch.compile if enabled
    compile_model = config.get("compile", False)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    model.train()

    # Data
    train_loader, test_loader = get_data_loaders(config)
    train_iter, test_iter = infinite_iter(train_loader), infinite_iter(test_loader)

    # Capture initial state for maxP
    if scheduler is not None:
        X_init, _ = next(iter(train_loader))
        X_init = X_init.to(device)
        scheduler.capture_initial(X_init)

    # Training loop
    n_steps = config["n_steps"]
    n_val_steps = config["n_val_steps"]
    log_freq = config["log_freq"]
    val_freq = config.get("val_freq", log_freq)  # Default to log_freq if not specified
    log_file = open(output_dir / "train.log", "w")

    print(f"Starting training for {n_steps} steps...")
    print(f"Logging train metrics every {log_freq} steps, val metrics every {val_freq} steps")
    
    # Build CSV header with dynamic columns for per-layer stats if using maxP
    base_cols = ["step", "train_loss", "train_acc", "test_loss", "test_acc"]
    if scheduler is not None:
        n_layers = scheduler.n_layers
        lr_cols = [f"lr_{i}" for i in range(n_layers)]
        alpha_cols = [f"alpha_{i}" for i in range(n_layers)]
        omega_cols = [f"omega_{i}" for i in range(n_layers)]
        u_cols = [f"u_{i}" for i in range(n_layers)]
        header_cols = base_cols + lr_cols + alpha_cols + omega_cols + u_cols
    else:
        header_cols = base_cols
    log_file.write(",".join(header_cols) + "\n")

    for step in range(1, n_steps + 1):
        X, y = next(train_iter)
        X, y = X.to(device), y.to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        # Check divergence
        if not math.isfinite(loss.item()):
            print(f"Step {step}: Loss diverged to {loss.item()}")
            break

        # Backward
        loss.backward()
        optimizer.step()

        # Update scheduler
        if scheduler is not None:
            scheduler.step(X)

        # Compute batch train accuracy
        with torch.no_grad():
            train_acc_batch = (logits.argmax(1) == y).float().mean().item()

        # Logging - train metrics every log_freq, val metrics every val_freq
        if step % log_freq == 0 or step == 1:
            # Evaluate on validation set if val_freq is reached
            if step % val_freq == 0 or step == 1:
                test_loss, test_acc = evaluate(model, test_iter, n_val_steps, device)
                model.train()
            else:
                test_loss, test_acc = float('nan'), float('nan')

            # Build log row
            base_values = [step, f"{loss.item():.6f}", f"{train_acc_batch:.6f}", f"{test_loss:.6f}", f"{test_acc:.6f}"]
            
            if scheduler is not None:
                lrs = scheduler.get_last_lr()
                alpha, omega, u = scheduler.get_alignment()
                
                lr_values = [f"{lr:.6e}" for lr in lrs]
                alpha_values = [f"{a:.6f}" if alpha else "nan" for a in (alpha or [float('nan')] * n_layers)]
                omega_values = [f"{o:.6f}" if omega else "nan" for o in (omega or [float('nan')] * n_layers)]
                u_values = [f"{v:.6f}" if u else "nan" for v in (u or [float('nan')] * n_layers)]
                
                row_values = base_values + lr_values + alpha_values + omega_values + u_values
            else:
                row_values = base_values
            
            log_file.write(",".join(map(str, row_values)) + "\n")
            log_file.flush()

            # Print with validation info if available
            if not math.isnan(test_acc):
                print(f"Step {step:>6} | train_loss: {loss.item():.4f} | train_acc: {train_acc_batch:.2%} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2%}")
            else:
                print(f"Step {step:>6} | train_loss: {loss.item():.4f} | train_acc: {train_acc_batch:.2%}")

    log_file.close()

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_iter, n_val_steps, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")

    # Save model
    if config["logging"].get("save_model", False):
        torch.save(model.state_dict(), output_dir / "model.pt")
        print(f"Model saved to {output_dir / 'model.pt'}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args = args.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
