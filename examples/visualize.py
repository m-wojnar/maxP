#!/usr/bin/env python3
"""
Visualization script for training results.

Features:
- Groups runs by configurable features (depth, width, etc.)
- Selects best run per group (by final loss or accuracy)
- Plots multiple experiments together (baseline vs maxP)
- Supports smoothing and various plot configurations

Usage:
    python visualize.py --help
    python visualize.py --exp maxp=outputs/mlp_sweep --exp baseline=outputs/mlp_baseline
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


@dataclass
class RunData:
    """Container for data from a single training run."""
    
    path: str
    config: Dict[str, Any]
    metrics: pd.DataFrame  # Columns: step, train_loss, train_acc, test_loss, test_acc
    
    @classmethod
    def from_directory(cls, path: str) -> Optional["RunData"]:
        """Load run data from output directory."""
        path = Path(path)
        config_path = path / "config.json"
        log_path = path / "train.log"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        
        # Load metrics from CSV log
        if log_path.exists():
            try:
                metrics = pd.read_csv(log_path)
            except Exception:
                metrics = pd.DataFrame()
        else:
            metrics = pd.DataFrame()
        
        return cls(path=str(path), config=config, metrics=metrics)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation (e.g., 'model.hidden_dim')."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


@dataclass
class GroupKey:
    """Hashable key for grouping runs."""
    
    features: Dict[str, Any]
    
    def __hash__(self) -> int:
        return hash(frozenset(self.features.items()))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupKey):
            return NotImplemented
        return self.features == other.features
    
    def __str__(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.features.items()))
    
    def __repr__(self) -> str:
        return f"GroupKey({self.features})"


@dataclass
class AggregatedRun:
    """Result of aggregating/selecting runs from a group."""
    
    run: RunData
    metadata: Dict[str, Any] = field(default_factory=dict)


class RunCollector:
    """Collects and manages training runs."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.runs: List[RunData] = []
        self._collect_runs()
    
    def _collect_runs(self) -> None:
        """Recursively collect all runs from base_dir."""
        for root, _, files in os.walk(self.base_dir):
            if "config.json" in files:
                run = RunData.from_directory(root)
                if run is not None and not run.metrics.empty:
                    self.runs.append(run)
    
    def filter(self, condition: Callable[[RunData], bool]) -> "RunCollector":
        """Filter runs based on a condition."""
        new_collector = object.__new__(RunCollector)
        new_collector.base_dir = self.base_dir
        new_collector.runs = [run for run in self.runs if condition(run)]
        return new_collector
    
    def group_by(self, key_fn: Callable[[RunData], Dict[str, Any]]) -> Dict[GroupKey, List[RunData]]:
        """Group runs by features returned by key_fn."""
        groups = defaultdict(list)
        for run in self.runs:
            try:
                features = key_fn(run)
                groups[GroupKey(features)].append(run)
            except Exception:
                continue
        return dict(groups)


class MetricAggregator:
    """Methods for selecting the best run from a group."""
    
    @staticmethod
    def best_by_final_loss(runs: List[RunData], window: int = 10, metric: str = "train_loss") -> AggregatedRun:
        """Select run with lowest average loss over last `window` steps."""
        def score(run: RunData) -> float:
            if metric not in run.metrics.columns:
                return float("inf")
            values = run.metrics[metric].dropna().values
            if len(values) == 0:
                return float("inf")
            return float(np.mean(values[-window:]))
        
        best_run = min(runs, key=score)
        best_score = score(best_run)
        
        return AggregatedRun(
            run=best_run,
            metadata={"final_loss": best_score, "lr": best_run.get_config_value("optimizer.lr_prefactor")}
        )
    
    @staticmethod
    def best_by_final_accuracy(runs: List[RunData], window: int = 10, metric: str = "test_acc") -> AggregatedRun:
        """Select run with highest average accuracy over last `window` steps."""
        def score(run: RunData) -> float:
            if metric not in run.metrics.columns:
                return float("-inf")
            values = run.metrics[metric].dropna().values
            if len(values) == 0:
                return float("-inf")
            return float(np.mean(values[-window:]))
        
        best_run = max(runs, key=score)
        best_score = score(best_run)
        
        return AggregatedRun(
            run=best_run,
            metadata={"final_acc": best_score, "lr": best_run.get_config_value("optimizer.lr_prefactor")}
        )


def smooth_ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    smoothed = np.full_like(values, np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(values))[0]
    
    if len(valid_idx) == 0:
        return smoothed
    
    first = valid_idx[0]
    smoothed[first] = values[first]
    
    for i in range(first + 1, len(values)):
        if np.isnan(values[i]):
            smoothed[i] = smoothed[i - 1]
        else:
            prev = smoothed[i - 1] if not np.isnan(smoothed[i - 1]) else values[i]
            smoothed[i] = alpha * values[i] + (1 - alpha) * prev
    
    return smoothed


class StyleManager:
    """Manages colors and styles for plots."""
    
    COLORS = [
        "#e41a1c",  # Red
        "#377eb8",  # Blue
        "#4daf4a",  # Green
        "#984ea3",  # Purple
        "#ff7f00",  # Orange
        "#a65628",  # Brown
        "#f781bf",  # Pink
        "#999999",  # Gray
    ]
    
    LINE_STYLES = ["-", "--", ":", "-."]
    
    def get_color(self, idx: int) -> str:
        return self.COLORS[idx % len(self.COLORS)]
    
    def get_linestyle(self, idx: int) -> str:
        return self.LINE_STYLES[idx % len(self.LINE_STYLES)]


class GridVisualizer:
    """Creates grid plots for comparing experiments."""
    
    def __init__(self, figsize_per_subplot: Tuple[float, float] = (4, 3)):
        self.figsize_per_subplot = figsize_per_subplot
        self.style = StyleManager()
    
    def plot_grid(
        self,
        named_experiments: Dict[str, Dict[GroupKey, AggregatedRun]],
        metric: str = "train_loss",
        title: Optional[str] = None,
        smoothing_alpha: Optional[float] = None,
        ylim: Optional[Tuple[float, float]] = None,
        log_scale: bool = False,
        grid_features: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Plot a metric across experiments in a grid layout.
        
        Args:
            named_experiments: Dict mapping experiment names to their grouped runs
            metric: Column name in the log to plot
            title: Plot title
            smoothing_alpha: EMA smoothing factor (lower = more smoothing)
            ylim: Y-axis limits
            log_scale: Use log scale for y-axis
            grid_features: Which features to use for grid rows/columns
        """
        # Get all unique group keys
        all_keys = set()
        for groups in named_experiments.values():
            all_keys.update(groups.keys())
        
        if not all_keys:
            raise ValueError("No runs found in experiments")
        
        # Determine grid features
        sample_key = next(iter(all_keys))
        if grid_features is None:
            grid_features = list(sample_key.features.keys())[:2]
        
        if len(grid_features) < 2:
            grid_features = grid_features + ["_dummy"]
        
        x_feature, y_feature = grid_features[:2]
        
        # Get unique values for each feature
        def get_feature_value(key: GroupKey, feature: str) -> Any:
            return key.features.get(feature, "_")
        
        x_values = sorted(set(get_feature_value(k, x_feature) for k in all_keys))
        y_values = sorted(set(get_feature_value(k, y_feature) for k in all_keys))
        
        # Create figure
        fig_width = len(x_values) * self.figsize_per_subplot[0]
        fig_height = len(y_values) * self.figsize_per_subplot[1] + 1.0
        
        fig, axes = plt.subplots(
            len(y_values), len(x_values),
            figsize=(fig_width, fig_height),
            squeeze=False,
        )
        
        if title:
            fig.suptitle(title, y=0.98, fontsize=12)
        
        # Legend elements
        legend_elements = []
        
        # Plot each cell
        for key in all_keys:
            x_val = get_feature_value(key, x_feature)
            y_val = get_feature_value(key, y_feature)
            
            i = y_values.index(y_val)
            j = x_values.index(x_val)
            ax = axes[i, j]
            ax.grid(alpha=0.3)
            ax.set_xlabel("Step")
            
            # Plot each experiment
            for exp_idx, (exp_name, groups) in enumerate(named_experiments.items()):
                if key not in groups:
                    continue
                
                agg = groups[key]
                df = agg.run.metrics
                
                if metric not in df.columns:
                    continue
                
                steps = df["step"].values if "step" in df.columns else np.arange(len(df))
                values = df[metric].values.astype(float)
                
                if smoothing_alpha is not None:
                    values = smooth_ema(values, smoothing_alpha)
                
                color = self.style.get_color(exp_idx)
                ax.plot(steps, values, color=color, linewidth=1.5, label=exp_name)
                
                # Add to legend
                if exp_idx >= len(legend_elements):
                    lr_info = agg.metadata.get("lr", "")
                    label = f"{exp_name}" + (f" (lr={lr_info})" if lr_info else "")
                    legend_elements.append(Line2D([0], [0], color=color, label=label))
            
            # Subplot title
            ax.set_title(str(key), fontsize=9)
            
            if ylim:
                ax.set_ylim(ylim)
            if log_scale:
                ax.set_yscale("log")
        
        # Add legend
        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc="lower center",
                ncol=min(4, len(legend_elements)),
                fontsize=9,
                bbox_to_anchor=(0.5, 0.02),
            )
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        return fig
    
    def plot_single(
        self,
        named_experiments: Dict[str, Dict[GroupKey, AggregatedRun]],
        metric: str = "train_loss",
        title: Optional[str] = None,
        smoothing_alpha: Optional[float] = None,
        ylim: Optional[Tuple[float, float]] = None,
        log_scale: bool = False,
        aggregate_groups: str = "best",  # "best", "mean", "all"
    ) -> plt.Figure:
        """
        Plot a single chart aggregating all groups.
        
        Args:
            aggregate_groups: How to handle multiple groups:
                - "best": Show only the best group per experiment
                - "mean": Show mean across all groups
                - "all": Show all groups with different alpha
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.grid(alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric.replace("_", " ").title())
        
        if title:
            ax.set_title(title)
        
        legend_elements = []
        
        for exp_idx, (exp_name, groups) in enumerate(named_experiments.items()):
            color = self.style.get_color(exp_idx)
            
            if aggregate_groups == "best":
                # Find best group by final metric value
                best_key = None
                best_score = float("inf") if "loss" in metric else float("-inf")
                
                for key, agg in groups.items():
                    df = agg.run.metrics
                    if metric not in df.columns:
                        continue
                    values = df[metric].dropna().values
                    if len(values) == 0:
                        continue
                    score = np.mean(values[-10:])
                    if "loss" in metric:
                        if score < best_score:
                            best_score = score
                            best_key = key
                    else:
                        if score > best_score:
                            best_score = score
                            best_key = key
                
                if best_key is not None:
                    agg = groups[best_key]
                    df = agg.run.metrics
                    steps = df["step"].values if "step" in df.columns else np.arange(len(df))
                    values = df[metric].values.astype(float)
                    
                    if smoothing_alpha is not None:
                        values = smooth_ema(values, smoothing_alpha)
                    
                    ax.plot(steps, values, color=color, linewidth=2)
                    legend_elements.append(Line2D([0], [0], color=color, label=f"{exp_name} ({best_key})"))
            
            elif aggregate_groups == "all":
                for group_idx, (key, agg) in enumerate(groups.items()):
                    df = agg.run.metrics
                    if metric not in df.columns:
                        continue
                    
                    steps = df["step"].values if "step" in df.columns else np.arange(len(df))
                    values = df[metric].values.astype(float)
                    
                    if smoothing_alpha is not None:
                        values = smooth_ema(values, smoothing_alpha)
                    
                    alpha = 0.3 + 0.7 * (group_idx / max(1, len(groups) - 1))
                    ax.plot(steps, values, color=color, alpha=alpha, linewidth=1)
                
                legend_elements.append(Line2D([0], [0], color=color, label=exp_name))
        
        if ylim:
            ax.set_ylim(ylim)
        if log_scale:
            ax.set_yscale("log")
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc="best")
        
        plt.tight_layout()
        return fig


def create_grouper(features: List[str]) -> Callable[[RunData], Dict[str, Any]]:
    """Create a grouping function for the specified features."""
    def grouper(run: RunData) -> Dict[str, Any]:
        result = {}
        for feature in features:
            value = run.get_config_value(feature)
            if value is None:
                raise ValueError(f"Feature {feature} not found in config")
            result[feature.split(".")[-1]] = value
        return result
    return grouper


def get_layer_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Get all column names matching a prefix (e.g., 'alpha_' -> ['alpha_0', 'alpha_1', ...])."""
    cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    # Sort by layer index
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def plot_per_layer_metrics(
    named_experiments: Dict[str, Dict[GroupKey, "AggregatedRun"]],
    metric_prefix: str,  # e.g., "alpha", "omega", "u", "lr"
    title: Optional[str] = None,
    smoothing_alpha: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
    log_scale: bool = False,
    figsize_per_subplot: Tuple[float, float] = (4, 3),
) -> plt.Figure:
    """
    Plot per-layer metrics with one subplot per layer.
    
    Each subplot shows one layer's metric across all experiments.
    """
    style = StyleManager()
    
    # Determine number of layers from first available run
    n_layers = 0
    sample_cols = []
    for groups in named_experiments.values():
        for agg in groups.values():
            sample_cols = get_layer_columns(agg.run.metrics, metric_prefix)
            n_layers = len(sample_cols)
            if n_layers > 0:
                break
        if n_layers > 0:
            break
    
    if n_layers == 0:
        raise ValueError(f"No columns found with prefix '{metric_prefix}_'")
    
    # Create grid layout
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig_width = n_cols * figsize_per_subplot[0]
    fig_height = n_rows * figsize_per_subplot[1] + 1.0
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)
    
    legend_elements = []
    
    for layer_idx in range(n_layers):
        row = layer_idx // n_cols
        col = layer_idx % n_cols
        ax = axes[row, col]
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Step")
        
        col_name = f"{metric_prefix}_{layer_idx}"
        
        for exp_idx, (exp_name, groups) in enumerate(named_experiments.items()):
            color = style.get_color(exp_idx)
            
            # Use best group (first one for simplicity, or aggregate)
            for key, agg in groups.items():
                df = agg.run.metrics
                if col_name not in df.columns:
                    continue
                
                steps = df["step"].values if "step" in df.columns else np.arange(len(df))
                values = df[col_name].values.astype(float)
                
                if smoothing_alpha is not None:
                    values = smooth_ema(values, smoothing_alpha)
                
                ax.plot(steps, values, color=color, linewidth=1.5, alpha=0.7)
                break  # Only plot first/best group per experiment
            
            # Add to legend only once
            if layer_idx == 0:
                legend_elements.append(Line2D([0], [0], color=color, label=exp_name))
        
        if ylim:
            ax.set_ylim(ylim)
        if log_scale:
            ax.set_yscale("log")
    
    # Hide unused subplots
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add legend
    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=min(4, len(legend_elements)),
            fontsize=9,
            bbox_to_anchor=(0.5, 0.02),
        )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig
    return grouper


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare maxP vs baseline experiments
    python visualize.py \\
        --exp maxp=outputs/mlp_sweep \\
        --exp baseline=outputs/mlp_baseline \\
        --group-by model.hidden_dim model.n_layers \\
        --metric train_loss \\
        --smoothing 0.1 \\
        --output plots/comparison.png

    # Plot test accuracy with log scale
    python visualize.py \\
        --exp results=outputs/sweep \\
        --metric test_acc \\
        --select-by accuracy
        """,
    )
    
    parser.add_argument(
        "--exp",
        action="append",
        required=True,
        help="Experiment in format name=path (can be repeated)",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["model.hidden_dim", "model.n_layers"],
        help="Config keys to group runs by (default: model.hidden_dim model.n_layers)",
    )
    parser.add_argument(
        "--select-by",
        choices=["loss", "accuracy"],
        default="loss",
        help="How to select best run from each group",
    )
    parser.add_argument(
        "--metric",
        default="train_loss",
        help="Metric to plot (e.g., train_loss, train_acc, test_loss, test_acc, lr_0, alpha_0, omega_0, u_0)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="EMA smoothing alpha (lower = more smoothing)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        help="Y-axis limits",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for y-axis",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (if not set, displays plot)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Create single plot instead of grid",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI",
    )
    parser.add_argument(
        "--per-layer",
        choices=["lr", "alpha", "omega", "u"],
        default=None,
        help="Plot per-layer metrics (all layers in subplots). Overrides --metric.",
    )
    
    args = parser.parse_args()
    
    # Parse experiments
    experiments = {}
    for exp_str in args.exp:
        if "=" not in exp_str:
            raise ValueError(f"Invalid experiment format: {exp_str}. Use name=path")
        name, path = exp_str.split("=", 1)
        experiments[name] = path
    
    # Create grouper function
    grouper = create_grouper(args.group_by)
    
    # Select aggregation method
    if args.select_by == "loss":
        aggregator = MetricAggregator.best_by_final_loss
    else:
        aggregator = MetricAggregator.best_by_final_accuracy
    
    # Load and process experiments
    named_experiments = {}
    for name, path in experiments.items():
        print(f"Loading {name} from {path}...")
        collector = RunCollector(path)
        print(f"  Found {len(collector.runs)} runs")
        
        if len(collector.runs) == 0:
            print(f"  Warning: No valid runs found in {path}")
            continue
        
        grouped = collector.group_by(grouper)
        print(f"  Grouped into {len(grouped)} groups")
        
        aggregated = {key: aggregator(runs) for key, runs in grouped.items()}
        named_experiments[name] = aggregated
    
    if not named_experiments:
        print("No experiments loaded!")
        return
    
    # Create visualization
    viz = GridVisualizer()
    
    ylim = tuple(args.ylim) if args.ylim else None
    
    if args.per_layer:
        # Plot per-layer metrics (alpha, omega, u, lr)
        fig = plot_per_layer_metrics(
            named_experiments,
            metric_prefix=args.per_layer,
            title=args.title or f"Per-Layer {args.per_layer.title()}",
            smoothing_alpha=args.smoothing,
            ylim=ylim,
            log_scale=args.log_scale,
        )
    elif args.single:
        fig = viz.plot_single(
            named_experiments,
            metric=args.metric,
            title=args.title or args.metric.replace("_", " ").title(),
            smoothing_alpha=args.smoothing,
            ylim=ylim,
            log_scale=args.log_scale,
        )
    else:
        fig = viz.plot_grid(
            named_experiments,
            metric=args.metric,
            title=args.title or args.metric.replace("_", " ").title(),
            smoothing_alpha=args.smoothing,
            ylim=ylim,
            log_scale=args.log_scale,
        )
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    main()
