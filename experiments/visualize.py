#!/usr/bin/env python3
"""
Training Visualization for Co-teaching EM Framework
====================================================
Real-time/offline visualization of key training metrics.

Features:
1. Training curves (loss, accuracy, q, overlap)
2. Multi-experiment comparison plots
3. Ablation study heatmaps
4. Sample-level q distribution visualization

Usage:
    # Single experiment visualization
    python experiments/visualize.py --log_dir logs/exp1
    
    # Multi-experiment comparison
    python experiments/visualize.py --compare logs/exp1 logs/exp2 logs/exp3
    
    # Generate paper-quality figures
    python experiments/visualize.py --paper --log_dir logs/exp1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Use English fonts only (no CJK fonts)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300

# Paper-quality style settings
PAPER_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 6,
}

# Color scheme (colorblind-friendly)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def load_training_log(log_path: Path) -> Dict:
    """加载训练日志"""
    log_file = log_path / "training_log.json"
    if not log_file.exists():
        # 尝试其他格式
        for ext in [".jsonl", ".log"]:
            alt = log_path / f"training_log{ext}"
            if alt.exists():
                log_file = alt
                break
    
    if not log_file.exists():
        raise FileNotFoundError(f"No training log found in {log_path}")
    
    with open(log_file, 'r') as f:
        if log_file.suffix == '.jsonl':
            data = [json.loads(line) for line in f]
            return {"epochs": data}
        else:
            return json.load(f)


def extract_metrics(log_data: Dict) -> Dict[str, np.ndarray]:
    """提取关键指标"""
    epochs = log_data.get("epochs", [])
    if not epochs:
        return {}
    
    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "val_acc": [],
        "q_mean": [],
        "q_std": [],
        "overlap": [],
        "pi_t": [],
        "active_models": [],
        "replay_size": [],
    }
    
    for ep in epochs:
        metrics["epoch"].append(ep.get("epoch", len(metrics["epoch"])))
        metrics["train_loss"].append(ep.get("train_loss", np.nan))
        metrics["train_acc"].append(ep.get("train_acc", np.nan))
        metrics["test_acc"].append(ep.get("test_acc", np.nan))
        metrics["val_acc"].append(ep.get("val_acc", ep.get("test_acc", np.nan)))
        metrics["q_mean"].append(ep.get("q_mean", np.nan))
        metrics["q_std"].append(ep.get("q_std", np.nan))
        metrics["overlap"].append(ep.get("overlap", np.nan))
        metrics["pi_t"].append(ep.get("pi_t", np.nan))
        metrics["active_models"].append(ep.get("active_models", np.nan))
        metrics["replay_size"].append(ep.get("replay_size", np.nan))
    
    return {k: np.array(v) for k, v in metrics.items()}


# =============================================================================
# Figure 1: Single Experiment Dashboard
# =============================================================================
def plot_dashboard(metrics: Dict[str, np.ndarray], title: str = "Training Dashboard",
                   save_path: Optional[Path] = None):
    """
    Comprehensive dashboard with 4 subplots:
    - (1,1) Accuracy curves
    - (1,2) Loss curve
    - (2,1) Q statistics
    - (2,2) Overlap & Active Models
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = metrics["epoch"]
    
    # --- Accuracy ---
    ax = axes[0, 0]
    ax.plot(epochs, metrics["train_acc"], label="Train Acc", color=COLORS[0], alpha=0.7)
    if not np.all(np.isnan(metrics["val_acc"])):
        ax.plot(epochs, metrics["val_acc"], label="Val Acc", color=COLORS[1], linestyle='--')
    ax.plot(epochs, metrics["test_acc"], label="Test Acc", color=COLORS[2], linewidth=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Annotate best test accuracy
    best_idx = np.nanargmax(metrics["test_acc"])
    best_acc = metrics["test_acc"][best_idx]
    ax.annotate(f'Best: {best_acc:.1f}%', 
                xy=(epochs[best_idx], best_acc),
                xytext=(epochs[best_idx] + 5, best_acc - 5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)
    
    # --- Loss ---
    ax = axes[0, 1]
    ax.plot(epochs, metrics["train_loss"], label="Train Loss", color=COLORS[0])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Q Statistics ---
    ax = axes[1, 0]
    q_mean = metrics["q_mean"]
    q_std = metrics["q_std"]
    ax.plot(epochs, q_mean, label="Q Mean", color=COLORS[3], linewidth=2)
    ax.fill_between(epochs, q_mean - q_std, q_mean + q_std, 
                    alpha=0.3, color=COLORS[3], label="Q ± Std")
    if not np.all(np.isnan(metrics["pi_t"])):
        ax.plot(epochs, metrics["pi_t"], label="π_t (clean prior)", 
                color=COLORS[4], linestyle='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Q (Responsibility) & π_t (Clean Prior)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # --- Overlap & Active Models ---
    ax = axes[1, 1]
    ax.plot(epochs, metrics["overlap"], label="Selection Overlap", color=COLORS[5])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overlap Ratio", color=COLORS[5])
    ax.tick_params(axis='y', labelcolor=COLORS[5])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Right Y-axis for active models
    if not np.all(np.isnan(metrics["active_models"])):
        ax2 = ax.twinx()
        ax2.plot(epochs, metrics["active_models"], label="Active Models", 
                 color=COLORS[6], linestyle='--', marker='o', markersize=3)
        ax2.set_ylabel("# Active Models", color=COLORS[6])
        ax2.tick_params(axis='y', labelcolor=COLORS[6])
    
    ax.set_title("Selection Overlap & Committee Size")
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# Figure 2: Multi-Experiment Comparison
# =============================================================================
def plot_comparison(experiments: Dict[str, Dict[str, np.ndarray]], 
                    metric: str = "test_acc",
                    title: str = "Experiment Comparison",
                    save_path: Optional[Path] = None):
    """
    Compare single metric across multiple experiments.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (name, metrics) in enumerate(experiments.items()):
        epochs = metrics["epoch"]
        values = metrics.get(metric, np.full_like(epochs, np.nan, dtype=float))
        
        ax.plot(epochs, values, label=name, color=COLORS[i % len(COLORS)], linewidth=2)
        
        # Annotate final value
        if not np.isnan(values[-1]):
            ax.annotate(f'{values[-1]:.1f}', 
                       xy=(epochs[-1], values[-1]),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, color=COLORS[i % len(COLORS)])
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# Figure 3: Ablation Study Heatmap
# =============================================================================
def plot_ablation_heatmap(results: Dict[str, float], 
                          row_labels: List[str], 
                          col_labels: List[str],
                          title: str = "Ablation Study Results",
                          save_path: Optional[Path] = None):
    """
    Ablation study results heatmap.
    results: {(row, col): value} or 2D array
    """
    n_rows, n_cols = len(row_labels), len(col_labels)
    
    # Build matrix
    if isinstance(results, dict):
        matrix = np.full((n_rows, n_cols), np.nan)
        for (r, c), v in results.items():
            ri = row_labels.index(r) if r in row_labels else -1
            ci = col_labels.index(c) if c in col_labels else -1
            if ri >= 0 and ci >= 0:
                matrix[ri, ci] = v
    else:
        matrix = np.array(results)
    
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.5), max(6, n_rows * 0.8)))
    
    # Heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=11, fontweight='bold')
    
    # Labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Test Accuracy (%)", rotation=-90, va="bottom")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# Figure 4: Q Distribution Evolution
# =============================================================================
def plot_q_evolution(q_history: List[np.ndarray], 
                     epochs: List[int] = None,
                     title: str = "Q Distribution Evolution",
                     save_path: Optional[Path] = None):
    """
    Q value distribution evolution over epochs (box plot and histogram).
    q_history: List of q value arrays for each epoch
    """
    if epochs is None:
        epochs = list(range(len(q_history)))
    
    # Select epochs to display (evenly spaced sampling)
    n_show = min(10, len(epochs))
    indices = np.linspace(0, len(epochs) - 1, n_show, dtype=int)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # --- Top: Box Plot ---
    ax = axes[0]
    positions = list(range(n_show))
    data_to_plot = [q_history[i] for i in indices]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS[0])
        patch.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f"E{epochs[i]}" for i in indices])
    ax.set_ylabel("Q Value")
    ax.set_title("Q Distribution Evolution (Box Plot)")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Bottom: Density Plot ---
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    
    for idx, i in enumerate(indices):
        q_vals = q_history[i]
        hist, _ = np.histogram(q_vals, bins=bins, density=True)
        ax.plot(bins[:-1] + 0.025, hist, label=f"Epoch {epochs[i]}", 
                color=plt.cm.viridis(idx / n_show), linewidth=1.5)
    
    ax.set_xlabel("Q Value")
    ax.set_ylabel("Density")
    ax.set_title("Q Distribution Evolution (Density)")
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# Figure 5: Paper-Quality Composite Figure
# =============================================================================
def plot_paper_figure(experiments: Dict[str, Dict[str, np.ndarray]],
                      title: str = "Co-teaching EM Framework Results",
                      save_path: Optional[Path] = None):
    """
    Paper-quality composite figure:
    (a) Test Accuracy comparison
    (b) Q Mean evolution
    (c) Overlap changes
    (d) Final accuracy bar chart
    """
    with plt.rc_context(PAPER_STYLE):
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # (a) Test Accuracy
        ax = fig.add_subplot(gs[0, 0])
        for i, (name, metrics) in enumerate(experiments.items()):
            ax.plot(metrics["epoch"], metrics["test_acc"], 
                   label=name, color=COLORS[i % len(COLORS)], linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("(a) Test Accuracy")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # (b) Q Mean
        ax = fig.add_subplot(gs[0, 1])
        for i, (name, metrics) in enumerate(experiments.items()):
            ax.plot(metrics["epoch"], metrics["q_mean"],
                   label=name, color=COLORS[i % len(COLORS)], linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Q Mean")
        ax.set_title("(b) Mean Responsibility Q")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # (c) Overlap
        ax = fig.add_subplot(gs[1, 0])
        for i, (name, metrics) in enumerate(experiments.items()):
            ax.plot(metrics["epoch"], metrics["overlap"],
                   label=name, color=COLORS[i % len(COLORS)], linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Selection Overlap")
        ax.set_title("(c) Selection Overlap")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # (d) Final Accuracy Bar Chart
        ax = fig.add_subplot(gs[1, 1])
        names = list(experiments.keys())
        final_accs = [experiments[n]["test_acc"][-1] for n in names]
        x = np.arange(len(names))
        bars = ax.bar(x, final_accs, color=[COLORS[i % len(COLORS)] for i in range(len(names))])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel("Final Test Accuracy (%)")
        ax.set_title("(d) Final Performance Comparison")
        ax.set_ylim([0, 100])
        
        # 添加数值标签
        for bar, acc in zip(bars, final_accs):
            ax.annotate(f'{acc:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig


# =============================================================================
# Helper: Generate Demo Data for Testing Visualization
# =============================================================================
def generate_demo_data(n_epochs: int = 100, noise_rate: float = 0.5) -> Dict[str, np.ndarray]:
    """Generate simulated training data for testing visualization."""
    epochs = np.arange(n_epochs)
    
    # Simulate accuracy curves
    train_acc = 100 * (1 - np.exp(-epochs / 20)) * (1 - noise_rate * 0.3) + np.random.randn(n_epochs) * 2
    test_acc = train_acc * 0.9 - noise_rate * 10 + np.random.randn(n_epochs) * 1.5
    test_acc = np.clip(test_acc, 0, 100)
    
    # Simulate loss curve
    train_loss = 2.5 * np.exp(-epochs / 30) + 0.3 + np.random.randn(n_epochs) * 0.05
    
    # Simulate Q statistics
    q_mean = 0.5 + 0.4 * (1 - np.exp(-epochs / 25)) - noise_rate * 0.2
    q_mean = np.clip(q_mean, 0, 1)
    q_std = 0.2 * np.exp(-epochs / 50) + 0.05
    
    # Simulate overlap
    overlap = 0.3 + 0.5 * (1 - np.exp(-epochs / 15)) + np.random.randn(n_epochs) * 0.05
    overlap = np.clip(overlap, 0, 1)
    
    return {
        "epoch": epochs,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "val_acc": test_acc * 0.98,
        "train_loss": train_loss,
        "q_mean": q_mean,
        "q_std": q_std,
        "overlap": overlap,
        "pi_t": 1 - noise_rate + 0.1 * np.sin(epochs / 10),
        "active_models": np.full(n_epochs, 3),
        "replay_size": np.minimum(epochs * 50, 5000),
    }


def main():
    parser = argparse.ArgumentParser(description="Training Visualization")
    parser.add_argument("--log_dir", type=str, help="Path to training log directory")
    parser.add_argument("--compare", nargs='+', help="Multiple log directories to compare")
    parser.add_argument("--paper", action="store_true", help="Generate paper-quality figures")
    parser.add_argument("--demo", action="store_true", help="Run with demo data")
    parser.add_argument("--save_dir", type=str, default="figures", help="Directory to save figures")
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if args.demo:
        print("Generating demo visualizations...")
        
        # Generate demo data with different configurations
        experiments = {
            "Baseline": generate_demo_data(100, 0.5),
            "EM-Posterior": generate_demo_data(100, 0.4),
            "Full System": generate_demo_data(100, 0.3),
        }
        
        # Dashboard
        plot_dashboard(experiments["Full System"], 
                      title="Full System Training Dashboard",
                      save_path=save_dir / "dashboard_demo.png")
        
        # Comparison
        plot_comparison(experiments, metric="test_acc",
                       title="Method Comparison (Test Accuracy)",
                       save_path=save_dir / "comparison_demo.png")
        
        # Ablation heatmap
        ablation_results = {
            ("γ=0.0", "sym-20%"): 82.3, ("γ=0.0", "sym-50%"): 65.1, ("γ=0.0", "asym-40%"): 68.4,
            ("γ=0.3", "sym-20%"): 84.5, ("γ=0.3", "sym-50%"): 69.8, ("γ=0.3", "asym-40%"): 71.2,
            ("γ=0.5", "sym-20%"): 85.2, ("γ=0.5", "sym-50%"): 72.4, ("γ=0.5", "asym-40%"): 73.8,
            ("γ=0.7", "sym-20%"): 84.8, ("γ=0.7", "sym-50%"): 71.6, ("γ=0.7", "asym-40%"): 72.5,
            ("γ=1.0", "sym-20%"): 83.1, ("γ=1.0", "sym-50%"): 68.9, ("γ=1.0", "asym-40%"): 70.1,
        }
        plot_ablation_heatmap(ablation_results,
                             row_labels=["γ=0.0", "γ=0.3", "γ=0.5", "γ=0.7", "γ=1.0"],
                             col_labels=["sym-20%", "sym-50%", "asym-40%"],
                             title="Q-Gamma Ablation Study",
                             save_path=save_dir / "ablation_heatmap_demo.png")
        
        # Q evolution (simulated)
        q_history = [np.random.beta(2 + i/10, 5 - i/30, 1000) for i in range(100)]
        plot_q_evolution(q_history, 
                        title="Q Distribution Evolution (Demo)",
                        save_path=save_dir / "q_evolution_demo.png")
        
        # Paper figure
        if args.paper:
            plot_paper_figure(experiments,
                            title="Co-teaching EM Framework (Demo)",
                            save_path=save_dir / "paper_figure_demo.pdf")
        
        print(f"\nAll demo figures saved to {save_dir}/")
        return
    
    # Load actual data
    if args.compare:
        experiments = {}
        for log_path in args.compare:
            name = Path(log_path).name
            try:
                log_data = load_training_log(Path(log_path))
                experiments[name] = extract_metrics(log_data)
            except Exception as e:
                print(f"Warning: Could not load {log_path}: {e}")
        
        if experiments:
            plot_comparison(experiments, metric="test_acc",
                           save_path=save_dir / "comparison.png")
            if args.paper:
                plot_paper_figure(experiments, save_path=save_dir / "paper_figure.pdf")
    
    elif args.log_dir:
        log_data = load_training_log(Path(args.log_dir))
        metrics = extract_metrics(log_data)
        plot_dashboard(metrics, 
                      title=f"Training: {Path(args.log_dir).name}",
                      save_path=save_dir / "dashboard.png")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python experiments/visualize.py --demo              # Test with demo data")
        print("  python experiments/visualize.py --log_dir logs/exp1 # Visualize single experiment")
        print("  python experiments/visualize.py --compare logs/exp1 logs/exp2")


if __name__ == "__main__":
    main()
