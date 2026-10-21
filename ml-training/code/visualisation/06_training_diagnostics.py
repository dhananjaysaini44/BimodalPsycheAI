"""
Phase 15 - Script 06: Training Diagnostics and Overfitting Analysis.

Analyzes iterative training logs from:
1. Phase 13A: CNN MLP Prediction Heads (8 models, CSV histories)
2. Phase 13B: CNN Temporal GRU+Attention (4 models, JSON histories)

Generates:
- Individual 3-panel training curves (Loss, Validation Metric, Overfitting Gap)
- Cross-model dev loss comparisons
- Overfitting gap summary bar chart

All figures saved exclusively in PNG format to:
outputs/model_visualizations_v1/training_diagnostics/
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("E:/Projects/BimodalPsycheAI")
ML_TRAINING = PROJECT_ROOT / "ml-training"
OUTPUT_DIR = ML_TRAINING / "outputs" / "model_visualizations_v1"
DIAG_DIR = OUTPUT_DIR / "training_diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Global styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_neural_head_history(csv_path):
    """Processes Phase 13A CSV history."""
    df = pd.read_csv(csv_path)
    stem = csv_path.stem.replace("__history", "")
    is_class = "class" in stem
    task_name = "Classification" if is_class else "Regression"

    epochs = df["epoch"].to_numpy()
    train_loss = df["train_loss"].to_numpy()
    dev_loss = df["dev_loss"].to_numpy()
    loss_gap = dev_loss - train_loss

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # 1. Loss Curves
    ax1.plot(epochs, train_loss, "b-", lw=2, label="Train Loss")
    ax1.plot(epochs, dev_loss, "r-", lw=2, label="Dev Loss")
    best_epoch_idx = int(np.argmin(dev_loss))
    ax1.scatter([epochs[best_epoch_idx]], [dev_loss[best_epoch_idx]], color="black", s=80, zorder=5, label=f"Min Dev Loss (Ep {epochs[best_epoch_idx]})")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax1.set_title("A. Loss Trajectory", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Metric Trajectory
    if is_class:
        if "dev_roc_auc" in df.columns:
            ax2.plot(epochs, df["dev_roc_auc"], color="#2b5c8f", lw=2, label="Dev ROC-AUC")
        if "dev_f1" in df.columns:
            ax2.plot(epochs, df["dev_f1"], color="#d95f02", lw=2, label="Dev F1")
        if "dev_balanced_accuracy" in df.columns:
            ax2.plot(epochs, df["dev_balanced_accuracy"], color="#1b9e77", lw=1.5, linestyle="--", label="Dev Bal Acc")
        ax2.set_ylabel("Score", fontsize=10, fontweight="bold")
        ax2.set_title("B. Validation Classification Metrics", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 1.0)
    else:
        if "dev_ccc" in df.columns:
            ax2.plot(epochs, df["dev_ccc"], color="#7570b3", lw=2, label="Dev CCC")
        if "dev_mae" in df.columns:
            ax2.plot(epochs, df["dev_mae"], color="#d95f02", lw=2, label="Dev MAE")
        if "dev_r2" in df.columns:
            ax2.plot(epochs, df["dev_r2"], color="#e7298a", lw=1.5, linestyle="--", label="Dev R²")
        ax2.set_ylabel("Value", fontsize=10, fontweight="bold")
        ax2.set_title("B. Validation Regression Metrics", fontsize=11, fontweight="bold")

    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.legend(loc="best", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Overfitting Gap
    ax3.plot(epochs, loss_gap, color="#e41a1c", lw=2, label="Generalization Gap (Dev - Train)")
    ax3.axhline(0, color="black", linestyle="--", lw=1)
    ax3.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Loss Delta", fontsize=10, fontweight="bold")
    ax3.set_title("C. Overfitting Diagnostic (Gap)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, linestyle=":", alpha=0.5)

    clean_name = stem.replace("__", " ").title()
    plt.suptitle(f"Training History: Neural Prediction Head — {clean_name} ({task_name})",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_p = DIAG_DIR / f"neural_head__{stem}__history.png"
    plt.savefig(out_p)
    plt.close(fig)

    return {
        "model": f"MLP {stem}",
        "suite": "Phase 13A (MLP)",
        "best_epoch": int(epochs[best_epoch_idx]),
        "min_dev_loss": float(dev_loss[best_epoch_idx]),
        "final_gap": float(loss_gap[-1]),
        "max_gap": float(np.max(loss_gap)),
    }


def plot_temporal_history(json_path):
    """Processes Phase 13B JSON history."""
    with open(json_path) as fp:
        data = json.load(fp)

    stem = json_path.stem.replace("__history", "")
    is_class = "classification" in stem
    task_name = "Classification" if is_class else "Regression"

    train_loss = np.array(data["train_loss"])
    dev_loss = np.array(data["dev_loss"])
    primary_metric = np.array(data["dev_primary_metric"])
    epochs = np.arange(1, len(train_loss) + 1)
    loss_gap = dev_loss - train_loss

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # 1. Loss Curves
    ax1.plot(epochs, train_loss, "b-", lw=2, label="Train Loss")
    ax1.plot(epochs, dev_loss, "r-", lw=2, label="Dev Loss")
    best_epoch_idx = int(np.argmin(dev_loss))
    ax1.scatter([epochs[best_epoch_idx]], [dev_loss[best_epoch_idx]], color="black", s=80, zorder=5, label=f"Min Dev Loss (Ep {epochs[best_epoch_idx]})")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax1.set_title("A. Loss Trajectory", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Metric Trajectory
    metric_name = "Dev ROC-AUC" if is_class else "Dev CCC"
    ax2.plot(epochs, primary_metric, color="#7570b3", lw=2, label=metric_name)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel(metric_name, fontsize=10, fontweight="bold")
    ax2.set_title(f"B. Primary Validation Metric ({metric_name})", fontsize=11, fontweight="bold")
    ax2.legend(loc="best", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Overfitting Gap
    ax3.plot(epochs, loss_gap, color="#e41a1c", lw=2, label="Generalization Gap (Dev - Train)")
    ax3.axhline(0, color="black", linestyle="--", lw=1)
    ax3.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Loss Delta", fontsize=10, fontweight="bold")
    ax3.set_title("C. Overfitting Diagnostic (Gap)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, linestyle=":", alpha=0.5)

    clean_name = stem.replace("__", " ").title()
    plt.suptitle(f"Training History: Temporal GRU+Attention — {clean_name} ({task_name})",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_p = DIAG_DIR / f"temporal__{stem}__history.png"
    plt.savefig(out_p)
    plt.close(fig)

    return {
        "model": f"Temporal {stem}",
        "suite": "Phase 13B (Temporal)",
        "best_epoch": int(epochs[best_epoch_idx]),
        "min_dev_loss": float(dev_loss[best_epoch_idx]),
        "final_gap": float(loss_gap[-1]),
        "max_gap": float(np.max(loss_gap)),
    }


def main():
    print("=" * 80)
    print("PHASE 15: TRAINING DIAGNOSTICS & OVERFITTING ANALYSIS")
    print("=" * 80)

    summary_records = []

    # 1. Phase 13A Neural Heads
    p13a_dir = ML_TRAINING / "outputs" / "cnn_embedding_neural_head_v1" / "histories"
    if p13a_dir.exists():
        for csv_f in sorted(p13a_dir.glob("*.csv")):
            rec = plot_neural_head_history(csv_f)
            summary_records.append(rec)
            print(f"Processed Phase 13A: {csv_f.name}")

    # 2. Phase 13B Temporal Models
    p13b_dir = ML_TRAINING / "outputs" / "cnn_embedding_temporal_v1" / "histories"
    if p13b_dir.exists():
        for json_f in sorted(p13b_dir.glob("*.json")):
            rec = plot_temporal_history(json_f)
            summary_records.append(rec)
            print(f"Processed Phase 13B: {json_f.name}")

    # 3. Overfitting Gap Summary
    if summary_records:
        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(DIAG_DIR / "overfitting_diagnostics_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
        x = np.arange(len(df_sum))
        w = 0.35
        ax.bar(x - w/2, df_sum["final_gap"], width=w, label="Final Epoch Gap", color="#2b5c8f", edgecolor="black", lw=0.6)
        ax.bar(x + w/2, df_sum["max_gap"], width=w, label="Maximum Observed Gap", color="#d95f02", edgecolor="black", lw=0.6)
        ax.axhline(0, color="black", lw=0.8)

        ax.set_xticks(x)
        labels = [m.replace("MLP ", "").replace("Temporal ", "").replace("__", "\n") for m in df_sum["model"]]
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Loss Generalization Gap (Dev - Train)", fontsize=10, fontweight="bold")
        ax.set_title("Overfitting Generalization Gap Across Neural Architectures (Dev vs. Train Loss)",
                     fontsize=11.5, fontweight="bold", pad=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(DIAG_DIR / "overfitting_gap_summary.png")
        plt.close(fig)
        print("Training diagnostics and overfitting summary completed.")


if __name__ == "__main__":
    main()
