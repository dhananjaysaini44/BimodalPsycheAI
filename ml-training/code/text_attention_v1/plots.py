"""
Diagnostic Plotting Module for Phase 18C.

Generates 7 publication-quality diagnostic visualizations:
1. Classification performance: Attention vs. Fixed-Pooling Baselines
2. Regression CCC and RMSE comparison
3. CV fold performance variability
4. Seed-to-seed performance variability
5. Paired-bootstrap difference distributions and confidence intervals
6. Training and validation loss curves across folds and seeds
7. Example participant-level attention-weight profiles
"""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def plot_classification_benchmark(
    df_results: pd.DataFrame,
    output_path: Path,
):
    """Plot 1: Classification F1 and ROC-AUC comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cls_df = df_results[df_results["task"] == "classification"].copy()
    cls_df = cls_df.sort_values("dev_f1", ascending=False)

    y_pos = np.arange(len(cls_df))
    models = cls_df["model_label"].tolist()

    # Colors: attention vs baseline
    colors = ["#2b5c8f" if "Attention" in m else "#7f8c8d" for m in models]

    # Subplot 1: F1 Score
    bars1 = ax1.barh(y_pos, cls_df["dev_f1"], color=colors, alpha=0.85, edgecolor="black", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Development F1 Score (Macro Positive)", fontsize=11, fontweight="bold")
    ax1.set_title("Development F1 Score by Model", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 0.65)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    # Reference lines
    ax1.axvline(0.4516, color="#e74c3c", linestyle="--", alpha=0.7, label="Phase 18B Whole Median (0.4516)")
    ax1.axvline(0.5161, color="#27ae60", linestyle=":", alpha=0.7, label="Phase 18A Middle MiniLM (0.5161)")
    ax1.legend(loc="lower right", fontsize=9)

    # Subplot 2: ROC-AUC
    bars2 = ax2.barh(y_pos, cls_df["dev_roc_auc"], color=colors, alpha=0.85, edgecolor="black", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Development ROC-AUC", fontsize=11, fontweight="bold")
    ax2.set_title("Development ROC-AUC by Model", fontsize=12, fontweight="bold")
    ax2.set_xlim(0.4, 0.8)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_regression_benchmark(
    df_results: pd.DataFrame,
    output_path: Path,
):
    """Plot 2: Regression CCC and RMSE comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    reg_df = df_results[df_results["task"] == "regression"].copy()
    reg_df = reg_df.sort_values("dev_ccc", ascending=False)

    y_pos = np.arange(len(reg_df))
    models = reg_df["model_label"].tolist()
    colors = ["#c0392b" if "Attention" in m else "#7f8c8d" for m in models]

    # Subplot 1: Lin's CCC (Higher is better)
    bars1 = ax1.barh(y_pos, reg_df["dev_ccc"], color=colors, alpha=0.85, edgecolor="black", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Development Lin's CCC", fontsize=11, fontweight="bold")
    ax1.set_title("Development Lin's CCC by Model", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 0.55)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    ax1.axvline(0.4048, color="#2980b9", linestyle="--", alpha=0.7, label="Phase 18B Whole Mean+Std (0.4048)")
    ax1.axvline(0.4022, color="#27ae60", linestyle=":", alpha=0.7, label="Phase 18B Middle Mean (0.4022)")
    ax1.legend(loc="lower right", fontsize=9)

    # Subplot 2: RMSE (Lower is better)
    bars2 = ax2.barh(y_pos, reg_df["dev_rmse"], color=colors, alpha=0.85, edgecolor="black", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Development RMSE (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_title("Development RMSE by Model", fontsize=12, fontweight="bold")
    ax2.set_xlim(4.0, 7.5)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cv_fold_variability(
    fold_records: List[Dict[str, Any]],
    output_path: Path,
):
    """Plot 3: 5-Fold Cross-Validation Performance Variability."""
    df_folds = pd.DataFrame(fold_records)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Classification Folds
    cls_folds = df_folds[df_folds["task"] == "classification"]
    if len(cls_folds) > 0:
        sns.boxplot(data=cls_folds, x="arm", y="val_f1", hue="arm", palette="Blues", ax=ax1, width=0.4)
        sns.stripplot(data=cls_folds, x="arm", y="val_f1", color="black", size=7, jitter=0.15, ax=ax1)
        ax1.set_title("Classification Validation F1 across 5 Folds", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Validation F1 Score", fontsize=11)
        ax1.set_xlabel("Experimental Arm", fontsize=11)

    # Regression Folds
    reg_folds = df_folds[df_folds["task"] == "regression"]
    if len(reg_folds) > 0:
        sns.boxplot(data=reg_folds, x="arm", y="val_ccc", hue="arm", palette="Reds", ax=ax2, width=0.4)
        sns.stripplot(data=reg_folds, x="arm", y="val_ccc", color="black", size=7, jitter=0.15, ax=ax2)
        ax2.set_title("Regression Validation CCC across 5 Folds", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Validation Lin's CCC", fontsize=11)
        ax2.set_xlabel("Experimental Arm", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_seed_stability(
    seed_records: List[Dict[str, Any]],
    output_path: Path,
):
    """Plot 4: Performance stability across random seeds (42, 123, 456)."""
    df_seeds = pd.DataFrame(seed_records)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cls_seeds = df_seeds[df_seeds["task"] == "classification"]
    if len(cls_seeds) > 0:
        sns.barplot(data=cls_seeds, x="arm", y="dev_f1", hue="seed", palette="crest", ax=ax1)
        ax1.set_title("Classification Dev F1 by Random Seed", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Development F1", fontsize=11)
        ax1.set_ylim(0, 0.6)

    reg_seeds = df_seeds[df_seeds["task"] == "regression"]
    if len(reg_seeds) > 0:
        sns.barplot(data=reg_seeds, x="arm", y="dev_ccc", hue="seed", palette="flare", ax=ax2)
        ax2.set_title("Regression Dev CCC by Random Seed", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Development Lin's CCC", fontsize=11)
        ax2.set_ylim(0, 0.55)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paired_bootstrap_differences(
    df_diffs: pd.DataFrame,
    output_path: Path,
):
    """Plot 5: Forest plot of paired bootstrap differences (Attention - Fixed Baseline)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(df_diffs))
    comps = df_diffs["comparison"].tolist()
    diffs = df_diffs["point_diff"].to_numpy()
    lowers = df_diffs["ci95_lower"].to_numpy()
    uppers = df_diffs["ci95_upper"].to_numpy()

    # Errors for errorbar
    xerr = [diffs - lowers, uppers - diffs]

    colors = ["#27ae60" if (d > 0 and "RMSE" not in c) or (d < 0 and "RMSE" in c) else "#c0392b" for d, c in zip(diffs, comps)]

    for i, (d, y, err_low, err_up, col) in enumerate(zip(diffs, y_pos, diffs - lowers, uppers - diffs, colors)):
        ax.errorbar(d, y, xerr=[[err_low], [err_up]], fmt="o", color=col, ecolor=col, elinewidth=2.5, capsize=5, markersize=8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.75)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(comps, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Paired Difference: Attention - Fixed Baseline (with 95% Bootstrap CI)", fontsize=11, fontweight="bold")
    ax.set_title("Paired Participant-Level Bootstrap Difference Tests (B=2,000, N=56/54)", fontsize=12, fontweight="bold")

    # Add text annotations
    for i, (d, l, u) in enumerate(zip(diffs, lowers, uppers)):
        ax.text(u + 0.02, i, f"Δ={d:+.4f}\n[{l:+.4f}, {u:+.4f}]", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_training_loss_curves(
    history_curves: Dict[str, Any],
    output_path: Path,
):
    """Plot 6: Training and validation loss curves across folds and seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Arm B Classification
    ax = axes[0, 0]
    for key, data in history_curves.get("whole_attention_classification", {}).items():
        epochs = range(1, len(data["train_loss"]) + 1)
        ax.plot(epochs, data["train_loss"], alpha=0.3, color="#3498db")
        ax.plot(epochs, data["val_loss"], alpha=0.7, linestyle="--", label=f"{key}")
    ax.set_title("Whole Attention Classification Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")

    # Arm C Classification
    ax = axes[0, 1]
    for key, data in history_curves.get("middle_attention_classification", {}).items():
        epochs = range(1, len(data["train_loss"]) + 1)
        ax.plot(epochs, data["train_loss"], alpha=0.3, color="#2ecc71")
        ax.plot(epochs, data["val_loss"], alpha=0.7, linestyle="--", label=f"{key}")
    ax.set_title("Middle Attention Classification Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")

    # Arm B Regression
    ax = axes[1, 0]
    for key, data in history_curves.get("whole_attention_regression", {}).items():
        epochs = range(1, len(data["train_loss"]) + 1)
        ax.plot(epochs, data["train_loss"], alpha=0.3, color="#e67e22")
        ax.plot(epochs, data["val_loss"], alpha=0.7, linestyle="--", label=f"{key}")
    ax.set_title("Whole Attention Regression Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")

    # Arm C Regression
    ax = axes[1, 1]
    for key, data in history_curves.get("middle_attention_regression", {}).items():
        epochs = range(1, len(data["train_loss"]) + 1)
        ax.plot(epochs, data["train_loss"], alpha=0.3, color="#9b59b6")
        ax.plot(epochs, data["val_loss"], alpha=0.7, linestyle="--", label=f"{key}")
    ax.set_title("Middle Attention Regression Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_attention_weight_profiles(
    example_weights: List[Dict[str, Any]],
    output_path: Path,
):
    """Plot 7: Qualitative inspection of participant attention weights across chunks."""
    fig, axes = plt.subplots(len(example_weights), 1, figsize=(12, 2.5 * len(example_weights)), sharex=False)
    if len(example_weights) == 1:
        axes = [axes]

    for ax, ex in zip(axes, example_weights):
        pid = ex["participant_id"]
        status = "Depressed" if ex["phq_binary"] == 1 else "Non-Depressed"
        score = ex["phq_score"]
        weights = ex["attention_weights"] # List of weights
        k_chunks = len(weights)
        chunks = np.arange(k_chunks)

        bars = ax.bar(chunks, weights, color="#34495e", alpha=0.85, width=0.6, edgecolor="black")
        uniform_weight = 1.0 / k_chunks
        ax.axhline(uniform_weight, color="#e74c3c", linestyle="--", alpha=0.7, label=f"Uniform Weight ({uniform_weight:.3f})")

        ax.set_title(f"Participant {pid} ({status}, PHQ-8 = {score:.1f}, Chunks = {k_chunks})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Attention α_ij", fontsize=9)
        ax.set_xlabel("Chronological Chunk Index", fontsize=9)
        ax.set_ylim(0, max(max(weights) * 1.3, 0.2))
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
