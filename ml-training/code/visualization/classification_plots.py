"""
Diagnostic Plotting Module for E-DAIC Classification Benchmark.

Generates:
1. ROC curves (with AUC and diagonal chance line).
2. Precision-Recall curves (with AP and prevalence baseline).
3. Raw and Normalized Confusion Matrices.
4. Metric comparison bar charts (across models, representations, pooling strategies).
5. Model-by-pooling performance heatmaps.

Saves clean PNG figures and underlying curve coordinates for full reproducibility.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_roc_curve(
    roc_data: Dict[str, Any],
    title: str,
    save_path: Path,
) -> None:
    """Plots and saves an ROC curve figure."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fpr = roc_data.get("fpr", [])
    tpr = roc_data.get("tpr", [])
    auc_val = roc_data.get("auc")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    if fpr and tpr:
        auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc_str})")
    
    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_pr_curve(
    pr_data: Dict[str, Any],
    title: str,
    save_path: Path,
    pos_prevalence: Optional[float] = None,
) -> None:
    """Plots and saves a Precision-Recall curve figure."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    precision = pr_data.get("precision", [])
    recall = pr_data.get("recall", [])
    ap_val = pr_data.get("ap")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    if precision and recall:
        ap_str = f"{ap_val:.3f}" if ap_val is not None else "N/A"
        ax.plot(recall, precision, color="#2ca02c", lw=2, label=f"PR (AP = {ap_str})")

    # Positive class prevalence line
    if pos_prevalence is not None:
        ax.axhline(y=pos_prevalence, color="gray", linestyle="--", lw=1.5, label=f"Prevalence ({pos_prevalence:.1%})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    title: str,
    save_path: Path,
    normalize: bool = False,
) -> None:
    """Plots and saves a Confusion Matrix (raw or row-normalized)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, where=row_sums != 0)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    cax = ax.matshow(cm_display, cmap=plt.cm.Blues, alpha=0.85)

    # Class labels
    classes = ["0 (Control)", "1 (Depressed)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # Annotate counts / percentages in each cell
    for i in range(2):
        for j in range(2):
            val = cm_display[i, j]
            text_str = f"{val:.1%}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color="white" if cm_display[i, j] > cm_display.max() / 2 else "black",
                fontsize=13, fontweight="bold",
            )

    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric_col: str,
    title: str,
    save_path: Path,
    top_n: int = 15,
) -> None:
    """Plots top models comparison bar chart for a given metric."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if metric_col not in results_df.columns:
        return

    # Sort descending
    sorted_df = results_df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False).head(top_n)
    if sorted_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    labels = [
        f"{row['model_name']} ({row['representation']} - {row['pooling_strategy']})"
        for _, row in sorted_df.iterrows()
    ]
    vals = sorted_df[metric_col].values

    bars = ax.barh(range(len(vals)), vals, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # Top model on top
    ax.set_xlabel(metric_col.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Annotate value on bar
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_model_pooling_heatmap(
    results_df: pd.DataFrame,
    metric_col: str,
    title: str,
    save_path: Path,
) -> None:
    """Plots a 2D heatmap of model performance across representations & pooling."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if metric_col not in results_df.columns or results_df.empty:
        return

    # Pivot: index=model_name, columns=representation+pooling
    df_copy = results_df.copy()
    df_copy["rep_pool"] = df_copy["representation"] + " (" + df_copy["pooling_strategy"] + ")"
    pivot = df_copy.pivot_table(index="model_name", columns="rep_pool", values=metric_col, aggfunc="mean")
    
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    cax = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.02, label=metric_col.replace("_", " ").title())

    # Text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black" if v < 0.6 else "white")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
