"""
Phase 20: Unified Visualization & Benchmark Generation Pipeline.
Generates all publication-grade PNG figures for:
- Text pipeline (Classification, Regression, Seed stability, Attention diagnostics, Pipeline progression)
- Voice/Audio pipeline (Classification, Regression, Confusion matrices, Actual-vs-Pred, Feature comparisons)
- Cross-modality comparisons (Audio vs Text)

Outputs are saved strictly in PNG format at 200 DPI into both:
1. ml-training/visualizations/ (Committed GitHub publication bundle)
2. ml-training/outputs/visualizations/ (Local research archive)
"""

import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Setup repository paths
REPO_ROOT = Path(r"E:\Projects\BimodalPsycheAI")
ML_ROOT = REPO_ROOT / "ml-training"
OUTPUTS_DIR = ML_ROOT / "outputs"

PUB_VIS_DIR = ML_ROOT / "visualizations"
OUT_VIS_DIR = OUTPUTS_DIR / "visualizations"

for base in [PUB_VIS_DIR, OUT_VIS_DIR]:
    (base / "text").mkdir(parents=True, exist_ok=True)
    (base / "audio").mkdir(parents=True, exist_ok=True)
    (base / "comparison").mkdir(parents=True, exist_ok=True)

# Plot styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
DANGER_COLOR = "#d62728"
PURPLE_COLOR = "#9467bd"
BROWN_COLOR = "#8c564b"
TEAL_COLOR = "#17becf"

def save_fig(fig, rel_path):
    """Saves figure to both publication and output directories in PNG format."""
    p1 = PUB_VIS_DIR / rel_path
    p2 = OUT_VIS_DIR / rel_path
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {rel_path}")

# ==============================================================================
# 1. Text Classification Visualizations (Stage D)
# ==============================================================================

def generate_text_classification_plots():
    print("--- Generating Text Classification Visualizations ---")
    
    # 1.1 F1 Comparison
    models = [
        "Char TF-IDF\n(LinearSVC, Cal.)",
        "MiniLM L6\n(Middle Mean)",
        "DistilBERT\n(Concat Mean, n=54)",
        "DistilBERT\n(Whole Median)",
        "DistilBERT\n(Middle Attn Seed 456)",
        "DistilBERT\n(Whole Attn Seed 42)",
        "Word TF-IDF\n(LinearSVC, Cal.)",
        "MiniLM L6\n(Whole Mean)",
        "DistilBERT\n(Whole Attn 3-Seed Mean)"
    ]
    f1_scores = [0.4571, 0.5161, 0.4615, 0.4516, 0.5000, 0.4348, 0.4167, 0.4000, 0.4006]
    cohorts = ["Dev (n=56)", "Dev (n=56)", "Dev (n=54)", "Dev (n=56)", "Dev (n=56)", "Dev (n=56)", "Dev (n=56)", "Dev (n=56)", "Dev (n=56)"]
    colors = [PRIMARY_COLOR, SUCCESS_COLOR, PURPLE_COLOR, TEAL_COLOR, SECONDARY_COLOR, SECONDARY_COLOR, "#aec7e8", "#98df8a", "#ffbb78"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, f1_scores, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_ylabel("Development F1-Score (Depressed Class)")
    ax.set_title("Text Pipeline: Classification F1-Score Comparison Across Architectures")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.012, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "text/text_classification_f1_comparison.png")

    # 1.2 ROC-AUC Comparison
    auc_models = [
        "Char TF-IDF (LinearSVC)",
        "DistilBERT (Whole Attn Seed 42)",
        "DistilBERT (Whole Attn 3-Seed Mean)",
        "MiniLM L6 (Middle Mean)",
        "Combined TF-IDF (LinearSVC)",
        "DistilBERT (Whole Median)",
        "DistilBERT (Middle Attn Seed 456)",
        "MiniLM L6 (Mean_Std)",
        "DistilBERT (Concat Mean, n=54)"
    ]
    auc_scores = [0.7159, 0.7121, 0.7146, 0.6894, 0.6875, 0.6591, 0.6572, 0.6534, 0.6210]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(auc_models[::-1], auc_scores[::-1], color=PRIMARY_COLOR, edgecolor="black", linewidth=0.8, height=0.6)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random Guess (0.50)")
    ax.set_xlabel("Development ROC-AUC Score")
    ax.set_title("Text Pipeline: ROC-AUC Discrimination Comparison")
    ax.set_xlim(0.4, 0.8)
    ax.grid(axis="x")
    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + 0.005, bar.get_y() + bar.get_height() / 2.0, f"{xval:.4f}", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right")
    save_fig(fig, "text/text_classification_auc_comparison.png")

    # 1.3 Balanced Accuracy Comparison
    bal_models = [
        "MiniLM L6 (Middle Mean)",
        "DistilBERT (Middle Attn Seed 456)",
        "Char TF-IDF (LinearSVC)",
        "DistilBERT (Concat Mean, n=54)",
        "DistilBERT (Whole Median)",
        "Combined TF-IDF (LinearSVC)",
        "DistilBERT (Whole Attn Seed 42)",
        "Word TF-IDF (LinearSVC)",
        "MiniLM L6 (Mean_Std)"
    ]
    bal_scores = [0.7083, 0.6818, 0.6629, 0.6607, 0.6553, 0.6515, 0.6439, 0.6288, 0.6136]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(bal_models, bal_scores, color=TEAL_COLOR, edgecolor="black", linewidth=0.8, width=0.6)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance Level (0.50)")
    ax.set_ylabel("Development Balanced Accuracy")
    ax.set_title("Text Pipeline: Balanced Accuracy Comparison")
    ax.set_ylim(0.45, 0.80)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    ax.legend(loc="upper right")
    save_fig(fig, "text/text_classification_balanced_accuracy.png")

    # 1.4 PR-AUC Comparison
    pr_models = [
        "Char TF-IDF (LinearSVC)",
        "DistilBERT (Whole Attn Seed 42)",
        "DistilBERT (Whole Attn Seed 456)",
        "DistilBERT (Whole Median)",
        "Combined TF-IDF (LinearSVC)",
        "Word TF-IDF (LinearSVC)",
        "DistilBERT (Concat Mean, n=54)"
    ]
    pr_scores = [0.4682, 0.4375, 0.4589, 0.3812, 0.4125, 0.3647, 0.3582]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(pr_models, pr_scores, color=PURPLE_COLOR, edgecolor="black", linewidth=0.8, width=0.55)
    ax.axhline(12/56, color="red", linestyle="--", linewidth=1.2, label=f"No-Skill Baseline ({12/56:.3f})")
    ax.set_ylabel("Development Average Precision (PR-AUC)")
    ax.set_title("Text Pipeline: Precision-Recall AUC (Average Precision) Comparison")
    ax.set_ylim(0.15, 0.55)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    ax.legend(loc="upper right")
    save_fig(fig, "text/text_classification_pr_auc_comparison.png")

    # 1.5 Confusion Matrices Panel (4 models)
    p_char = OUTPUTS_DIR / "text_baselines_final_v1/development/dev_predictions_classification__char_tfidf__linear_svc__calibrated.csv"
    p_minilm = OUTPUTS_DIR / "text_segments_v1/predictions/dev_cls__whole__mean__linear_svc.csv"
    p_dist_med = OUTPUTS_DIR / "text_distilbert_v1/predictions/dev_cls__whole__median__linear_svc.csv"
    p_att42 = OUTPUTS_DIR / "text_attention_v1/predictions/dev_cla__whole_attention__seed42.csv"

    cm_data = []
    if p_char.exists():
        d = pd.read_csv(p_char)
        cm_data.append(("Char TF-IDF LinearSVC", confusion_matrix(d["y_true"], d["y_pred_calibrated"])))
    if p_minilm.exists():
        d = pd.read_csv(p_minilm)
        cm_data.append(("MiniLM Whole Mean LinearSVC", confusion_matrix(d["y_true"], d["y_pred"])))
    if p_dist_med.exists():
        d = pd.read_csv(p_dist_med)
        cm_data.append(("DistilBERT Whole Median LinearSVC", confusion_matrix(d["y_true"], d["y_pred"])))
    if p_att42.exists():
        d = pd.read_csv(p_att42)
        cm_data.append(("DistilBERT Whole Attention (Seed 42)", confusion_matrix(d["y_true"], d["y_pred"])))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, cm) in zip(axes, cm_data):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Non-Dep", "Depressed"], yticklabels=["Non-Dep", "Depressed"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    fig.suptitle("Text Pipeline: Development Split Confusion Matrices (n=56, Positives=12)", y=1.05, fontsize=14, fontweight="bold")
    save_fig(fig, "text/text_classification_confusion_matrices.png")

    # 1.6 ROC Curves Overlay
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if p_char.exists():
        d = pd.read_csv(p_char)
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_score"])
        ax.plot(fpr, tpr, color=PRIMARY_COLOR, lw=2, label=f"Char TF-IDF (AUC = {roc_auc_score(d['y_true'], d['y_score']):.4f})")
    if p_dist_med.exists():
        d = pd.read_csv(p_dist_med)
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_score"])
        ax.plot(fpr, tpr, color=TEAL_COLOR, lw=2, label=f"DistilBERT Whole Median (AUC = {roc_auc_score(d['y_true'], d['y_score']):.4f})")
    if p_att42.exists():
        d = pd.read_csv(p_att42)
        fpr, tpr, _ = roc_curve(d["y_true"], d["decision_score"])
        ax.plot(fpr, tpr, color=SECONDARY_COLOR, lw=2, label=f"DistilBERT Whole Attn Seed 42 (AUC = {roc_auc_score(d['y_true'], d['decision_score']):.4f})")
    
    p_att456 = OUTPUTS_DIR / "text_attention_v1/predictions/dev_cla__middle_attention__seed456.csv"
    if p_att456.exists():
        d = pd.read_csv(p_att456)
        fpr, tpr, _ = roc_curve(d["y_true"], d["decision_score"])
        ax.plot(fpr, tpr, color=SUCCESS_COLOR, lw=2, label=f"DistilBERT Middle Attn Seed 456 (AUC = {roc_auc_score(d['y_true'], d['decision_score']):.4f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Text Pipeline: Receiver Operating Characteristic (ROC) Curves (Dev, n=56)")
    ax.legend(loc="lower right")
    ax.grid(True)
    save_fig(fig, "text/text_classification_roc_curves.png")

    # 1.7 Precision-Recall Curves Overlay
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if p_char.exists():
        d = pd.read_csv(p_char)
        p, r, _ = precision_recall_curve(d["y_true"], d["y_score"])
        ax.plot(r, p, color=PRIMARY_COLOR, lw=2, label=f"Char TF-IDF (AP = {average_precision_score(d['y_true'], d['y_score']):.4f})")
    if p_dist_med.exists():
        d = pd.read_csv(p_dist_med)
        p, r, _ = precision_recall_curve(d["y_true"], d["y_score"])
        ax.plot(r, p, color=TEAL_COLOR, lw=2, label=f"DistilBERT Whole Median (AP = {average_precision_score(d['y_true'], d['y_score']):.4f})")
    if p_att42.exists():
        d = pd.read_csv(p_att42)
        p, r, _ = precision_recall_curve(d["y_true"], d["decision_score"])
        ax.plot(r, p, color=SECONDARY_COLOR, lw=2, label=f"DistilBERT Whole Attn Seed 42 (AP = {average_precision_score(d['y_true'], d['decision_score']):.4f})")
    
    ax.axhline(12/56, color="red", linestyle="--", lw=1.2, label=f"Prevalence Baseline ({12/56:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Text Pipeline: Precision-Recall Curves (Dev, n=56)")
    ax.legend(loc="upper right")
    ax.grid(True)
    save_fig(fig, "text/text_classification_precision_recall_curves.png")

    # 1.8 Seed Stability in Attention Models
    seeds = ["Seed 42", "Seed 123", "Seed 456"]
    whole_f1 = [0.4348, 0.3750, 0.3922]
    whole_auc = [0.7121, 0.6970, 0.7348]
    middle_f1 = [0.4211, 0.3396, 0.5000]
    middle_auc = [0.6420, 0.6250, 0.6572]

    x = np.arange(len(seeds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - width*1.5, whole_f1, width, label="Whole Attn F1", color="#1f77b4", edgecolor="black")
    ax.bar(x - width*0.5, whole_auc, width, label="Whole Attn ROC-AUC", color="#aec7e8", edgecolor="black")
    ax.bar(x + width*0.5, middle_f1, width, label="Middle Attn F1", color="#ff7f0e", edgecolor="black")
    ax.bar(x + width*1.5, middle_auc, width, label="Middle Attn ROC-AUC", color="#ffbb78", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylabel("Metric Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Attention Pooling: Classification Stability Across Random Seeds (Dev, n=56)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y")
    save_fig(fig, "text/text_attention_seed_stability.png")

# ==============================================================================
# 2. Text Regression Visualizations (Stage D)
# ==============================================================================

def generate_text_regression_plots():
    print("--- Generating Text Regression Visualizations ---")
    
    # 2.1 CCC Comparison
    models = [
        "MiniLM L6 (Whole Median)",
        "MiniLM L6 (Whole Mean)",
        "Char TF-IDF (Ridge Tuned)",
        "DistilBERT (Whole Mean_Std)",
        "DistilBERT (Middle Mean)",
        "DistilBERT (Concat Mean, n=54)",
        "Combined TF-IDF (Ridge)",
        "Word TF-IDF (Ridge)",
        "DistilBERT (Whole Attn Seed 42)",
        "DistilBERT (Whole Attn 3-Seed)"
    ]
    ccc_scores = [0.3958, 0.3707, 0.3211, 0.4048, 0.4022, 0.2817, 0.2284, 0.1450, 0.0723, 0.0466]
    colors = [SUCCESS_COLOR, "#98df8a", PRIMARY_COLOR, TEAL_COLOR, "#17becf", PURPLE_COLOR, "#aec7e8", "#c7c7c7", DANGER_COLOR, "#ff9896"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, ccc_scores, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_ylabel("Lin's Concordance Correlation Coefficient (CCC)")
    ax.set_title("Text Pipeline: Regression CCC Agreement Comparison (Dev Split)")
    ax.set_ylim(0, 0.48)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "text/text_regression_ccc_comparison.png")

    # 2.2 RMSE Comparison
    rmse_scores = [5.7390, 5.4370, 4.9812, 5.3423, 5.1890, 5.0706, 5.1323, 5.3356, 5.5017, 5.5686]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, rmse_scores, color=PRIMARY_COLOR, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_ylabel("Root Mean Squared Error (RMSE) - Lower is Better")
    ax.set_title("Text Pipeline: Regression RMSE Comparison (Dev Split)")
    ax.set_ylim(4.0, 6.2)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.04, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "text/text_regression_rmse_comparison.png")

    # 2.3 MAE Comparison
    mae_scores = [4.4512, 4.3168, 3.9942, 4.4008, 4.2374, 4.2280, 4.1420, 4.3937, 4.5404, 4.6084]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, mae_scores, color=TEAL_COLOR, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_ylabel("Mean Absolute Error (MAE) - Lower is Better")
    ax.set_title("Text Pipeline: Regression MAE Comparison (Dev Split)")
    ax.set_ylim(3.5, 5.0)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.03, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "text/text_regression_mae_comparison.png")

    # 2.4 R2 Comparison
    r2_scores = [-0.0272, 0.0781, 0.2262, 0.1102, 0.1606, 0.2210, 0.1785, 0.1121, 0.0564, 0.0333]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_colors = [SUCCESS_COLOR if x >= 0 else DANGER_COLOR for x in r2_scores]
    bars = ax.bar(models, r2_scores, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("Coefficient of Determination (R²)")
    ax.set_title("Text Pipeline: Regression R² Variance Explained (Dev Split)")
    ax.set_ylim(-0.1, 0.3)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        offset = 0.008 if yval >= 0 else -0.015
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + offset, f"{yval:.4f}", ha="center", va=va, fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "text/text_regression_r2_comparison.png")

    # 2.5 Actual vs Predicted Scatter Plots (4 models)
    p_char_r = OUTPUTS_DIR / "text_baselines_final_v1/development/dev_predictions_regression__char_tfidf__ridge__tuned.csv"
    p_dist_w = OUTPUTS_DIR / "text_distilbert_v1/predictions/dev_reg__whole__mean_std__ridge.csv"
    p_dist_m = OUTPUTS_DIR / "text_distilbert_v1/predictions/dev_reg__middle__mean__ridge.csv"
    p_att_r = OUTPUTS_DIR / "text_attention_v1/predictions/dev_reg__whole_attention__seed42.csv"

    scatter_models = []
    if p_char_r.exists():
        d = pd.read_csv(p_char_r)
        scatter_models.append(("Char TF-IDF (Ridge, Tuned)", d["y_true"], d["y_pred_tuned"], 0.3211, 4.9812))
    if p_dist_w.exists():
        d = pd.read_csv(p_dist_w)
        scatter_models.append(("DistilBERT Whole Mean_Std (Ridge)", d["y_true"], d["y_pred"], 0.4048, 5.3423))
    if p_dist_m.exists():
        d = pd.read_csv(p_dist_m)
        scatter_models.append(("DistilBERT Middle Mean (Ridge)", d["y_true"], d["y_pred"], 0.4022, 5.1890))
    if p_att_r.exists():
        d = pd.read_csv(p_att_r)
        scatter_models.append(("DistilBERT Whole Attn (Seed 42)", d["y_true"], d["y_pred"], 0.0723, 5.5017))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (title, yt, yp, ccc, rmse) in zip(axes, scatter_models):
        ax.scatter(yt, yp, color=PRIMARY_COLOR, alpha=0.7, edgecolors="black", linewidths=0.6, s=45)
        # Identity line
        lims = [0, 24]
        ax.plot(lims, lims, color="red", linestyle="--", lw=1.5, label="Identity (y = ŷ)")
        ax.set_xlim(lims)
        ax.set_ylim([-2, 24])
        ax.set_xlabel("Actual PHQ-8 Score")
        ax.set_ylabel("Predicted PHQ-8 Score")
        ax.set_title(f"{title}\nCCC={ccc:.4f} | RMSE={rmse:.2f}", fontsize=10, fontweight="bold")
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Text Pipeline: Actual vs Predicted PHQ Severity on Development Split (n=56)", y=1.05, fontsize=14, fontweight="bold")
    save_fig(fig, "text/text_regression_actual_vs_predicted.png")

    # 2.6 Residuals Plots
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (title, yt, yp, ccc, rmse) in zip(axes, scatter_models):
        residuals = yt - yp
        ax.scatter(yp, residuals, color=PURPLE_COLOR, alpha=0.7, edgecolors="black", linewidths=0.6, s=45)
        ax.axhline(0, color="red", linestyle="--", lw=1.5, label="Zero Error (Residual = 0)")
        ax.set_xlim([-2, 20])
        ax.set_ylim([-15, 18])
        ax.set_xlabel("Predicted PHQ-8 Score (ŷ)")
        ax.set_ylabel("Residual (y - ŷ)")
        ax.set_title(f"{title}", fontsize=10, fontweight="bold")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Text Pipeline: Regression Residual Plots (y - ŷ vs ŷ) (Dev, n=56)", y=1.05, fontsize=14, fontweight="bold")
    save_fig(fig, "text/text_regression_residuals.png")

    # 2.7 Attention Variance Compression Diagnostic
    if p_dist_w.exists() and p_att_r.exists():
        df_w = pd.read_csv(p_dist_w)
        df_a = pd.read_csv(p_att_r)
        
        y_true = df_w["y_true"].values
        pred_fixed = df_w["y_pred"].values
        pred_attn = df_a["y_pred"].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # Distribution KDE / Histogram
        sns.kdeplot(y_true, ax=ax1, label=f"Ground Truth (std={np.std(y_true):.2f})", color="black", lw=2.2, fill=True, alpha=0.1)
        sns.kdeplot(pred_fixed, ax=ax1, label=f"Fixed-Pooling Ridge (std={np.std(pred_fixed):.2f})", color=SUCCESS_COLOR, lw=2.0)
        sns.kdeplot(pred_attn, ax=ax1, label=f"Attention Pooling (std={np.std(pred_attn):.2f})", color=DANGER_COLOR, lw=2.0)
        ax1.set_xlabel("PHQ-8 Score")
        ax1.set_ylabel("Density")
        ax1.set_title("Distribution Spread & Prediction Compression")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        # Standard Deviation Bar Contrast
        std_labels = ["Ground Truth\ny_true", "DistilBERT Fixed\n(whole__mean_std)", "DistilBERT Attn\n(whole_attention)"]
        stds = [np.std(y_true), np.std(pred_fixed), np.std(pred_attn)]
        bars = ax2.bar(std_labels, stds, color=["black", SUCCESS_COLOR, DANGER_COLOR], edgecolor="black", width=0.5)
        ax2.set_ylabel("Standard Deviation (σ)")
        ax2.set_title("Variance Compression Diagnostic (Phase 18C Finding)")
        ax2.set_ylim(0, 7.5)
        ax2.grid(axis="y")
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"σ = {yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.text(2, stds[2] + 0.8, "Severe Variance\nCollapse (-63%)", ha="center", color=DANGER_COLOR, fontweight="bold", fontsize=9)

        fig.suptitle("Phase 18C Forensic Diagnostic: Prediction Variance Collapse in Attention Pooling", y=1.03, fontsize=14, fontweight="bold")
        save_fig(fig, "text/text_attention_regression_diagnostics.png")

    # 2.8 Pipeline Progression Across Phases
    phases = [
        "Phase 17A\n(Char TF-IDF)",
        "Phase 17B\n(MiniLM L6 Whole)",
        "Phase 18A\n(MiniLM Middle)",
        "Phase 18B\n(DistilBERT Whole)",
        "Phase 18C\n(DistilBERT Attn)"
    ]
    p_f1 = [0.4571, 0.4000, 0.5161, 0.4516, 0.4348]
    p_auc = [0.7159, 0.6534, 0.6894, 0.6591, 0.7121]
    p_ccc = [0.3211, 0.3057, 0.3138, 0.4048, 0.0723]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # Classification progression
    ax1.plot(phases, p_f1, marker="o", lw=2.2, color=PRIMARY_COLOR, label="Dev F1-Score")
    ax1.plot(phases, p_auc, marker="s", lw=2.2, color=SECONDARY_COLOR, label="Dev ROC-AUC")
    ax1.set_ylabel("Classification Score")
    ax1.set_ylim(0.35, 0.78)
    ax1.set_title("Classification Progression Across Phases")
    ax1.legend(loc="lower left")
    ax1.grid(True)
    for i, (f, a) in enumerate(zip(p_f1, p_auc)):
        ax1.annotate(f"{f:.4f}", (i, f), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")
        ax1.annotate(f"{a:.4f}", (i, a), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, fontweight="bold", color=SECONDARY_COLOR)

    # Regression progression
    ax2.plot(phases, p_ccc, marker="^", lw=2.5, color=SUCCESS_COLOR, label="Dev CCC Agreement")
    ax2.set_ylabel("Lin's CCC")
    ax2.set_ylim(0.0, 0.48)
    ax2.set_title("Regression Concordance (CCC) Progression Across Phases")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    for i, c in enumerate(p_ccc):
        ax2.annotate(f"{c:.4f}", (i, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Text Pipeline Architecture Progression: Phase 17A to Phase 18C (Dev Split)", y=1.03, fontsize=14, fontweight="bold")
    save_fig(fig, "text/text_pipeline_progression.png")

# ==============================================================================
# 3. Audio Visualizations (Stage E)
# ==============================================================================

def generate_audio_plots():
    print("--- Generating Audio Visualizations ---")
    
    # 3.1 Audio Classification F1 Comparison
    audio_models = [
        "VGG16 Mean\n(LinearSVC)",
        "VGG16 Mean_Std\n(LinearSVC)",
        "Raw MFCC\n(LinearSVC)",
        "VGG16 Mean_Std\n(MLP Classifier)",
        "Raw eGeMAPS\n(LinearSVC)",
        "BoAW MFCC\n(LinearSVC)",
        "DenseNet201\n(LinearSVC)",
        "VGG16 GRU+Attn\n(Temporal)"
    ]
    f1_audio = [0.4286, 0.4118, 0.4000, 0.3529, 0.2222, 0.2857, 0.2667, 0.0000]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(audio_models, f1_audio, color=PRIMARY_COLOR, edgecolor="black", width=0.6)
    ax.set_ylabel("Development F1-Score")
    ax.set_title("Voice/Audio: Classification F1-Score Across Acoustic & Deep Architectures")
    ax.set_ylim(0, 0.52)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "audio/audio_classification_f1_comparison.png")

    # 3.2 Audio Classification ROC-AUC Comparison
    auc_audio = [0.6477, 0.6458, 0.6761, 0.6629, 0.5057, 0.5985, 0.5909, 0.6648]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(audio_models, auc_audio, color=TEAL_COLOR, edgecolor="black", width=0.6)
    ax.axhline(0.5, color="red", linestyle="--", label="Random Chance (0.50)")
    ax.set_ylabel("Development ROC-AUC")
    ax.set_title("Voice/Audio: ROC-AUC Discrimination Across Audio Representations")
    ax.set_ylim(0.4, 0.75)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    ax.legend(loc="upper right")
    save_fig(fig, "audio/audio_classification_roc_auc_comparison.png")

    # 3.3 Audio Confusion Matrices
    p_mfcc_cls = OUTPUTS_DIR / "classical_controlled/predictions/raw_mfcc__mean_std__linear_svc__dev__predictions.csv"
    p_vgg_cls = OUTPUTS_DIR / "cnn_embedding_classical_v1/predictions/dev/vgg16__mean_std__linear_svc__dev_pred.csv"
    p_mlp_cls = OUTPUTS_DIR / "cnn_embedding_neural_head_v1/predictions/dev/vgg16__mean_std__mlp_class__dev_pred.csv"
    p_gru_cls = OUTPUTS_DIR / "cnn_embedding_temporal_v1/predictions/dev/vgg16__classification__dev.csv"

    cm_audio = []
    if p_mfcc_cls.exists():
        d = pd.read_csv(p_mfcc_cls)
        cm_audio.append(("Raw MFCC LinearSVC", confusion_matrix(d["y_true"], d["y_pred"])))
    if p_vgg_cls.exists():
        d = pd.read_csv(p_vgg_cls)
        cm_audio.append(("VGG16 Mean_Std LinearSVC", confusion_matrix(d["y_true"], d["y_pred"])))
    if p_mlp_cls.exists():
        d = pd.read_csv(p_mlp_cls)
        cm_audio.append(("VGG16 Mean_Std MLP", confusion_matrix(d["y_true"], d["y_pred"])))
    if p_gru_cls.exists():
        d = pd.read_csv(p_gru_cls)
        y_hard = (d["y_pred"] >= 0.5).astype(int) if d["y_pred"].dtype == float else d["y_pred"]
        cm_audio.append(("VGG16 GRU+Attention", confusion_matrix(d["y_true"], y_hard, labels=[0, 1])))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, cm) in zip(axes, cm_audio):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax,
                    xticklabels=["Non-Dep", "Depressed"], yticklabels=["Non-Dep", "Depressed"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    fig.suptitle("Voice/Audio: Development Split Confusion Matrices (n=56)", y=1.05, fontsize=14, fontweight="bold")
    save_fig(fig, "audio/audio_classification_confusion_matrices.png")

    # 3.4 Audio Regression CCC Comparison
    reg_models = [
        "Raw MFCC (Ridge)",
        "DenseNet201 (MLP)",
        "BoAW MFCC (XGBoost)",
        "DenseNet201 (Ridge)",
        "VGG16 GRU+Attn (Temporal)",
        "Raw eGeMAPS (Ridge)",
        "VGG16 Mean_Std (Ridge)"
    ]
    ccc_audio = [0.2844, 0.1654, 0.1613, 0.1496, 0.0958, 0.0273, 0.0479]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(reg_models, ccc_audio, color=SECONDARY_COLOR, edgecolor="black", width=0.55)
    ax.set_ylabel("Concordance Correlation Coefficient (CCC)")
    ax.set_title("Voice/Audio: Regression CCC Agreement (Dev Split)")
    ax.set_ylim(0, 0.35)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.007, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "audio/audio_regression_ccc_comparison.png")

    # 3.5 Audio Regression RMSE Comparison
    rmse_audio = [6.2167, 6.4523, 5.7172, 7.4676, 6.9533, 6.3981, 13.5263]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(reg_models, rmse_audio, color=BROWN_COLOR, edgecolor="black", width=0.55)
    ax.set_ylabel("Root Mean Squared Error (RMSE) - Lower is Better")
    ax.set_title("Voice/Audio: Regression RMSE Error (Dev Split)")
    ax.set_ylim(4.5, 14.5)
    ax.grid(axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "audio/audio_regression_rmse_comparison.png")

    # 3.6 Audio Actual vs Predicted
    p_mfcc_reg = OUTPUTS_DIR / "classical_controlled/predictions/raw_mfcc__mean_std__ridge__dev__predictions.csv"
    p_xgb_reg = OUTPUTS_DIR / "classical_controlled/predictions/boaw_mfcc__mean_std__xgboost__dev__predictions.csv"
    p_dense_reg = OUTPUTS_DIR / "cnn_embedding_classical_v1/predictions/dev/densenet201__mean__ridge__dev_pred.csv"
    p_gru_reg = OUTPUTS_DIR / "cnn_embedding_temporal_v1/predictions/dev/vgg16__regression__dev.csv"

    audio_scatters = []
    if p_mfcc_reg.exists():
        d = pd.read_csv(p_mfcc_reg)
        audio_scatters.append(("Raw MFCC (Ridge)", d["y_true"], d["y_pred"], 0.2844, 6.22))
    if p_xgb_reg.exists():
        d = pd.read_csv(p_xgb_reg)
        audio_scatters.append(("BoAW MFCC (XGBoost)", d["y_true"], d["y_pred"], 0.1613, 5.72))
    if p_dense_reg.exists():
        d = pd.read_csv(p_dense_reg)
        audio_scatters.append(("DenseNet201 (Ridge)", d["y_true"], d["y_pred"], 0.1496, 7.47))
    if p_gru_reg.exists():
        d = pd.read_csv(p_gru_reg)
        audio_scatters.append(("VGG16 GRU+Attn", d["y_true"], d["y_pred"], 0.0958, 6.95))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    lims = [0, 24]
    for ax, (title, yt, yp, ccc, rmse) in zip(axes, audio_scatters):
        ax.scatter(yt, yp, color=SECONDARY_COLOR, alpha=0.7, edgecolors="black", linewidths=0.6, s=45)
        ax.plot(lims, lims, color="red", linestyle="--", lw=1.5, label="Identity (y = ŷ)")
        ax.set_xlim(lims)
        ax.set_ylim([-3, 24])
        ax.set_xlabel("Actual PHQ-8 Score")
        ax.set_ylabel("Predicted PHQ-8 Score")
        ax.set_title(f"{title}\nCCC={ccc:.4f} | RMSE={rmse:.2f}", fontsize=10, fontweight="bold")
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Voice/Audio: Actual vs Predicted PHQ Severity on Development Split (n=56)", y=1.05, fontsize=14, fontweight="bold")
    save_fig(fig, "audio/audio_regression_actual_vs_predicted.png")

    # 3.7 Audio Feature Representation Overview
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rep_types = ["Classical Acoustic\n(MFCC / eGeMAPS)", "Bag-of-Audio-Words\n(BoAW Codebooks)", "Pretrained CNN\n(VGG16 / DenseNet)", "Temporal GRU\n(Attention Stream)"]
    rep_f1 = [0.4000, 0.2857, 0.4286, 0.0000]
    rep_ccc = [0.2844, 0.1613, 0.1654, 0.0958]

    bars1 = ax1.bar(rep_types, rep_f1, color=PRIMARY_COLOR, edgecolor="black", width=0.5)
    ax1.set_ylabel("Peak Dev F1-Score")
    ax1.set_title("Peak Audio Classification by Feature Type")
    ax1.set_ylim(0, 0.5)
    ax1.grid(axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(rep_types, rep_ccc, color=SECONDARY_COLOR, edgecolor="black", width=0.5)
    ax2.set_ylabel("Peak Dev CCC Agreement")
    ax2.set_title("Peak Audio Regression by Feature Type")
    ax2.set_ylim(0, 0.35)
    ax2.grid(axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Acoustic Feature Comparison: Classical LLDs vs BoAW vs Pretrained CNNs", y=1.03, fontsize=14, fontweight="bold")
    save_fig(fig, "audio/audio_feature_comparison.png")

# ==============================================================================
# 4. Cross-Modality Comparison Visualizations
# ==============================================================================

def generate_cross_modality_plots():
    print("--- Generating Cross-Modality Visualizations ---")
    
    # 4.1 Audio vs Text Classification
    categories = [
        "Top Classical",
        "Top Pretrained Transfer",
        "Top Deep/Attention"
    ]
    audio_f1 = [0.4000, 0.4286, 0.3529] # MFCC, VGG16 mean, VGG16 MLP
    text_f1 = [0.4571, 0.4516, 0.5000]  # Char TF-IDF, DistilBERT median, DistilBERT middle attn
    
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars_a = ax.bar(x - width/2, audio_f1, width, label="Voice / Audio", color=SECONDARY_COLOR, edgecolor="black")
    bars_t = ax.bar(x + width/2, text_f1, width, label="Conversational Text", color=PRIMARY_COLOR, edgecolor="black")

    ax.set_ylabel("Development F1-Score")
    ax.set_title("Cross-Modality Benchmark: Audio vs Text Classification Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 0.60)
    ax.grid(axis="y")
    ax.legend(loc="upper left")

    for bar in bars_a:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_t:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    save_fig(fig, "comparison/audio_vs_text_classification_comparison.png")

    # 4.2 Audio vs Text Regression
    audio_ccc = [0.2844, 0.1654, 0.0958] # MFCC, DenseNet MLP, VGG16 GRU
    text_ccc = [0.3211, 0.4048, 0.0723]  # Char TF-IDF, DistilBERT mean_std, DistilBERT attn
    
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars_a = ax.bar(x - width/2, audio_ccc, width, label="Voice / Audio", color=SECONDARY_COLOR, edgecolor="black")
    bars_t = ax.bar(x + width/2, text_ccc, width, label="Conversational Text", color=PRIMARY_COLOR, edgecolor="black")

    ax.set_ylabel("Development CCC Agreement")
    ax.set_title("Cross-Modality Benchmark: Audio vs Text Severity Tracking (CCC)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 0.48)
    ax.grid(axis="y")
    ax.legend(loc="upper right")

    for bar in bars_a:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_t:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    save_fig(fig, "comparison/audio_vs_text_regression_comparison.png")

if __name__ == "__main__":
    generate_text_classification_plots()
    generate_text_regression_plots()
    generate_audio_plots()
    generate_cross_modality_plots()
    print("\nAll Phase 20 visualizations successfully generated!")
