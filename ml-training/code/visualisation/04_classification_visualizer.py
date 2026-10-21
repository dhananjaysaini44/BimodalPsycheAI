"""
Phase 15 - Script 04: Classification Model Visualizer and Benchmark Comparisons.

Produces:
1. Individual Performance Visualizations (PNG only):
   - Confusion Matrix (counts + percentages, TP/FP/TN/FN)
   - ROC Curve (FPR vs TPR, chance line, AUROC)
   - Precision-Recall Curve (Recall vs Precision, baseline prevalence line, AP/PR-AUC)
   - Score / Probability Distribution (KDE/histogram split by class)
   - Calibration Curve (Reliability diagram, Brier score)
   - Combined 4-panel diagnostic card per model
   - Model card markdown summary (.md)
2. Benchmark / Cross-Model Comparisons:
   - Best F1 and ROC-AUC by representation on Dev
   - Family / Suite comparison (Broad Acoustic vs Controlled vs CNN Classical vs CNN MLP vs CNN Temporal)
   - Multi-model ROC overlay on Dev
   - Multi-model PR overlay on Dev
   - Split generalization comparison (Train OOF vs Dev vs Test)
   - Threshold starvation analysis

All figures saved exclusively in PNG format.
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

# Paths
PROJECT_ROOT = Path("E:/Projects/BimodalPsycheAI")
ML_TRAINING = PROJECT_ROOT / "ml-training"
OUTPUT_DIR = ML_TRAINING / "outputs" / "model_visualizations_v1"
CLS_DIR = OUTPUT_DIR / "classification"

# Subdirectories
CM_DIR = CLS_DIR / "confusion_matrices"
ROC_DIR = CLS_DIR / "roc_curves"
PR_DIR = CLS_DIR / "precision_recall_curves"
DIST_DIR = CLS_DIR / "score_distributions"
CALIB_DIR = CLS_DIR / "calibration_curves"
CARD_DIR = CLS_DIR / "individual_models"
COMP_DIR = CLS_DIR / "comparisons"

for d in [CM_DIR, ROC_DIR, PR_DIR, DIST_DIR, CALIB_DIR, CARD_DIR, COMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Global styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_prediction_data(pred_path_str):
    """Loads prediction CSV and normalizes column names and types."""
    p_path = PROJECT_ROOT / pred_path_str
    if not p_path.exists():
        p_path = ML_TRAINING / pred_path_str
    if not p_path.exists():
        return None

    df = pd.read_csv(p_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return None

    y_true = df["y_true"].to_numpy().astype(int)
    y_pred = df["y_pred"].to_numpy()

    # Determine continuous score/probability
    y_score = None
    if "y_score" in df.columns:
        y_score = df["y_score"].to_numpy(dtype=float)
    elif "decision_function" in df.columns:
        y_score = df["decision_function"].to_numpy(dtype=float)
    elif "probability" in df.columns:
        y_score = df["probability"].to_numpy(dtype=float)
    elif np.issubdtype(y_pred.dtype, np.floating) and not np.all(np.isin(y_pred, [0.0, 1.0])):
        # y_pred contains continuous probabilities (e.g. Phase 13B temporal)
        y_score = y_pred.astype(float)
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_pred = y_pred.astype(int)

    if y_pred.dtype != int:
        y_pred = (y_pred >= 0.5).astype(int)

    return y_true, y_pred, y_score


def plot_confusion_matrix(y_true, y_pred, model_id, model_label, split_name, out_path):
    """Generates an annotated 2x2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = len(y_true)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1.5,
        linecolor="#ffffff",
        ax=ax,
    )

    labels = [
        [f"TN\n{tn}\n({tn/total:.1%})", f"FP\n{fp}\n({fp/total:.1%})"],
        [f"FN\n{fn}\n({fn/total:.1%})", f"TP\n{tp}\n({tp/total:.1%})"],
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > total * 0.4 else "black"
            ax.text(j + 0.5, i + 0.5, labels[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Non-Depressed (0)", "Depressed (1)"], fontsize=10)
    ax.set_yticklabels(["Non-Depressed (0)", "Depressed (1)"], fontsize=10, va="center")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold", labelpad=8)

    acc = (tp + tn) / total if total > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    ax.set_title(f"Confusion Matrix: {model_label}\nSplit: {split_name} (n={total}) | Acc: {acc:.3f} | F1: {f1:.3f}",
                 fontsize=10.5, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(y_true, y_score, model_id, model_label, split_name, out_path):
    """Generates an individual ROC curve."""
    if y_score is None or len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.2, label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.2, linestyle="--", label="Chance (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, fontweight="bold")
    ax.set_title(f"ROC Curve: {model_label}\nSplit: {split_name} (n={len(y_true)})", fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_pr_curve(y_true, y_score, model_id, model_label, split_name, out_path):
    """Generates an individual Precision-Recall curve."""
    if y_score is None or len(np.unique(y_true)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap_val = average_precision_score(y_true, y_score)
    prevalence = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    ax.plot(rec, prec, color="#d95f02", lw=2.2, label=f"PR (AP = {ap_val:.3f})")
    ax.axhline(prevalence, color="#999999", lw=1.2, linestyle="--", label=f"Baseline ({prevalence:.1%})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10, fontweight="bold")
    ax.set_title(f"Precision-Recall Curve: {model_label}\nSplit: {split_name} (n={len(y_true)})", fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_score_distribution(y_true, y_score, model_id, model_label, split_name, out_path):
    """Generates prediction score/probability distribution split by ground truth."""
    if y_score is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)

    scores_0 = y_score[y_true == 0]
    scores_1 = y_score[y_true == 1]

    # Normalize to probability scale if scores look like logits / decision function
    if np.min(y_score) < 0 or np.max(y_score) > 1:
        # Standard logistic sigmoid for visualization comparison
        s0_plot = 1.0 / (1.0 + np.exp(-scores_0))
        s1_plot = 1.0 / (1.0 + np.exp(-scores_1))
        xlabel = "Sigmoid-Transformed Decision Score"
        thresh_line = 0.5
    else:
        s0_plot = scores_0
        s1_plot = scores_1
        xlabel = "Predicted Probability P(Depressed = 1)"
        thresh_line = 0.5

    bins = np.linspace(0, 1, 15)
    ax.hist(s0_plot, bins=bins, alpha=0.6, color="#2b5c8f", label=f"Non-Depressed (0, n={len(s0_plot)})", edgecolor="white")
    ax.hist(s1_plot, bins=bins, alpha=0.65, color="#d95f02", label=f"Depressed (1, n={len(s1_plot)})", edgecolor="white")
    ax.axvline(thresh_line, color="black", linestyle="--", lw=1.2, label=f"Threshold ({thresh_line})")

    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel("Participant Count", fontsize=10, fontweight="bold")
    ax.set_title(f"Score Distribution: {model_label}\nSplit: {split_name}", fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_calibration_curve_fig(y_true, y_score, model_id, model_label, split_name, out_path):
    """Generates reliability diagram."""
    if y_score is None or len(np.unique(y_true)) < 2:
        return

    # Convert to probabilities if necessary
    if np.min(y_score) < 0 or np.max(y_score) > 1:
        y_prob = 1.0 / (1.0 + np.exp(-y_score))
    else:
        y_prob = y_score

    brier = brier_score_loss(y_true, y_prob)

    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#7570b3", lw=2, markersize=6, label=f"Model (Brier={brier:.3f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontsize=10, fontweight="bold")
    ax.set_title(f"Calibration Curve: {model_label}\nSplit: {split_name}", fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_combined_model_card(y_true, y_pred, y_score, model_row, out_path):
    """Generates a comprehensive 4-panel diagnostic card for the model."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=150)

    total = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Panel 1: Confusion Matrix
    ax = axes[0, 0]
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=False, square=True, linewidths=1.2, ax=ax)
    labels = [
        [f"TN\n{tn}\n({tn/total:.1%})", f"FP\n{fp}\n({fp/total:.1%})"],
        [f"FN\n{fn}\n({fn/total:.1%})", f"TP\n{tp}\n({tp/total:.1%})"],
    ]
    for i in range(2):
        for j in range(2):
            c = "white" if cm[i, j] > total * 0.4 else "black"
            ax.text(j + 0.5, i + 0.5, labels[i][j], ha="center", va="center", fontsize=11, fontweight="bold", color=c)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9.5)
    ax.set_yticklabels(["True 0", "True 1"], fontsize=9.5, va="center")
    ax.set_title("A. Confusion Matrix", fontsize=11, fontweight="bold")

    # Panel 2: ROC Curve
    ax = axes[0, 1]
    if y_score is not None and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.50)")
        ax.set_title("B. Receiver Operating Characteristic", fontsize=11, fontweight="bold")
        ax.set_xlabel("FPR (1 - Specificity)", fontsize=9.5)
        ax.set_ylabel("TPR (Recall)", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "Continuous scores unavailable\n(Discrete operating point only)",
                ha="center", va="center", fontsize=10, color="#666666")
        ax.set_title("B. ROC Curve (Unavailable)", fontsize=11, fontweight="bold")

    # Panel 3: Precision-Recall Curve
    ax = axes[1, 0]
    if y_score is not None and len(np.unique(y_true)) > 1:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap_val = average_precision_score(y_true, y_score)
        prev = np.mean(y_true)
        ax.plot(rec, prec, color="#d95f02", lw=2, label=f"PR (AP = {ap_val:.3f})")
        ax.axhline(prev, color="k", linestyle="--", lw=1, label=f"Prevalence ({prev:.1%})")
        ax.set_title("C. Precision-Recall Curve", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=9.5)
        ax.set_ylabel("Precision", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "Continuous scores unavailable\n(Discrete operating point only)",
                ha="center", va="center", fontsize=10, color="#666666")
        ax.set_title("C. PR Curve (Unavailable)", fontsize=11, fontweight="bold")

    # Panel 4: Score Distribution or Class Balance
    ax = axes[1, 1]
    if y_score is not None:
        if np.min(y_score) < 0 or np.max(y_score) > 1:
            s_trans = 1.0 / (1.0 + np.exp(-y_score))
            lbl = "Sigmoid-Transformed Score"
        else:
            s_trans = y_score
            lbl = "Predicted Probability"

        s0 = s_trans[y_true == 0]
        s1 = s_trans[y_true == 1]
        bins = np.linspace(0, 1, 12)
        ax.hist(s0, bins=bins, alpha=0.6, color="#2b5c8f", label=f"Class 0 (n={len(s0)})", edgecolor="white")
        ax.hist(s1, bins=bins, alpha=0.65, color="#d95f02", label=f"Class 1 (n={len(s1)})", edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", lw=1.2, label="Cutoff (0.5)")
        ax.set_title("D. Prediction Score Separation", fontsize=11, fontweight="bold")
        ax.set_xlabel(lbl, fontsize=9.5)
        ax.set_ylabel("Count", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
    else:
        ax.bar(["Pred 0", "Pred 1"], [np.sum(y_pred == 0), np.sum(y_pred == 1)], color=["#2b5c8f", "#d95f02"], width=0.4)
        ax.set_title("D. Predicted Class Counts", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count", fontsize=9.5)

    suite_name = model_row["benchmark_suite"]
    arch = model_row["architecture"]
    rep = model_row["representation"]
    pool = model_row["pooling"]
    split = model_row["evaluation_split"]
    f1_val = model_row["f1"]
    acc_val = model_row["accuracy"]
    bacc_val = model_row["balanced_accuracy"]

    plt.suptitle(
        f"Performance Card: {arch} | {rep} ({pool}) | Split: {split}\n"
        f"Suite: {suite_name} | F1: {f1_val:.3f} | Acc: {acc_val:.3f} | Bal Acc: {bacc_val:.3f}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def generate_model_markdown(row, out_path):
    """Generates an individual model card markdown file."""
    model_id = row["model_id"]
    content = f"""# Model Card: `{model_id}`

## Model Summary
- **Benchmark Suite**: {row['benchmark_suite']}
- **Task**: Binary Depression Classification (`PHQ_Binary` >= 10)
- **Architecture**: `{row['architecture']}`
- **Feature Representation**: `{row['representation']}`
- **Pooling Strategy**: `{row['pooling']}`
- **Evaluation Split**: `{row['evaluation_split']}`
- **Participant Count**: {row['participant_count']}

## Quantitative Metrics
| Metric | Verified Value |
| :--- | :--- |
| **F1-Score** | **{row['f1']:.4f}** |
| **Balanced Accuracy** | **{row['balanced_accuracy']:.4f}** |
| **Accuracy** | {row['accuracy']:.4f} |
| **Precision** | {row['precision']:.4f} |
| **Recall / Sensitivity** | {row['recall']:.4f} |
| **ROC-AUC** | {f"{row['roc_auc']:.4f}" if pd.notna(row['roc_auc']) else "N/A"} |
| **Average Precision (PR-AUC)** | {f"{row['average_precision']:.4f}" if pd.notna(row['average_precision']) else "N/A"} |

## Confusion Breakdown
- **True Negatives (TN)**: {row['tn']}
- **False Positives (FP)**: {row['fp']}
- **False Negatives (FN)**: {row['fn']}
- **True Positives (TP)**: {row['tp']}

## Diagnostic Interpretation
"""
    if row["tp"] == 0:
        content += "- **Threshold Starvation Detected**: Model predicted 0 for all instances (TP = 0, FP = 0), resulting in an undefined or zero F1-score due to severe class imbalance.\n"
    elif row["f1"] > 0.4:
        content += f"- **High-Performing Benchmark**: This model achieves strong positive class detection (F1 = {row['f1']:.3f}) under the official split.\n"
    else:
        content += f"- **Moderate Baseline**: Standard operating point yielded F1 = {row['f1']:.3f}.\n"

    content += f"""
## Artifact References
- **Prediction Artifact**: `{row['prediction_artifact']}`
- **Config Artifact**: `{row['config_artifact'] if pd.notna(row['config_artifact']) else "None"}`
- **Diagnostic Visual Card**: `classification/individual_models/{model_id}__card.png`
"""
    out_path.write_text(content, encoding="utf-8")


def process_single_model_worker(row_dict):
    """Worker function for multiprocessing."""
    model_id = row_dict["model_id"]
    pred_art = row_dict["prediction_artifact"]
    split = row_dict["evaluation_split"]
    label = f"{row_dict['architecture']} ({row_dict['representation']})"

    data = load_prediction_data(pred_art)
    if data is None:
        return False
    y_true, y_pred, y_score = data

    # 1. Confusion Matrix
    cm_path = CM_DIR / f"{model_id}__cm.png"
    plot_confusion_matrix(y_true, y_pred, model_id, label, split, cm_path)

    # 2. Continuous plots (if available)
    if y_score is not None:
        roc_path = ROC_DIR / f"{model_id}__roc.png"
        plot_roc_curve(y_true, y_score, model_id, label, split, roc_path)

        pr_path = PR_DIR / f"{model_id}__pr.png"
        plot_pr_curve(y_true, y_score, model_id, label, split, pr_path)

        dist_path = DIST_DIR / f"{model_id}__dist.png"
        plot_score_distribution(y_true, y_score, model_id, label, split, dist_path)

        calib_path = CALIB_DIR / f"{model_id}__calib.png"
        plot_calibration_curve_fig(y_true, y_score, model_id, label, split, calib_path)

    # 3. Combined Card & Markdown
    card_path = CARD_DIR / f"{model_id}__card.png"
    plot_combined_model_card(y_true, y_pred, y_score, row_dict, card_path)

    md_path = CARD_DIR / f"{model_id}.md"
    generate_model_markdown(row_dict, md_path)

    return True


def generate_benchmark_comparisons(df_cls):
    """Generates comprehensive cross-model and cross-family benchmark comparisons."""
    print("--- Generating Classification Benchmark Comparisons ---")

    # Filter for Dev split for benchmark comparison
    df_dev = df_cls[df_cls["evaluation_split"] == "dev"].copy()

    # 1. Best F1 by Representation on Dev
    rep_best = (
        df_dev.groupby(["representation", "pooling", "benchmark_suite"])["f1"]
        .max()
        .reset_index()
        .sort_values("f1", ascending=False)
    )
    rep_best["label"] = rep_best["representation"] + " (" + rep_best["pooling"] + ")"

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    colors = sns.color_palette("muted", len(rep_best))
    bars = ax.barh(np.arange(len(rep_best)), rep_best["f1"], color="#2b5c8f", edgecolor="black", linewidth=0.6)
    ax.set_yticks(np.arange(len(rep_best)))
    ax.set_yticklabels(rep_best["label"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Best Validation F1-Score (Dev Split, n=56)", fontsize=10, fontweight="bold")
    ax.set_title("Benchmark Comparison: Best F1-Score Across All Feature Representations (Dev Split)",
                 fontsize=11.5, fontweight="bold", pad=12)
    ax.axvline(0.25, color="red", linestyle="--", lw=1, label="Chance Positive F1 (~0.25)")
    ax.legend(loc="lower right", fontsize=9)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2.0, f"{w:.3f}", ha="left", va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(0, max(rep_best["f1"].max() * 1.15, 0.55))
    plt.tight_layout()
    plt.savefig(COMP_DIR / "benchmark_f1_by_representation_dev.png")
    plt.close(fig)

    # 2. Family / Suite Comparison
    # Group by benchmark suite and take best model per suite
    suite_best = []
    for suite, g in df_dev.groupby("benchmark_suite"):
        top_m = g.sort_values(["f1", "balanced_accuracy"], ascending=False).iloc[0]
        suite_best.append({
            "suite": suite,
            "best_model": f"{top_m['architecture']}\n({top_m['representation']})",
            "f1": top_m["f1"],
            "balanced_accuracy": top_m["balanced_accuracy"],
            "roc_auc": top_m["roc_auc"] if pd.notna(top_m["roc_auc"]) else 0.5,
        })
    df_suite = pd.DataFrame(suite_best).sort_values("f1", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    x = np.arange(len(df_suite))
    w = 0.25
    ax.bar(x - w, df_suite["f1"], width=w, label="F1-Score", color="#2b5c8f", edgecolor="black", lw=0.6)
    ax.bar(x, df_suite["balanced_accuracy"], width=w, label="Balanced Accuracy", color="#d95f02", edgecolor="black", lw=0.6)
    ax.bar(x + w, df_suite["roc_auc"], width=w, label="ROC-AUC", color="#7570b3", edgecolor="black", lw=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(df_suite["suite"] + "\n" + df_suite["best_model"], fontsize=8.5)
    ax.set_ylabel("Metric Score", fontsize=10, fontweight="bold")
    ax.set_title("Cross-Family Benchmark Comparison: Best Models by Suite (Dev Split, n=56)",
                 fontsize=11.5, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for i in range(len(df_suite)):
        ax.text(i - w, df_suite["f1"].iloc[i] + 0.02, f"{df_suite['f1'].iloc[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i, df_suite["balanced_accuracy"].iloc[i] + 0.02, f"{df_suite['balanced_accuracy'].iloc[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        if df_suite["roc_auc"].iloc[i] > 0.5:
            ax.text(i + w, df_suite["roc_auc"].iloc[i] + 0.02, f"{df_suite['roc_auc'].iloc[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(COMP_DIR / "family_comparison_dev.png")
    plt.close(fig)

    # 3. Multi-Model ROC Overlay (Top Model from each Suite on Dev)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    plotted_any = False

    for idx, (suite, g) in enumerate(df_dev.groupby("benchmark_suite")):
        valid_scores = g[g["has_continuous_scores"] & g["roc_auc"].notna()]
        if valid_scores.empty:
            continue
        top_row = valid_scores.sort_values("roc_auc", ascending=False).iloc[0]
        data = load_prediction_data(top_row["prediction_artifact"])
        if data is None:
            continue
        y_true, _, y_score = data
        if y_score is None or len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_v = roc_auc_score(y_true, y_score)
        c = colors[idx % len(colors)]
        label = f"{suite[:18]}: {top_row['architecture']} ({auc_v:.3f})"
        ax.plot(fpr, tpr, color=c, lw=2, label=label)
        plotted_any = True

    if plotted_any:
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance (0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=10, fontweight="bold")
        ax.set_title("Multi-Model ROC Overlay: Top Model per Benchmark Suite (Dev Split)",
                     fontsize=11, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(COMP_DIR / "top_models_multi_roc_overlay_dev.png")
    plt.close(fig)

    # 4. Multi-Model PR Overlay (Top Model from each Suite on Dev)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    plotted_pr = False
    for idx, (suite, g) in enumerate(df_dev.groupby("benchmark_suite")):
        valid_scores = g[g["has_continuous_scores"] & g["average_precision"].notna()]
        if valid_scores.empty:
            continue
        top_row = valid_scores.sort_values("average_precision", ascending=False).iloc[0]
        data = load_prediction_data(top_row["prediction_artifact"])
        if data is None:
            continue
        y_true, _, y_score = data
        if y_score is None or len(np.unique(y_true)) < 2:
            continue

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap_v = average_precision_score(y_true, y_score)
        c = colors[idx % len(colors)]
        label = f"{suite[:18]}: {top_row['architecture']} ({ap_v:.3f})"
        ax.plot(rec, prec, color=c, lw=2, label=label)
        plotted_pr = True

    if plotted_pr:
        ax.axhline(0.25, color="k", linestyle="--", lw=1.2, label="Prevalence Baseline (0.25)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Precision (PPV)", fontsize=10, fontweight="bold")
        ax.set_title("Multi-Model PR Overlay: Top Model per Benchmark Suite (Dev Split)",
                     fontsize=11, fontweight="bold", pad=12)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(COMP_DIR / "top_models_multi_pr_overlay_dev.png")
    plt.close(fig)

    # 5. Split Generalization Comparison (Train OOF vs Dev vs Test)
    # Using Controlled Classical and Official Final Test models
    ctrl_models = df_cls[df_cls["benchmark_suite"].isin(["Controlled Classical Benchmark", "Official Final Test Benchmark"])]
    if not ctrl_models.empty:
        # Group by representation and architecture
        split_piv = ctrl_models.pivot_table(index=["representation", "architecture"], columns="evaluation_split", values="f1")
        if "train_oof" in split_piv and "dev" in split_piv:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
            x_pos = np.arange(len(split_piv))
            w = 0.28
            ax.bar(x_pos - w, split_piv["train_oof"], width=w, label="Train 5-Fold OOF (n=163)", color="#2b5c8f", edgecolor="black")
            ax.bar(x_pos, split_piv["dev"], width=w, label="Development Split (n=56)", color="#d95f02", edgecolor="black")
            if "test" in split_piv:
                ax.bar(x_pos + w, split_piv["test"], width=w, label="Historical Final Test (n=56)", color="#1b9e77", edgecolor="black")

            labels = [f"{r}\n({a})" for r, a in split_piv.index]
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("F1-Score", fontsize=10, fontweight="bold")
            ax.set_title("Generalization Across Splits: Train OOF vs. Dev vs. Test", fontsize=11.5, fontweight="bold", pad=12)
            ax.legend(loc="upper right", fontsize=9)
            ax.set_ylim(0, 0.75)
            ax.grid(True, axis="y", linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(COMP_DIR / "split_comparison_train_dev_test.png")
            plt.close(fig)

    # 6. Threshold Starvation Analysis
    # Count how many models predicted all 0s (TP=0, FP=0)
    starved = df_dev[(df_dev["tp"] == 0) & (df_dev["fp"] == 0)]
    starved_by_suite = starved.groupby("benchmark_suite").size()
    total_by_suite = df_dev.groupby("benchmark_suite").size()
    starv_df = pd.DataFrame({"Starved (All 0s)": starved_by_suite, "Total Models": total_by_suite}).fillna(0)
    starv_df["Percentage"] = (starv_df["Starved (All 0s)"] / starv_df["Total Models"]) * 100

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(starv_df.index, starv_df["Percentage"], color="#d95f02", edgecolor="black", lw=0.7, width=0.45)
    ax.set_ylabel("Percentage of Models Starved (%)", fontsize=10, fontweight="bold")
    ax.set_title("Threshold Starvation Impact (Dev Split, Default 0.5 Cutoff)\nModels Predicting Zero Depressed Participants (TP=0, FP=0)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xticklabels([s.replace(" ", "\n") for s in starv_df.index], fontsize=8.5)
    ax.set_ylim(0, max(starv_df["Percentage"].max() * 1.25, 40))

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(COMP_DIR / "threshold_starvation_analysis.png")
    plt.close(fig)

    print("Classification benchmark comparisons completed.")


from concurrent.futures import ProcessPoolExecutor


def main():
    print("=" * 80)
    print("PHASE 15: CLASSIFICATION VISUALIZER & BENCHMARK AUDIT")
    print("=" * 80)

    reg_path = OUTPUT_DIR / "model_registry.csv"
    if not reg_path.exists():
        print(f"Error: Master registry not found at {reg_path}")
        return

    df_reg = pd.read_csv(reg_path)
    df_cls = df_reg[df_reg["task"] == "classification"].copy()
    print(f"Loaded {len(df_cls)} classification models from registry.")

    # 1. Generate individual model visualizations in parallel across CPU cores
    records = df_cls.to_dict(orient="records")
    print(f"Generating individual visualizations across 8 parallel worker processes...")

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single_model_worker, records))

    success_count = sum(1 for r in results if r)
    print(f"Individual visualizations generated for {success_count} / {len(records)} models.")

    # 2. Benchmark Comparisons
    generate_benchmark_comparisons(df_cls)
    print("Classification visualizer completed successfully.")


if __name__ == "__main__":
    main()
