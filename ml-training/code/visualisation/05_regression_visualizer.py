"""
Phase 15 - Script 05: Regression Model Visualizer and Benchmark Comparisons.

Produces:
1. Individual Performance Visualizations (PNG only):
   - Actual vs Predicted Scatter Plots (with y=x parity line and CCC/RMSE)
   - Residual Plots (Residuals vs Predicted + Residual Histogram)
   - Error by Clinical Severity Tier (MAE & bias per PHQ-8 tier)
   - Combined 4-panel diagnostic card per model
   - Model card markdown summary (.md)
2. Benchmark / Cross-Model Comparisons:
   - Best CCC and RMSE by representation on Dev
   - Family / Suite comparison (Broad Acoustic vs Controlled vs CNN Classical vs CNN MLP vs CNN Temporal)
   - Generalization across splits (Train OOF vs Dev vs Test)
   - Systemic Severity Bias Analysis (clinical tier error analysis)

All figures saved exclusively in PNG format.
"""

import sys
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths
PROJECT_ROOT = Path("E:/Projects/BimodalPsycheAI")
ML_TRAINING = PROJECT_ROOT / "ml-training"
OUTPUT_DIR = ML_TRAINING / "outputs" / "model_visualizations_v1"
REG_DIR = OUTPUT_DIR / "regression"

# Subdirectories
ACT_PRED_DIR = REG_DIR / "actual_vs_predicted"
RESID_DIR = REG_DIR / "residuals"
SEV_DIR = REG_DIR / "error_by_severity"
CARD_DIR = REG_DIR / "individual_models"
COMP_DIR = REG_DIR / "comparisons"

for d in [ACT_PRED_DIR, RESID_DIR, SEV_DIR, CARD_DIR, COMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Global styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

TIER_NAMES = [
    "None (0-4)",
    "Mild (5-9)",
    "Moderate (10-14)",
    "Mod. Severe (15-19)",
    "Severe (20-24)",
]


def compute_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient."""
    if len(y_true) < 2:
        return 0.0
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-12:
        return 0.0
    return float(2 * covar / denom)


def get_tier(val):
    if val < 5:
        return "None (0-4)"
    elif val < 10:
        return "Mild (5-9)"
    elif val < 15:
        return "Moderate (10-14)"
    elif val < 20:
        return "Mod. Severe (15-19)"
    else:
        return "Severe (20-24)"


def load_regression_data(pred_path_str):
    """Loads prediction CSV for regression."""
    p_path = PROJECT_ROOT / pred_path_str
    if not p_path.exists():
        p_path = ML_TRAINING / pred_path_str
    if not p_path.exists():
        return None

    df = pd.read_csv(p_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return None

    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    residual = y_true - y_pred

    return y_true, y_pred, residual


def plot_actual_vs_predicted(y_true, y_pred, model_id, label, split, out_path, ccc_val, rmse_val, r2_val):
    """Generates actual vs predicted scatter plot with parity line."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)

    ax.scatter(y_true, y_pred, color="#2b5c8f", alpha=0.75, s=45, edgecolors="black", linewidth=0.5, label="Participants")
    
    # Parity line
    lim_max = max(np.max(y_true), np.max(y_pred), 24.0) * 1.05
    lim_min = min(np.min(y_true), np.min(y_pred), 0.0) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="Parity (y = x)")

    # Severity boundaries
    for b in [5, 10, 15, 20]:
        ax.axvline(b, color="#999999", linestyle=":", lw=0.8, alpha=0.7)

    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.set_xlabel("True PHQ-8 Score", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted PHQ-8 Score", fontsize=10, fontweight="bold")
    ax.set_title(f"Actual vs. Predicted: {label}\nSplit: {split} | CCC: {ccc_val:.3f} | RMSE: {rmse_val:.2f} | R²: {r2_val:.3f}",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_residuals(y_true, y_pred, residual, model_id, label, split, out_path):
    """Generates residual vs predicted and residual distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.5), dpi=150)

    # Left: Residual vs Predicted
    ax1.scatter(y_pred, residual, color="#d95f02", alpha=0.75, s=40, edgecolors="black", linewidth=0.5)
    ax1.axhline(0, color="black", linestyle="--", lw=1.2)
    ax1.set_xlabel("Predicted PHQ-8 Score", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Residual (True - Pred)", fontsize=9.5, fontweight="bold")
    ax1.set_title("Residuals vs. Predicted", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: Residual Histogram
    ax2.hist(residual, bins=12, color="#7570b3", edgecolor="white", alpha=0.8, density=True)
    ax2.axvline(0, color="black", linestyle="--", lw=1.2)
    ax2.set_xlabel("Residual (True - Pred)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=9.5, fontweight="bold")
    ax2.set_title(f"Residual Distribution (Mean: {np.mean(residual):.2f})", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(f"Residual Diagnostics: {label} (Split: {split})", fontsize=11, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_error_by_severity(y_true, y_pred, residual, model_id, label, split, out_path):
    """Generates MAE and Mean Residual breakdown across 5 clinical severity tiers."""
    df_t = pd.DataFrame({"true": y_true, "pred": y_pred, "resid": residual})
    df_t["tier"] = df_t["true"].apply(get_tier)

    tier_stats = []
    for t in TIER_NAMES:
        sub = df_t[df_t["tier"] == t]
        if not sub.empty:
            mae = mean_absolute_error(sub["true"], sub["pred"])
            bias = np.mean(sub["resid"])
            count = len(sub)
        else:
            mae, bias, count = 0.0, 0.0, 0
        tier_stats.append({"tier": t, "mae": mae, "bias": bias, "count": count})

    df_ts = pd.DataFrame(tier_stats)

    fig, ax1 = plt.subplots(figsize=(7, 4.8), dpi=150)
    x = np.arange(len(df_ts))
    w = 0.4

    bars = ax1.bar(x - w/2, df_ts["mae"], width=w, label="MAE", color="#2b5c8f", edgecolor="black", lw=0.6)
    bars2 = ax1.bar(x + w/2, df_ts["bias"], width=w, label="Mean Residual (Bias)", color="#d95f02", edgecolor="black", lw=0.6)
    ax1.axhline(0, color="black", linestyle="-", lw=0.8)

    labels = [f"{t}\n(n={c})" for t, c in zip(df_ts["tier"], df_ts["count"])]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Score Points", fontsize=9.5, fontweight="bold")
    ax1.set_title(f"Error by Clinical Severity Tier: {label}\nSplit: {split}", fontsize=10.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_combined_regression_card(y_true, y_pred, residual, model_row, out_path):
    """Generates 4-panel comprehensive diagnostic card."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=150)

    # Panel A: Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, color="#2b5c8f", alpha=0.75, s=40, edgecolors="black", lw=0.5)
    lim_max = max(np.max(y_true), np.max(y_pred), 24.0) * 1.05
    lim_min = min(np.min(y_true), np.min(y_pred), 0.0) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="Parity (y = x)")
    for b in [5, 10, 15, 20]:
        ax.axvline(b, color="#999999", linestyle=":", lw=0.7)
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.set_title("A. Actual vs. Predicted (PHQ-8)", fontsize=11, fontweight="bold")
    ax.set_xlabel("True Score", fontsize=9.5)
    ax.set_ylabel("Predicted Score", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residual, color="#d95f02", alpha=0.75, s=40, edgecolors="black", lw=0.5)
    ax.axhline(0, color="black", linestyle="--", lw=1.2)
    ax.set_title("B. Residuals vs. Predicted", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Score", fontsize=9.5)
    ax.set_ylabel("Residual (True - Pred)", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Residual Distribution
    ax = axes[1, 0]
    ax.hist(residual, bins=12, color="#7570b3", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", lw=1.2)
    ax.set_title(f"C. Residual Distribution (Mean: {np.mean(residual):.2f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual (True - Pred)", fontsize=9.5)
    ax.set_ylabel("Count", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel D: MAE by Clinical Severity Tier
    ax = axes[1, 1]
    df_t = pd.DataFrame({"true": y_true, "pred": y_pred, "resid": residual})
    df_t["tier"] = df_t["true"].apply(get_tier)
    maes = [mean_absolute_error(df_t[df_t["tier"] == t]["true"], df_t[df_t["tier"] == t]["pred"]) if not df_t[df_t["tier"] == t].empty else 0.0 for t in TIER_NAMES]
    counts = [len(df_t[df_t["tier"] == t]) for t in TIER_NAMES]
    x_pos = np.arange(len(TIER_NAMES))
    bars = ax.bar(x_pos, maes, color="#1b9e77", edgecolor="black", lw=0.6, width=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{t[:7]}\n(n={c})" for t, c in zip(TIER_NAMES, counts)], fontsize=8.5)
    ax.set_title("D. MAE Across Clinical Severity Tiers", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE (Points)", fontsize=9.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    suite = model_row["benchmark_suite"]
    arch = model_row["architecture"]
    rep = model_row["representation"]
    pool = model_row["pooling"]
    split = model_row["evaluation_split"]
    ccc_v = model_row["ccc"]
    rmse_v = model_row["rmse"]
    mae_v = model_row["mae"]
    r2_v = model_row["r2"]

    plt.suptitle(
        f"Performance Card: {arch} | {rep} ({pool}) | Split: {split}\n"
        f"Suite: {suite} | CCC: {ccc_v:.3f} | RMSE: {rmse_v:.2f} | MAE: {mae_v:.2f} | R²: {r2_v:.3f}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def generate_regression_markdown(row, out_path):
    """Generates individual regression model markdown card."""
    model_id = row["model_id"]
    content = f"""# Model Card: `{model_id}`

## Model Summary
- **Benchmark Suite**: {row['benchmark_suite']}
- **Task**: Continuous Depression Severity Regression (`PHQ_Score`)
- **Architecture**: `{row['architecture']}`
- **Feature Representation**: `{row['representation']}`
- **Pooling Strategy**: `{row['pooling']}`
- **Evaluation Split**: `{row['evaluation_split']}`
- **Participant Count**: {row['participant_count']}

## Quantitative Metrics
| Metric | Verified Value | Interpretation |
| :--- | :--- | :--- |
| **CCC** | **{row['ccc']:.4f}** | Concordance correlation with clinical ground truth |
| **RMSE** | **{row['rmse']:.4f}** | Root mean square error |
| **MAE** | **{row['mae']:.4f}** | Mean absolute error |
| **R²** | **{row['r2']:.4f}** | Coefficient of determination ({'negative, failed to outperform mean predictor' if row['r2'] < 0 else 'positive variance explained'}) |

## Diagnostic Interpretation
"""
    if row["r2"] < 0:
        content += f"- **Negative R² ({row['r2']:.3f})**: Consistent with severe regression attenuation observed on E-DAIC audio benchmarks; predicting the sample mean achieves lower MSE than the model's feature-driven projections.\n"
    if row["ccc"] > 0.3:
        content += f"- **Strong Concordance ({row['ccc']:.3f})**: Demonstrates notable alignment with the clinical severity trajectory.\n"
    else:
        content += f"- **Baseline Concordance ({row['ccc']:.3f})**: Modest agreement across the severity distribution.\n"

    content += f"""
## Artifact References
- **Prediction Artifact**: `{row['prediction_artifact']}`
- **Config Artifact**: `{row['config_artifact'] if pd.notna(row['config_artifact']) else "None"}`
- **Diagnostic Visual Card**: `regression/individual_models/{model_id}__card.png`
"""
    out_path.write_text(content, encoding="utf-8")


def process_single_reg_model_worker(row_dict):
    """Worker function for multiprocessing."""
    model_id = row_dict["model_id"]
    pred_art = row_dict["prediction_artifact"]
    split = row_dict["evaluation_split"]
    label = f"{row_dict['architecture']} ({row_dict['representation']})"

    data = load_regression_data(pred_art)
    if data is None:
        return False
    y_true, y_pred, residual = data

    ccc_v = row_dict["ccc"]
    rmse_v = row_dict["rmse"]
    r2_v = row_dict["r2"]

    # 1. Actual vs Predicted
    act_path = ACT_PRED_DIR / f"{model_id}__act_vs_pred.png"
    plot_actual_vs_predicted(y_true, y_pred, model_id, label, split, act_path, ccc_v, rmse_v, r2_v)

    # 2. Residuals
    res_path = RESID_DIR / f"{model_id}__residuals.png"
    plot_residuals(y_true, y_pred, residual, model_id, label, split, res_path)

    # 3. Error by Severity
    sev_path = SEV_DIR / f"{model_id}__severity_error.png"
    plot_error_by_severity(y_true, y_pred, residual, model_id, label, split, sev_path)

    # 4. Combined Card & Markdown
    card_path = CARD_DIR / f"{model_id}__card.png"
    plot_combined_regression_card(y_true, y_pred, residual, row_dict, card_path)

    md_path = CARD_DIR / f"{model_id}.md"
    generate_regression_markdown(row_dict, md_path)

    return True


def generate_regression_benchmark_comparisons(df_reg):
    """Generates cross-model benchmark comparisons for regression."""
    print("--- Generating Regression Benchmark Comparisons ---")
    df_dev = df_reg[df_reg["evaluation_split"] == "dev"].copy()

    # 1. Best CCC by Representation on Dev
    rep_best_ccc = (
        df_dev.groupby(["representation", "pooling", "benchmark_suite"])["ccc"]
        .max()
        .reset_index()
        .sort_values("ccc", ascending=False)
    )
    rep_best_ccc["label"] = rep_best_ccc["representation"] + " (" + rep_best_ccc["pooling"] + ")"

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    bars = ax.barh(np.arange(len(rep_best_ccc)), rep_best_ccc["ccc"], color="#7570b3", edgecolor="black", lw=0.6)
    ax.set_yticks(np.arange(len(rep_best_ccc)))
    ax.set_yticklabels(rep_best_ccc["label"], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="-", lw=0.8)
    ax.set_xlabel("Best Validation Concordance Correlation Coefficient (CCC, Dev Split, n=56)", fontsize=10, fontweight="bold")
    ax.set_title("Benchmark Comparison: Best CCC Across All Feature Representations (Dev Split)",
                 fontsize=11.5, fontweight="bold", pad=12)

    for bar in bars:
        w = bar.get_width()
        offset = 0.008 if w >= 0 else -0.008
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2.0, f"{w:.3f}", ha=ha, va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(min(rep_best_ccc["ccc"].min() * 1.15, -0.05), max(rep_best_ccc["ccc"].max() * 1.2, 0.4))
    plt.tight_layout()
    plt.savefig(COMP_DIR / "benchmark_ccc_by_representation_dev.png")
    plt.close(fig)

    # 2. Best RMSE by Representation on Dev
    # Filter out extreme outliers for meaningful comparison
    df_valid_rmse = df_dev[df_dev["rmse"] < 25]
    rep_best_rmse = (
        df_valid_rmse.groupby(["representation", "pooling", "benchmark_suite"])["rmse"]
        .min()
        .reset_index()
        .sort_values("rmse", ascending=True)
    )
    rep_best_rmse["label"] = rep_best_rmse["representation"] + " (" + rep_best_rmse["pooling"] + ")"

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    bars = ax.barh(np.arange(len(rep_best_rmse)), rep_best_rmse["rmse"], color="#1b9e77", edgecolor="black", lw=0.6)
    ax.set_yticks(np.arange(len(rep_best_rmse)))
    ax.set_yticklabels(rep_best_rmse["label"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Best Validation RMSE (Lower is Better, Dev Split, n=56)", fontsize=10, fontweight="bold")
    ax.set_title("Benchmark Comparison: Best RMSE Across All Feature Representations (Dev Split)",
                 fontsize=11.5, fontweight="bold", pad=12)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2.0, f"{w:.2f}", ha="left", va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(0, max(rep_best_rmse["rmse"].max() * 1.15, 8.0))
    plt.tight_layout()
    plt.savefig(COMP_DIR / "benchmark_rmse_by_representation_dev.png")
    plt.close(fig)

    # 3. Family / Suite Comparison
    suite_best = []
    for suite, g in df_dev.groupby("benchmark_suite"):
        top_m = g.sort_values("ccc", ascending=False).iloc[0]
        suite_best.append({
            "suite": suite,
            "best_model": f"{top_m['architecture']}\n({top_m['representation']})",
            "ccc": top_m["ccc"],
            "rmse": top_m["rmse"],
            "mae": top_m["mae"],
            "r2": top_m["r2"],
        })
    df_suite = pd.DataFrame(suite_best).sort_values("ccc", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    x = np.arange(len(df_suite))
    w = 0.35

    # CCC
    bars_ccc = ax1.bar(x, df_suite["ccc"], width=w, color="#7570b3", edgecolor="black", lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_suite["suite"] + "\n" + df_suite["best_model"], fontsize=8)
    ax1.set_ylabel("CCC (Higher is Better)", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Best Concordance Correlation (CCC)", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    for bar in bars_ccc:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # RMSE & MAE
    bars_rmse = ax2.bar(x - w/2, df_suite["rmse"], width=w, label="RMSE", color="#1b9e77", edgecolor="black", lw=0.6)
    bars_mae = ax2.bar(x + w/2, df_suite["mae"], width=w, label="MAE", color="#d95f02", edgecolor="black", lw=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_suite["suite"] + "\n" + df_suite["best_model"], fontsize=8)
    ax2.set_ylabel("Error Points (Lower is Better)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Best Error Metrics (RMSE & MAE)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Cross-Family Benchmark Comparison: Best Regression Models by Suite (Dev Split, n=56)",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(COMP_DIR / "family_comparison_dev.png")
    plt.close(fig)

    # 4. Split Generalization Comparison (Train OOF vs Dev vs Test)
    ctrl_models = df_reg[df_reg["benchmark_suite"].isin(["Controlled Classical Benchmark", "Official Final Test Benchmark"])]
    if not ctrl_models.empty:
        split_piv = ctrl_models.pivot_table(index=["representation", "architecture"], columns="evaluation_split", values="ccc")
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
            ax.set_ylabel("CCC", fontsize=10, fontweight="bold")
            ax.set_title("Regression Generalization Across Splits: Train OOF vs. Dev vs. Test", fontsize=11.5, fontweight="bold", pad=12)
            ax.legend(loc="upper right", fontsize=9)
            ax.axhline(0, color="black", lw=0.8)
            ax.grid(True, axis="y", linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(COMP_DIR / "split_comparison_train_dev_test.png")
            plt.close(fig)

    # 5. Systemic Severity Bias Analysis (Mean Residual across Tiers)
    # Collect predictions from top models across suites on Dev
    tier_bias_records = []
    for suite, g in df_dev.groupby("benchmark_suite"):
        top_m = g.sort_values("ccc", ascending=False).iloc[0]
        data = load_regression_data(top_m["prediction_artifact"])
        if data is None:
            continue
        y_true, y_pred, residual = data
        df_p = pd.DataFrame({"true": y_true, "pred": y_pred, "resid": residual})
        df_p["tier"] = df_p["true"].apply(get_tier)
        for t in TIER_NAMES:
            sub = df_p[df_p["tier"] == t]
            if not sub.empty:
                tier_bias_records.append({
                    "suite": suite,
                    "tier": t,
                    "mean_residual": np.mean(sub["resid"]),
                    "mae": mean_absolute_error(sub["true"], sub["pred"]),
                })

    df_tb = pd.DataFrame(tier_bias_records)
    if not df_tb.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        sns.barplot(data=df_tb, x="tier", y="mean_residual", hue="suite", palette="muted", ax=ax, edgecolor="black", lw=0.6)
        ax.axhline(0, color="black", linestyle="--", lw=1.2)
        ax.set_xlabel("Clinical Severity Tier (True PHQ-8 Score)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Residual (True - Predicted)", fontsize=10, fontweight="bold")
        ax.set_title("Systemic Psychiatric Regression Bias Across Model Families (Dev Split)\n(Negative: Over-Predicting Severity | Positive: Under-Predicting Severity)",
                     fontsize=11, fontweight="bold", pad=12)
        ax.legend(title="Benchmark Suite", fontsize=8.5, title_fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(COMP_DIR / "systemic_severity_bias_analysis.png")
        plt.close(fig)

    print("Regression benchmark comparisons completed.")


def main():
    print("=" * 80)
    print("PHASE 15: REGRESSION VISUALIZER & BENCHMARK AUDIT")
    print("=" * 80)

    reg_path = OUTPUT_DIR / "model_registry.csv"
    if not reg_path.exists():
        print(f"Error: Master registry not found at {reg_path}")
        return

    df_reg = pd.read_csv(reg_path)
    df_r = df_reg[df_reg["task"] == "regression"].copy()
    print(f"Loaded {len(df_r)} regression models from registry.")

    # 1. Generate individual model visualizations in parallel
    records = df_r.to_dict(orient="records")
    print(f"Generating individual regression visualizations across 8 parallel worker processes...")

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single_reg_model_worker, records))

    success_count = sum(1 for r in results if r)
    print(f"Individual visualizations generated for {success_count} / {len(records)} models.")

    # 2. Benchmark Comparisons
    generate_regression_benchmark_comparisons(df_r)
    print("Regression visualizer completed successfully.")


if __name__ == "__main__":
    main()
