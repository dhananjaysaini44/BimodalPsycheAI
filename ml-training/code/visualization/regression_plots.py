"""
Visualization Utilities for E-DAIC Regression Benchmark.

Generates:
- Predicted vs. Actual PHQ_Score scatter plots with y=x identity diagonal
- Residual distributions (histograms) and Residuals vs. Actual plots
- Model comparison charts across CCC, RMSE, MAE, R^2
- Cross-representation and pooling performance heatmaps
- Training CV vs. Development performance comparisons
- Repeated-CV variability error-bar plots
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import REG_COMPARISONS_DIR, REGRESSION_OUTPUT_DIR


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment_id: str,
    ccc: float,
    rmse: float,
    mae: float,
    output_path: str,
    protocol: str = "cv5",
):
    """Generates a scatter plot of predicted vs actual PHQ_Score with identity line."""
    plt.figure(figsize=(7, 6))
    
    # Scatter points
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolors="none", s=50, c="#1f77b4", label="Participants")
    
    # Ideal identity line y = x
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)), 0.0)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)), 24.0)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal (y = x)")
    
    # Linear fit trendline
    if len(y_true) > 1 and np.var(y_pred) > 1e-6:
        m, b = np.polyfit(y_true, y_pred, 1)
        x_trend = np.linspace(min_val, max_val, 100)
        plt.plot(x_trend, m * x_trend + b, "b-.", alpha=0.8, linewidth=1.5, label=f"Trend (slope={m:.2f})")

    protocol_label = "5-Fold Out-of-Fold (Train n=163)" if "cv" in protocol else "Development (Held-out n=56)"
    plt.title(f"Predicted vs. Actual PHQ_Score\n{experiment_id}\n[{protocol_label}]", fontsize=11, fontweight="bold")
    plt.xlabel("Actual Ground Truth PHQ_Score (0–24)", fontsize=10)
    plt.ylabel("Predicted PHQ_Score", fontsize=10)
    
    # Annotate metrics
    textstr = f"CCC:  {ccc:.3f}\nRMSE: {rmse:.2f}\nMAE:  {mae:.2f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc")
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment="top", bbox=props)

    plt.xlim(min_val - 1, max_val + 1)
    plt.ylim(min_val - 1, max_val + 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment_id: str,
    output_path: str,
    protocol: str = "cv5",
):
    """Generates side-by-side plots: (Left) Residuals vs. Actual, (Right) Residual Histogram."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Residuals vs Actual
    axes[0].scatter(y_true, residuals, color="#2ca02c", alpha=0.75, s=45)
    axes[0].axhline(0.0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residuals vs. Actual Score", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Actual PHQ_Score", fontsize=10)
    axes[0].set_ylabel("Residual (Actual - Predicted)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color="#9467bd", bins=15)
    axes[1].axvline(0.0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Residual Distribution (Mean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f})",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual (Actual - Predicted)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    protocol_label = "5-Fold CV (Train n=163)" if "cv" in protocol else "Development Split (n=56)"
    fig.suptitle(f"Residual Analysis: {experiment_id} [{protocol_label}]", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_regression_comparison_plots(
    df_results: pd.DataFrame,
    output_dir: str = str(REG_COMPARISONS_DIR),
):
    """Generates all macro benchmark comparison figures across models, representations, and splits."""
    os.makedirs(output_dir, exist_ok=True)
    df_success = df_results[df_results["status"] == "SUCCESS"].copy()

    # Split CV vs Dev
    df_cv = df_success[df_success["evaluation_protocol"] == "cv5"]
    df_dev = df_success[df_success["evaluation_protocol"] == "dev"]

    # 1. CCC across Model Families (CV vs Dev)
    plt.figure(figsize=(12, 6))
    model_order = df_cv.groupby("model_name")["ccc"].mean().sort_values(ascending=False).index.tolist()
    
    df_combined_plot = df_success[df_success["model_name"].isin(model_order)].copy()
    sns.boxplot(
        data=df_combined_plot,
        x="model_name",
        y="ccc",
        hue="evaluation_protocol",
        order=model_order,
        palette={"cv5": "#1f77b4", "dev": "#ff7f0e"}
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.title("Concordance Correlation Coefficient (CCC) by Model Family (CV vs Dev)", fontsize=13, fontweight="bold")
    plt.xlabel("Model Family", fontsize=11)
    plt.ylabel("CCC (Agreement Metric)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Protocol", labels=["5-Fold CV (Train)", "Development"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ccc_comparison_by_model.png"), dpi=300)
    plt.close()

    # 2. RMSE Comparison Across Models
    plt.figure(figsize=(12, 6))
    rmse_order = df_cv.groupby("model_name")["rmse"].mean().sort_values().index.tolist()
    sns.boxplot(
        data=df_combined_plot,
        x="model_name",
        y="rmse",
        hue="evaluation_protocol",
        order=rmse_order,
        palette={"cv5": "#2ca02c", "dev": "#d62728"}
    )
    dummy_rmse = df_cv[df_cv["model_name"] == "dummy"]["rmse"].mean()
    plt.axhline(dummy_rmse, color="gray", linestyle=":", linewidth=1.5, label=f"Dummy Baseline ({dummy_rmse:.2f})")
    plt.title("Root Mean Squared Error (RMSE) by Model Family (Lower is Better)", fontsize=13, fontweight="bold")
    plt.xlabel("Model Family", fontsize=11)
    plt.ylabel("RMSE", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Protocol")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_comparison_by_model.png"), dpi=300)
    plt.close()

    # 3. MAE Comparison Across Models
    plt.figure(figsize=(12, 6))
    mae_order = df_cv.groupby("model_name")["mae"].mean().sort_values().index.tolist()
    sns.boxplot(
        data=df_combined_plot,
        x="model_name",
        y="mae",
        hue="evaluation_protocol",
        order=mae_order,
        palette={"cv5": "#9467bd", "dev": "#8c564b"}
    )
    plt.title("Mean Absolute Error (MAE) by Model Family (Lower is Better)", fontsize=13, fontweight="bold")
    plt.xlabel("Model Family", fontsize=11)
    plt.ylabel("MAE", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_comparison_by_model.png"), dpi=300)
    plt.close()

    # 4. Pooling Strategy Heatmap (Cross-Validation CCC)
    pivot_cv = df_cv[df_cv["model_name"] != "dummy"].pivot_table(
        index="pooling_strategy",
        columns="representation",
        values="ccc",
        aggfunc="mean"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_cv, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Mean CCC"})
    plt.title("Average Cross-Validation CCC by Representation & Pooling", fontsize=12, fontweight="bold")
    plt.xlabel("Acoustic Representation", fontsize=10)
    plt.ylabel("Pooling Strategy", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_ccc_heatmap.png"), dpi=300)
    plt.close()

    # 5. Pooling Strategy Heatmap (Development CCC)
    pivot_dev = df_dev[df_dev["model_name"] != "dummy"].pivot_table(
        index="pooling_strategy",
        columns="representation",
        values="ccc",
        aggfunc="mean"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_dev, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Mean CCC"})
    plt.title("Average Development CCC by Representation & Pooling", fontsize=12, fontweight="bold")
    plt.xlabel("Acoustic Representation", fontsize=10)
    plt.ylabel("Pooling Strategy", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dev_ccc_heatmap.png"), dpi=300)
    plt.close()

    # 6. Training CV CCC vs. Development CCC Scatter Plot
    merged = pd.merge(
        df_cv[["experiment_id", "representation", "pooling_strategy", "model_name", "ccc", "rmse"]],
        df_dev[["representation", "pooling_strategy", "model_name", "ccc", "rmse"]],
        on=["representation", "pooling_strategy", "model_name"],
        suffixes=("_cv", "_dev")
    )
    plt.figure(figsize=(8, 7))
    sns.scatterplot(
        data=merged[merged["model_name"] != "dummy"],
        x="ccc_cv", y="ccc_dev", hue="model_name", style="representation", s=70, alpha=0.85
    )
    lim_min = min(merged["ccc_cv"].min(), merged["ccc_dev"].min(), -0.1)
    lim_max = max(merged["ccc_cv"].max(), merged["ccc_dev"].max(), 0.5)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.6, label="Ideal Parity (y=x)")
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    plt.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    plt.title("Generalization Diagnostic: 5-Fold CV CCC vs. Development CCC", fontsize=12, fontweight="bold")
    plt.xlabel("Training 5-Fold Out-of-Fold CCC", fontsize=10)
    plt.ylabel("Development Split CCC", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_vs_dev_ccc_generalization.png"), dpi=300)
    plt.close()
    print(f"Generated all regression comparison plots in {output_dir}")
