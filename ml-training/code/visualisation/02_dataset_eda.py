"""
Phase 15 - Script 02: Dataset EDA and Dedicated Mean vs. Mean+Std Analysis.

Performs exhaustive exploratory data analysis on all 28 feature datasets, target distributions,
split demographics, and produces dedicated pairwise analyses of mean vs. mean+std pooling.

Outputs to:
- outputs/model_visualizations_v1/data_exploration/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path("E:/Projects/BimodalPsycheAI")
ML_TRAINING = PROJECT_ROOT / "ml-training"
OUTPUT_DIR = ML_TRAINING / "outputs" / "model_visualizations_v1"
DATASET_DIR = ML_TRAINING / "dataset"
EDAIC_WOZ = ML_TRAINING / "edaic-woz"
EDA_DIR = OUTPUT_DIR / "data_exploration"

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def analyze_targets_and_splits():
    print("--- Visualizing Target Distributions and Split Demographics ---")
    train_split = pd.read_csv(EDAIC_WOZ / "labels" / "train_split.csv")
    dev_split = pd.read_csv(EDAIC_WOZ / "labels" / "dev_split.csv")
    test_split = pd.read_csv(EDAIC_WOZ / "labels" / "test_split.csv")

    train_split["split"] = "Train"
    dev_split["split"] = "Development"
    test_split["split"] = "Official Test"

    # 1. Participant count by split
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    split_counts = pd.Series({
        "Train": len(train_split),
        "Development": len(dev_split),
        "Official Test": len(test_split),
    })
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    bars = ax.bar(split_counts.index, split_counts.values, color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval} ({yval/275*100:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("E-DAIC Participant Cohort Distribution by Official Split (N = 275)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Participants", fontsize=10)
    ax.set_ylim(0, 190)
    plt.tight_layout()
    p_splits = EDA_DIR / "dataset_coverage" / "participant_splits_breakdown.png"
    plt.savefig(p_splits)
    plt.close()

    # 2. Continuous PHQ-8 score distribution (Train vs Dev)
    combined_td = pd.concat([train_split[["Participant_ID", "PHQ_Score", "split"]], dev_split[["Participant_ID", "PHQ_Score", "split"]]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Histogram & KDE
    for split_name, col, col_code in [("Train", "#2b5c8f", train_split), ("Development", "#d95f02", dev_split)]:
        scores = col_code["PHQ_Score"].dropna()
        sns.histplot(scores, kde=True, label=f"{split_name} (n={len(scores)}, mean={scores.mean():.1f}, med={scores.median():.0f})",
                     color=col, ax=ax1, bins=12, stat="density", alpha=0.45)
    ax1.axvline(10, color="crimson", linestyle="--", linewidth=1.5, label="Clinical Cutoff (PHQ-8 >= 10)")
    ax1.set_title("Continuous PHQ-8 Score Density by Split", fontsize=11, fontweight="bold")
    ax1.set_xlabel("PHQ-8 Total Score (0 - 24)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Clinical Severity Bands
    def severity_band(score):
        if score < 5: return "0-4 None/Minimal"
        elif score < 10: return "5-9 Mild"
        elif score < 15: return "10-14 Moderate"
        elif score < 20: return "15-19 Mod. Severe"
        else: return "20-24 Severe"

    combined_td["severity"] = combined_td["PHQ_Score"].apply(severity_band)
    order = ["0-4 None/Minimal", "5-9 Mild", "10-14 Moderate", "15-19 Mod. Severe", "20-24 Severe"]
    sev_df = combined_td.groupby(["split", "severity"]).size().unstack(fill_value=0)[order]
    sev_pct = sev_df.div(sev_df.sum(axis=1), axis=0) * 100

    sev_pct.T.plot(kind="bar", ax=ax2, color=["#d95f02", "#2b5c8f"], width=0.6, edgecolor="black", linewidth=0.6)
    ax2.set_title("Depression Severity Band Proportions (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Clinical Severity Category", fontsize=10)
    ax2.set_ylabel("Percentage of Cohort (%)", fontsize=10)
    ax2.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
    ax2.legend(["Development Split", "Training Split"], fontsize=9)

    plt.tight_layout()
    p_phq = EDA_DIR / "target_distributions" / "phq8_continuous_distribution_by_split.png"
    plt.savefig(p_phq)
    plt.close()

    # 3. Binary class balance
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    train_bin = train_split["PHQ_Binary"].value_counts(normalize=True) * 100
    dev_bin = dev_split["PHQ_Binary"].value_counts(normalize=True) * 100

    bin_df = pd.DataFrame({
        "Train Split (n=163)": [train_bin.get(0, 0), train_bin.get(1, 0)],
        "Dev Split (n=56)": [dev_bin.get(0, 0), dev_bin.get(1, 0)],
    }, index=["Non-Depressed (PHQ < 10)", "Depressed (PHQ >= 10)"])

    bin_df.plot(kind="bar", ax=ax, color=["#2b5c8f", "#d95f02"], width=0.55, edgecolor="black", linewidth=0.7)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("PHQ-8 Binary Depression Prevalence by Split", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage of Cohort (%)", fontsize=10)
    ax.set_ylim(0, 95)
    ax.set_xticklabels(bin_df.index, rotation=0, fontsize=9.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p_bin = EDA_DIR / "target_distributions" / "phq8_binary_class_balance.png"
    plt.savefig(p_bin)
    plt.close()
    print("Target distributions saved.")


def analyze_feature_dimensions():
    print("--- Analyzing Feature Dimensionality Across 28 Representations ---")
    cov_path = OUTPUT_DIR / "data_coverage_report.csv"
    if not cov_path.exists():
        return
    df_cov = pd.read_csv(cov_path)

    # Sort by feature dimension
    df_sorted = df_cov.sort_values("feature_dim").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    colors = ["#7570b3" if "cnn" in d else "#2b5c8f" for d in df_sorted["dataset_id"]]
    bars = ax.barh(df_sorted["dataset_id"], df_sorted["feature_dim"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_title("Feature Dimensionality Across All 28 Representations (Log Scale)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Features (Log Scale)", fontsize=11)
    ax.set_ylabel("Dataset Representation", fontsize=10)

    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.12, bar.get_y() + bar.get_height() / 2.0, f"{int(w):,}", ha="left", va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlim(10, 25000)
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", edgecolor="black", label="Classical Acoustic (23 - 400 dims)"),
        Patch(facecolor="#7570b3", edgecolor="black", label="Deep CNN Embeddings (1,920 - 8,192 dims)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    p_dim = EDA_DIR / "dataset_coverage" / "feature_dimensionality_comparison.png"
    plt.savefig(p_dim)
    plt.close()
    print("Dimensionality comparison saved.")


def dedicated_mean_vs_mean_std_analysis():
    print("--- Executing Dedicated Mean vs. Mean+Std Deep Dive ---")
    out_dir = EDA_DIR / "mean_vs_mean_std"

    pairs = [
        ("raw_egemaps", DATASET_DIR / "raw_egemaps" / "mean.csv", DATASET_DIR / "raw_egemaps" / "mean_std.csv", 23, 46),
        ("raw_mfcc", DATASET_DIR / "raw_mfcc" / "mean.csv", DATASET_DIR / "raw_mfcc" / "mean_std.csv", 39, 78),
        ("boaw_egemaps", DATASET_DIR / "boaw_egemaps" / "mean.csv", DATASET_DIR / "boaw_egemaps" / "mean_std.csv", 100, 200),
        ("boaw_mfcc", DATASET_DIR / "boaw_mfcc" / "mean.csv", DATASET_DIR / "boaw_mfcc" / "mean_std.csv", 100, 200),
        ("densenet201", DATASET_DIR / "deep_features_v1" / "pooled" / "densenet201__mean.csv",
         DATASET_DIR / "deep_features_v1" / "pooled" / "densenet201__mean_std.csv", 1920, 3840),
        ("vgg16", DATASET_DIR / "deep_features_v1" / "pooled" / "vgg16__mean.csv",
         DATASET_DIR / "deep_features_v1" / "pooled" / "vgg16__mean_std.csv", 4096, 8192),
    ]

    pair_audit_records = []
    correlation_distribution_data = {}

    for name, p_mean, p_std, exp_mean_dim, exp_std_dim in pairs:
        if not p_mean.exists() or not p_std.exists():
            continue

        df_m = pd.read_csv(p_mean)
        df_ms = pd.read_csv(p_std)

        # Detect PID column
        pid_col = "Participant_ID" if "Participant_ID" in df_m.columns else "participant_id"
        feat_cols_m = [c for c in df_m.columns if c != pid_col and "phq" not in c.lower()]
        feat_cols_ms = [c for c in df_ms.columns if c != pid_col and "phq" not in c.lower()]

        # Align on Participant ID
        merged = pd.merge(df_m[[pid_col] + feat_cols_m], df_ms[[pid_col] + feat_cols_ms], on=pid_col, suffixes=("_mean_single", "_mean_std_joint"))
        
        # Verify schema: does mean_std contain the exact mean values in its first half?
        mean_half_cols = feat_cols_ms[:len(feat_cols_m)]
        std_half_cols = feat_cols_ms[len(feat_cols_m):]

        m_vals_from_single = df_m[feat_cols_m].to_numpy()
        m_vals_from_joint = df_ms[mean_half_cols].to_numpy()

        # Check numerical equality
        is_exact_match = np.allclose(m_vals_from_single, m_vals_from_joint, atol=1e-5)
        max_diff = float(np.max(np.abs(m_vals_from_single - m_vals_from_joint)))

        # Compute correlations between mean_k and std_k across all dimensions
        corr_list = []
        for k in range(len(feat_cols_m)):
            col_m_name = mean_half_cols[k]
            col_s_name = std_half_cols[k]
            v_m = df_ms[col_m_name].to_numpy()
            v_s = df_ms[col_s_name].to_numpy()
            if np.std(v_m) > 1e-7 and np.std(v_s) > 1e-7:
                r, _ = stats.pearsonr(v_m, v_s)
                if not np.isnan(r):
                    corr_list.append(r)

        correlation_distribution_data[name] = corr_list
        med_r = float(np.median(corr_list)) if corr_list else np.nan
        pct_high_corr = float(np.mean(np.abs(corr_list) > 0.8) * 100) if corr_list else 0.0

        pair_audit_records.append({
            "representation_pair": name,
            "mean_path": str(p_mean.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "mean_std_path": str(p_std.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "mean_dim": len(feat_cols_m),
            "mean_std_dim": len(feat_cols_ms),
            "expected_mean_dim": exp_mean_dim,
            "expected_mean_std_dim": exp_std_dim,
            "dim_matches_expected": (len(feat_cols_m) == exp_mean_dim and len(feat_cols_ms) == exp_std_dim),
            "mean_components_exact_match": is_exact_match,
            "max_absolute_mean_difference": round(max_diff, 8),
            "median_pearson_mean_vs_std": round(med_r, 4),
            "pct_dimensions_abs_r_gt_0_8": round(pct_high_corr, 2),
            "total_dimensions_evaluated": len(corr_list),
        })

    df_pair_audit = pd.DataFrame(pair_audit_records)
    df_pair_audit.to_csv(out_dir / "mean_vs_mean_std_pair_audit.csv", index=False)

    # Plot correlation distribution across dimensions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
    axes = axes.flatten()
    for idx, (rep_name, corr_vals) in enumerate(correlation_distribution_data.items()):
        ax = axes[idx]
        if corr_vals:
            sns.histplot(corr_vals, kde=True, ax=ax, color="#2b5c8f", bins=25, edgecolor="black", linewidth=0.5)
            med = np.median(corr_vals)
            ax.axvline(med, color="crimson", linestyle="--", linewidth=1.5, label=f"Median r = {med:.2f}")
            ax.set_title(f"{rep_name} (Dim: {len(corr_vals)})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Pearson r(Mean_k, Std_k)", fontsize=9.5)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(fontsize=8.5)
            ax.set_xlim(-1, 1)

    plt.suptitle("Inter-Component Correlation: r(Mean, Std) Across Dimensions", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    p_corr = out_dir / "mean_vs_std_component_correlations.png"
    plt.savefig(p_corr)
    plt.close()

    # Model Performance Comparison: Mean vs. Mean_Std
    reg_path = OUTPUT_DIR / "model_registry.csv"
    if reg_path.exists():
        df_reg = pd.read_csv(reg_path)
        # Filter for models where both mean and mean_std were evaluated under the same split and architecture
        df_paired = df_reg[df_reg["pooling"].isin(["mean", "mean_std"]) & (df_reg["evaluation_split"] == "dev")].copy()

        # Classification comparison (F1 & ROC-AUC)
        c_models = df_paired[df_paired["task"] == "classification"]
        if not c_models.empty:
            pivot_f1 = c_models.pivot_table(index=["representation", "architecture"], columns="pooling", values="f1")
            pivot_auc = c_models.pivot_table(index=["representation", "architecture"], columns="pooling", values="roc_auc")

            if not pivot_f1.empty and "mean" in pivot_f1 and "mean_std" in pivot_f1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
                # Scatter of Mean vs Mean_std F1
                ax1.scatter(pivot_f1["mean"], pivot_f1["mean_std"], color="#2b5c8f", s=65, edgecolors="black", linewidth=0.7)
                lims = [0, max(pivot_f1.max().max(), 0.6) * 1.1]
                ax1.plot(lims, lims, "r--", linewidth=1.2, label="Parity (y = x)")
                ax1.set_title("Dev F1-Score: Mean vs. Mean_Std", fontsize=11, fontweight="bold")
                ax1.set_xlabel("Mean Pooling F1", fontsize=10)
                ax1.set_ylabel("Mean + Std Pooling F1", fontsize=10)
                ax1.legend(fontsize=9)

                # Scatter of Mean vs Mean_std ROC-AUC
                ax2.scatter(pivot_auc["mean"], pivot_auc["mean_std"], color="#d95f02", s=65, edgecolors="black", linewidth=0.7)
                lims_auc = [0.4, 0.8]
                ax2.plot(lims_auc, lims_auc, "r--", linewidth=1.2, label="Parity (y = x)")
                ax2.set_title("Dev ROC-AUC: Mean vs. Mean_Std", fontsize=11, fontweight="bold")
                ax2.set_xlabel("Mean Pooling ROC-AUC", fontsize=10)
                ax2.set_ylabel("Mean + Std Pooling ROC-AUC", fontsize=10)
                ax2.legend(fontsize=9)

                plt.tight_layout()
                p_c_perf = out_dir / "mean_vs_mean_std_classification_impact.png"
                plt.savefig(p_c_perf)
                plt.close()

        # Regression comparison (CCC & RMSE)
        r_models = df_paired[df_paired["task"] == "regression"]
        if not r_models.empty:
            pivot_ccc = r_models.pivot_table(index=["representation", "architecture"], columns="pooling", values="ccc")
            pivot_rmse = r_models.pivot_table(index=["representation", "architecture"], columns="pooling", values="rmse")

            if not pivot_ccc.empty and "mean" in pivot_ccc and "mean_std" in pivot_ccc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
                ax1.scatter(pivot_ccc["mean"], pivot_ccc["mean_std"], color="#7570b3", s=65, edgecolors="black", linewidth=0.7)
                ax1.plot([-0.2, 0.4], [-0.2, 0.4], "r--", linewidth=1.2, label="Parity (y = x)")
                ax1.set_title("Dev CCC: Mean vs. Mean_Std", fontsize=11, fontweight="bold")
                ax1.set_xlabel("Mean Pooling CCC", fontsize=10)
                ax1.set_ylabel("Mean + Std Pooling CCC", fontsize=10)
                ax1.legend(fontsize=9)

                # Filter out extreme outliers for readable plot
                valid_rmse = pivot_rmse[(pivot_rmse["mean"] < 15) & (pivot_rmse["mean_std"] < 15)]
                ax2.scatter(valid_rmse["mean"], valid_rmse["mean_std"], color="#1b9e77", s=65, edgecolors="black", linewidth=0.7)
                ax2.plot([4, 12], [4, 12], "r--", linewidth=1.2, label="Parity (y = x)")
                ax2.set_title("Dev RMSE: Mean vs. Mean_Std (Lower is Better)", fontsize=11, fontweight="bold")
                ax2.set_xlabel("Mean Pooling RMSE", fontsize=10)
                ax2.set_ylabel("Mean + Std Pooling RMSE", fontsize=10)
                ax2.legend(fontsize=9)

                plt.tight_layout()
                p_r_perf = out_dir / "mean_vs_mean_std_regression_impact.png"
                plt.savefig(p_r_perf)
                plt.close()

    print("Mean vs. Mean_Std dedicated analysis complete.")


def main():
    print("=" * 80)
    print("PHASE 15: DATASET EDA & DEDICATED MEAN VS. MEAN+STD AUDIT")
    print("=" * 80)
    analyze_targets_and_splits()
    analyze_feature_dimensions()
    dedicated_mean_vs_mean_std_analysis()
    print("EDA execution successfully completed.")


if __name__ == "__main__":
    main()
