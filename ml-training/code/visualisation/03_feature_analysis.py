"""
Phase 15 - Script 03: Feature Correlation and Feature-Target Association Analysis.

Computes:
- Feature-feature correlation matrices (Pearson & Spearman) on Train-only cohort (n=163) for leakage safety.
- Identification of collinear and redundant feature pairs (|r| > 0.95).
- Continuous target associations (PHQ_Score): Pearson r and Spearman rho.
- Binary target associations (PHQ_Binary): Point-biserial r and t-statistics.

Outputs to:
- outputs/model_visualizations_v1/data_exploration/feature_correlations/
- outputs/model_visualizations_v1/data_exploration/feature_target_associations/
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

CORR_DIR = OUTPUT_DIR / "data_exploration" / "feature_correlations"
ASSOC_DIR = OUTPUT_DIR / "data_exploration" / "feature_target_associations"
CORR_DIR.mkdir(parents=True, exist_ok=True)
ASSOC_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def load_train_features_and_targets():
    """Loads Train participants (n=163) with aligned targets."""
    train_split_csv = EDAIC_WOZ / "labels" / "train_split.csv"
    df_labels = pd.read_csv(train_split_csv)
    train_pids = df_labels["Participant_ID"].dropna().astype(int).tolist()
    labels_dict = df_labels.set_index("Participant_ID")[["PHQ_Score", "PHQ_Binary"]].to_dict(orient="index")
    return train_pids, labels_dict


def compute_feature_correlations(train_pids, labels_dict):
    print("--- Computing Feature-Feature Correlations (Train Cohort n=163) ---")
    datasets_to_analyze = [
        ("raw_egemaps__mean", DATASET_DIR / "raw_egemaps" / "mean.csv", "Acoustic (eGeMAPS Mean)"),
        ("raw_mfcc__mean", DATASET_DIR / "raw_mfcc" / "mean.csv", "Acoustic (MFCC Mean)"),
        ("boaw_egemaps__mean", DATASET_DIR / "boaw_egemaps" / "mean.csv", "BoAW eGeMAPS Mean"),
        ("densenet201__mean", DATASET_DIR / "deep_features_v1" / "pooled" / "densenet201__mean.csv", "DenseNet201 Mean"),
        ("vgg16__mean", DATASET_DIR / "deep_features_v1" / "pooled" / "vgg16__mean.csv", "VGG16 Mean"),
    ]

    all_redundant_pairs = []

    for name, path, label in datasets_to_analyze:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        pid_col = "Participant_ID" if "Participant_ID" in df.columns else "participant_id"
        df_train = df[df[pid_col].isin(train_pids)].copy()

        feat_cols = [c for c in df_train.columns if c != pid_col and "phq" not in c.lower()]
        # Remove zero variance columns
        stds = df_train[feat_cols].std(numeric_only=True)
        valid_cols = stds[stds > 1e-7].index.tolist()

        # If high-dimensional, select top 30 variance features for readable heatmap
        if len(valid_cols) > 35:
            top_cols = df_train[valid_cols].var().nlargest(30).index.tolist()
            plot_cols = top_cols
            title_suffix = "(Top 30 Highest Variance Features)"
        else:
            plot_cols = valid_cols
            title_suffix = f"(All {len(valid_cols)} Features)"

        corr_matrix = df_train[plot_cols].corr(method="pearson")

        fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
            ax=ax,
        )
        ax.set_title(f"Feature-Feature Correlation Matrix: {label}\n{title_suffix} (Train Cohort, n={len(df_train)})",
                     fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        p_out = CORR_DIR / f"correlation_matrix_{name}.png"
        plt.savefig(p_out)
        plt.close()

        # Full correlation matrix scan for redundant pairs (|r| > 0.95)
        full_corr = df_train[valid_cols].corr(method="pearson").to_numpy()
        triu_i, triu_j = np.triu_indices(len(valid_cols), k=1)
        r_vals = full_corr[triu_i, triu_j]
        high_corr_mask = np.abs(r_vals) > 0.95
        high_corr_indices = np.where(high_corr_mask)[0]

        for idx in high_corr_indices:
            i_idx, j_idx = triu_i[idx], triu_j[idx]
            r_val = float(r_vals[idx])
            all_redundant_pairs.append({
                "dataset": name,
                "feature_1": valid_cols[i_idx],
                "feature_2": valid_cols[j_idx],
                "pearson_r": round(r_val, 4),
                "abs_r": round(abs(r_val), 4),
            })

    df_red = pd.DataFrame(all_redundant_pairs)
    df_red.to_csv(CORR_DIR / "redundant_feature_pairs.csv", index=False)
    print(f"Feature correlations saved. Redundant pairs (|r| > 0.95): {len(df_red)}")


def compute_feature_target_associations(train_pids, labels_dict):
    print("--- Computing Feature-Target Associations (Train Cohort n=163) ---")
    datasets_to_analyze = [
        ("raw_egemaps__mean_std", DATASET_DIR / "raw_egemaps" / "mean_std.csv", "eGeMAPS Mean+Std"),
        ("raw_mfcc__mean_std", DATASET_DIR / "raw_mfcc" / "mean_std.csv", "MFCC Mean+Std"),
        ("densenet201__mean", DATASET_DIR / "deep_features_v1" / "pooled" / "densenet201__mean.csv", "DenseNet201 Mean"),
        ("vgg16__mean", DATASET_DIR / "deep_features_v1" / "pooled" / "vgg16__mean.csv", "VGG16 Mean"),
    ]

    all_assoc_records = []

    y_score_series = pd.Series({pid: labels_dict[pid]["PHQ_Score"] for pid in train_pids})
    y_bin_series = pd.Series({pid: labels_dict[pid]["PHQ_Binary"] for pid in train_pids})

    for name, path, label in datasets_to_analyze:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        pid_col = "Participant_ID" if "Participant_ID" in df.columns else "participant_id"
        df_train = df[df[pid_col].isin(train_pids)].copy().set_index(pid_col)

        feat_cols = [c for c in df_train.columns if "phq" not in c.lower()]

        for col in feat_cols:
            x = df_train[col].to_numpy()
            if np.std(x) < 1e-7:
                continue

            # Continuous target association
            r_pearson, p_pearson = stats.pearsonr(x, y_score_series.loc[df_train.index])
            rho_spearman, p_spearman = stats.spearmanr(x, y_score_series.loc[df_train.index])

            # Binary target association (Point-biserial)
            r_pb, p_pb = stats.pointbiserialr(y_bin_series.loc[df_train.index], x)

            all_assoc_records.append({
                "dataset": name,
                "dataset_label": label,
                "feature": col,
                "pearson_r_score": round(float(r_pearson), 4),
                "pearson_p_score": float(p_pearson),
                "spearman_rho_score": round(float(rho_spearman), 4),
                "spearman_p_score": float(p_spearman),
                "point_biserial_r_binary": round(float(r_pb), 4),
                "point_biserial_p_binary": float(p_pb),
                "abs_pearson_r": round(float(abs(r_pearson)), 4),
            })

    df_assoc = pd.DataFrame(all_assoc_records)
    df_assoc.to_csv(ASSOC_DIR / "feature_target_associations_master.csv", index=False)

    # Plot top 15 features associated with continuous PHQ_Score
    top_score = df_assoc.sort_values("abs_pearson_r", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    colors = ["#2b5c8f" if r > 0 else "#d95f02" for r in top_score["pearson_r_score"]]
    labels = [f"{row['dataset_label']}: {row['feature'][:28]}" for _, row in top_score.iterrows()]
    y_pos = np.arange(len(top_score))

    bars = ax.barh(y_pos, top_score["pearson_r_score"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Top 15 Features Associated with Continuous PHQ-8 Score\n(Train Cohort, n=163, Pearson r)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Pearson Correlation (r with PHQ_Score)", fontsize=10)

    # Annotate bars
    for bar in bars:
        w = bar.get_width()
        offset = 0.01 if w >= 0 else -0.01
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2.0, f"{w:+.3f}", ha=ha, va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlim(-0.4, 0.4)
    plt.tight_layout()
    p_assoc_score = ASSOC_DIR / "top_features_associated_with_phq_score.png"
    plt.savefig(p_assoc_score)
    plt.close()

    # Plot top 15 features associated with binary PHQ_Binary
    top_bin = df_assoc.sort_values("point_biserial_r_binary", key=abs, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    colors = ["#2b5c8f" if r > 0 else "#d95f02" for r in top_bin["point_biserial_r_binary"]]
    labels = [f"{row['dataset_label']}: {row['feature'][:28]}" for _, row in top_bin.iterrows()]
    y_pos = np.arange(len(top_bin))

    bars = ax.barh(y_pos, top_bin["point_biserial_r_binary"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Top 15 Features Associated with Binary Depression (PHQ >= 10)\n(Train Cohort, n=163, Point-Biserial r_pb)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Point-Biserial Correlation (r_pb with PHQ_Binary)", fontsize=10)

    for bar in bars:
        w = bar.get_width()
        offset = 0.01 if w >= 0 else -0.01
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2.0, f"{w:+.3f}", ha=ha, va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlim(-0.4, 0.4)
    plt.tight_layout()
    p_assoc_bin = ASSOC_DIR / "top_features_associated_with_phq_binary.png"
    plt.savefig(p_assoc_bin)
    plt.close()
    print("Feature-target associations saved.")


def main():
    print("=" * 80)
    print("PHASE 15: FEATURE CORRELATION & TARGET ASSOCIATION PIPELINE")
    print("=" * 80)
    train_pids, labels_dict = load_train_features_and_targets()
    compute_feature_correlations(train_pids, labels_dict)
    compute_feature_target_associations(train_pids, labels_dict)
    print("Feature analysis completed successfully.")


if __name__ == "__main__":
    main()
