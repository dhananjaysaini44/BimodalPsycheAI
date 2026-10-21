"""
Phase 15 - Script 07: Prediction and Error Correlation Analysis.

Computes and visualizes:
1. Classification Prediction Correlation Matrix (probabilities / decision scores on Dev n=56)
2. Regression Prediction Correlation Matrix (predicted PHQ-8 scores on Dev n=56)
3. Regression Residual Correlation Matrix (error residuals on Dev n=56)
4. Model Diversity and Error Orthogonality Analysis for Ensembling

All figures saved exclusively in PNG format to:
outputs/model_visualizations_v1/prediction_analysis/
"""

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
PRED_DIR = OUTPUT_DIR / "prediction_analysis"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Global styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_dev_participants():
    """Loads official Dev participant IDs (n=56)."""
    dev_split_csv = ML_TRAINING / "edaic-woz" / "labels" / "dev_split.csv"
    df = pd.read_csv(dev_split_csv)
    return sorted(df["Participant_ID"].dropna().astype(int).tolist())


def select_representative_models(df_reg, task_name, max_models=16):
    """Selects a diverse cohort of models spanning all benchmark families."""
    df_dev = df_reg[(df_reg["task"] == task_name) & (df_reg["evaluation_split"] == "dev")].copy()
    
    selected = []
    # Sort by metric
    sort_col = "f1" if task_name == "classification" else "ccc"
    
    # 1. Broad classical acoustic: best of each representation
    for rep in ["raw_egemaps", "raw_mfcc", "boaw_egemaps", "boaw_mfcc"]:
        sub = df_dev[(df_dev["benchmark_suite"] == "Broad Classical Acoustic") & (df_dev["representation"] == rep)]
        if not sub.empty:
            selected.append(sub.sort_values(sort_col, ascending=False).iloc[0])
            
    # 2. Controlled Classical
    sub_ctrl = df_dev[df_dev["benchmark_suite"] == "Controlled Classical Benchmark"]
    if not sub_ctrl.empty:
        selected.append(sub_ctrl.sort_values(sort_col, ascending=False).iloc[0])

    # 3. Phase 12 (CNN Classical)
    for rep in ["vgg16", "densenet201"]:
        sub = df_dev[(df_dev["benchmark_suite"] == "Phase 12 (CNN Classical Baselines)") & (df_dev["representation"] == rep)]
        if not sub.empty:
            selected.append(sub.sort_values(sort_col, ascending=False).iloc[0])

    # 4. Phase 13A (CNN MLP Neural Heads)
    for rep in ["vgg16", "densenet201"]:
        sub = df_dev[(df_dev["benchmark_suite"] == "Phase 13A (Neural Prediction Heads)") & (df_dev["representation"] == rep)]
        if not sub.empty:
            selected.append(sub.sort_values(sort_col, ascending=False).iloc[0])

    # 5. Phase 13B (CNN Temporal GRU+Attention)
    for rep in ["vgg16", "densenet201"]:
        sub = df_dev[(df_dev["benchmark_suite"] == "Phase 13B (Temporal Sequence Models)") & (df_dev["representation"] == rep)]
        if not sub.empty:
            selected.append(sub.sort_values(sort_col, ascending=False).iloc[0])

    df_sel = pd.DataFrame(selected).drop_duplicates(subset=["model_id"]).head(max_models)
    return df_sel


def analyze_classification_correlations(df_reg, dev_pids):
    print("--- Analyzing Classification Prediction Correlations on Dev (n=56) ---")
    df_sel = select_representative_models(df_reg, "classification", max_models=12)
    if df_sel.empty:
        return

    pred_matrix = {}
    col_labels = []

    for _, row in df_sel.iterrows():
        p_path = PROJECT_ROOT / row["prediction_artifact"]
        if not p_path.exists():
            p_path = ML_TRAINING / row["prediction_artifact"]
        if not p_path.exists():
            continue
        df_p = pd.read_csv(p_path)
        pid_col = "Participant_ID" if "Participant_ID" in df_p.columns else "participant_id"
        df_p = df_p.set_index(pid_col).reindex(dev_pids)

        if "y_score" in df_p.columns:
            scores = df_p["y_score"].to_numpy(dtype=float)
        elif "decision_function" in df_p.columns:
            scores = df_p["decision_function"].to_numpy(dtype=float)
        elif "probability" in df_p.columns:
            scores = df_p["probability"].to_numpy(dtype=float)
        else:
            scores = df_p["y_pred"].to_numpy(dtype=float)

        # Normalize with sigmoid if outside [0, 1]
        if np.nanmin(scores) < 0 or np.nanmax(scores) > 1:
            scores = 1.0 / (1.0 + np.exp(-scores))

        short_label = f"{row['architecture'][:8]}_{row['representation'][:8]}"
        pred_matrix[short_label] = scores
        col_labels.append(short_label)

    df_preds = pd.DataFrame(pred_matrix, index=dev_pids)
    corr = df_preds.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=0.0,
        center=0.5,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title("Classification Prediction Score Correlations (Dev Split, n=56)\nCross-Model Diversity & Alignment",
                 fontsize=11.5, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(PRED_DIR / "classification_prediction_correlation_matrix_dev.png")
    plt.close(fig)
    print("Classification prediction correlation matrix saved.")


def analyze_regression_correlations(df_reg, dev_pids):
    print("--- Analyzing Regression Prediction & Residual Correlations on Dev (n=56) ---")
    df_sel = select_representative_models(df_reg, "regression", max_models=12)
    if df_sel.empty:
        return

    pred_matrix = {}
    resid_matrix = {}

    for _, row in df_sel.iterrows():
        p_path = PROJECT_ROOT / row["prediction_artifact"]
        if not p_path.exists():
            p_path = ML_TRAINING / row["prediction_artifact"]
        if not p_path.exists():
            continue
        df_p = pd.read_csv(p_path)
        pid_col = "Participant_ID" if "Participant_ID" in df_p.columns else "participant_id"
        df_p = df_p.set_index(pid_col).reindex(dev_pids)

        y_true = df_p["y_true"].to_numpy(dtype=float)
        y_pred = df_p["y_pred"].to_numpy(dtype=float)
        residual = y_true - y_pred

        short_label = f"{row['architecture'][:8]}_{row['representation'][:8]}"
        pred_matrix[short_label] = y_pred
        resid_matrix[short_label] = residual

    df_preds = pd.DataFrame(pred_matrix, index=dev_pids)
    df_resids = pd.DataFrame(resid_matrix, index=dev_pids)

    # 1. Prediction Correlation
    corr_preds = df_preds.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    mask = np.triu(np.ones_like(corr_preds, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_preds,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.2,
        center=0.4,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title("Regression Prediction Correlations (Dev Split, n=56)\nPredicted Continuous PHQ-8 Scores",
                 fontsize=11.5, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(PRED_DIR / "regression_prediction_correlation_matrix_dev.png")
    plt.close(fig)

    # 2. Residual Correlation (Error Complementarity)
    corr_resids = df_resids.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    sns.heatmap(
        corr_resids,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.2,
        center=0.4,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Residual Correlation (r)"},
        ax=ax,
    )
    ax.set_title("Regression Residual (Error) Correlations (Dev Split, n=56)\nLower Correlation Indicates Complementary Error Distributions (Ensembling Potential)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(PRED_DIR / "regression_residual_correlation_matrix_dev.png")
    plt.close(fig)

    print("Regression prediction and residual correlation matrices saved.")


def main():
    print("=" * 80)
    print("PHASE 15: PREDICTION & ERROR CORRELATION PIPELINE")
    print("=" * 80)

    reg_path = OUTPUT_DIR / "model_registry.csv"
    if not reg_path.exists():
        return

    df_reg = pd.read_csv(reg_path)
    dev_pids = load_dev_participants()
    print(f"Loaded {len(dev_pids)} validation participants.")

    analyze_classification_correlations(df_reg, dev_pids)
    analyze_regression_correlations(df_reg, dev_pids)
    print("Prediction and error correlation analysis completed.")


if __name__ == "__main__":
    main()
