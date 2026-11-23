"""
E-DAIC Regression Benchmark Orchestrator (Prompt 5).

Evaluates 10 baseline regressors across all 24 pooled acoustic feature representations
targeting continuous PHQ_Score (0–24).
- 5-Fold Cross-Validation on official training participants (n=163).
- Development split evaluation (fitted on full Train n=163, evaluated on Dev n=56).
- Official test split is STRICTLY UNTOUCHED.
- No trained models or fitted pipelines are saved.
- Resumable execution without duplicate rows.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# Add code directory to path
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from config import (
    RANDOM_SEED, TARGET_REGRESSION,
    REGRESSION_OUTPUT_DIR, REG_PREDICTIONS_DIR, REG_CONFIGS_DIR,
    REG_COMPARISONS_DIR, REG_AUDIT_DIR, DATASET_DIR
)
from split_loader import load_split_dataset
from models.regression_models import REGRESSION_REGISTRY, get_regressor
from evaluation.regression_evaluator import run_5fold_cv_regression, run_development_regression
from visualization.regression_plots import (
    plot_predicted_vs_actual, plot_residuals, generate_regression_comparison_plots
)

# Ensure directories exist
for d in [REGRESSION_OUTPUT_DIR, REG_PREDICTIONS_DIR, REG_CONFIGS_DIR, REG_COMPARISONS_DIR, REG_AUDIT_DIR]:
    os.makedirs(d, exist_ok=True)

RESULTS_CSV = os.path.join(REGRESSION_OUTPUT_DIR, "results.csv")
FOLD_RESULTS_CSV = os.path.join(REGRESSION_OUTPUT_DIR, "fold_results.csv")

def run_regression_benchmark():
    print("=" * 70)
    print("E-DAIC REGRESSION BENCHMARK: CONTINUOUS PHQ_SCORE (0–24)")
    print("=" * 70)

    # Load dataset manifest
    manifest_csv = os.path.join(DATASET_DIR, "manifest.csv")
    df_manifest = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df_manifest)} pooled datasets from manifest.")

    # Load existing results for resumability
    completed_exp_ids = set()
    master_rows = []
    fold_rows = []

    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        completed_exp_ids = set(df_existing["experiment_id"].tolist())
        master_rows = df_existing.to_dict("records")
        print(f"Resuming: {len(completed_exp_ids)} experiments already completed.")

    if os.path.exists(FOLD_RESULTS_CSV):
        df_fold_existing = pd.read_csv(FOLD_RESULTS_CSV)
        fold_rows = df_fold_existing.to_dict("records")

    total_configs = len(df_manifest) * len(REGRESSION_REGISTRY)
    current_count = 0
    t_start = time.time()

    for _, row in df_manifest.iterrows():
        rep = row["representation"]
        pooling = row["pooling_strategy"]

        # 1. Load Train and Dev datasets using official split loader
        train_data = load_split_dataset(rep, pooling, split="train")
        dev_data = load_split_dataset(rep, pooling, split="dev")

        X_train = train_data.X
        y_train = train_data.y_score.astype(np.float64)
        train_pids = train_data.participant_ids

        X_dev = dev_data.X
        y_dev = dev_data.y_score.astype(np.float64)
        dev_pids = dev_data.participant_ids

        for model_name, model_spec in REGRESSION_REGISTRY.items():
            current_count += 1
            cv_exp_id = f"{rep}__{pooling}__{model_name}__cv5"
            dev_exp_id = f"{rep}__{pooling}__{model_name}__dev"

            # ------------------------------------------------------------------
            # A. 5-Fold Cross-Validation on Training Split
            # ------------------------------------------------------------------
            if cv_exp_id not in completed_exp_ids:
                cv_res = run_5fold_cv_regression(
                    X_train=X_train,
                    y_train=y_train,
                    participant_ids=train_pids,
                    representation=rep,
                    pooling_strategy=pooling,
                    model_name=model_name,
                    feature_names=train_data.feature_names,
                    n_splits=5,
                    random_seed=RANDOM_SEED,
                )

                # Save predictions
                pred_path = os.path.join(REG_PREDICTIONS_DIR, f"{cv_exp_id}__predictions.csv")
                cv_res.oof_predictions_df.to_csv(pred_path, index=False)

                # Generate diagnostic plots
                scatter_path = os.path.join(REG_PREDICTIONS_DIR, f"{cv_exp_id}__scatter.png")
                plot_predicted_vs_actual(
                    y_true=y_train,
                    y_pred=cv_res.oof_predictions_df["y_pred"].to_numpy(),
                    experiment_id=cv_exp_id,
                    ccc=cv_res.aggregate_oof_metrics.ccc,
                    rmse=cv_res.aggregate_oof_metrics.rmse,
                    mae=cv_res.aggregate_oof_metrics.mae,
                    output_path=scatter_path,
                    protocol="cv5",
                )
                resid_path = os.path.join(REG_PREDICTIONS_DIR, f"{cv_exp_id}__residuals.png")
                plot_residuals(
                    y_true=y_train,
                    y_pred=cv_res.oof_predictions_df["y_pred"].to_numpy(),
                    experiment_id=cv_exp_id,
                    output_path=resid_path,
                    protocol="cv5",
                )

                # Save config snapshot
                cfg_path = os.path.join(REG_CONFIGS_DIR, f"{cv_exp_id}__config.json")
                with open(cfg_path, "w") as f:
                    json.dump({
                        "experiment_id": cv_exp_id,
                        "representation": rep,
                        "pooling_strategy": pooling,
                        "model_name": model_name,
                        "target": TARGET_REGRESSION,
                        "protocol": "cv5",
                        "feature_count": cv_res.feature_count,
                        "train_samples": len(y_train),
                        "model_params": model_spec.default_params,
                        "requires_scaling": model_spec.requires_scaling,
                        "random_seed": RANDOM_SEED,
                    }, f, indent=2)

                # Append fold records
                for fr in cv_res.fold_results:
                    fold_rows.append({
                        "experiment_id": cv_exp_id,
                        "representation": rep,
                        "pooling_strategy": pooling,
                        "model_name": model_name,
                        "fold": fr.fold,
                        "train_samples": fr.train_samples,
                        "val_samples": fr.val_samples,
                        "ccc": fr.metrics.ccc,
                        "rmse": fr.metrics.rmse,
                        "mae": fr.metrics.mae,
                        "r2": fr.metrics.r2,
                        "runtime_sec": fr.runtime_sec,
                        "status": fr.status,
                        "error": fr.error,
                    })

                # Append master row
                master_rows.append({
                    "experiment_id": cv_exp_id,
                    "representation": rep,
                    "pooling_strategy": pooling,
                    "model_name": model_name,
                    "model_display_name": model_spec.display_name,
                    "target": TARGET_REGRESSION,
                    "evaluation_protocol": "cv5",
                    "feature_count": cv_res.feature_count,
                    "participant_count": len(y_train),
                    "ccc": cv_res.aggregate_oof_metrics.ccc,
                    "rmse": cv_res.aggregate_oof_metrics.rmse,
                    "mae": cv_res.aggregate_oof_metrics.mae,
                    "r2": cv_res.aggregate_oof_metrics.r2,
                    "ccc_mean": cv_res.mean_fold_metrics["ccc_mean"],
                    "ccc_std": cv_res.std_fold_metrics["ccc_std"],
                    "rmse_mean": cv_res.mean_fold_metrics["rmse_mean"],
                    "rmse_std": cv_res.std_fold_metrics["rmse_std"],
                    "mae_mean": cv_res.mean_fold_metrics["mae_mean"],
                    "mae_std": cv_res.std_fold_metrics["mae_std"],
                    "r2_mean": cv_res.mean_fold_metrics["r2_mean"],
                    "r2_std": cv_res.std_fold_metrics["r2_std"],
                    "runtime_sec": cv_res.total_runtime_sec,
                    "status": cv_res.status,
                    "warnings": "; ".join(cv_res.aggregate_oof_metrics.warnings) or "none",
                })
                completed_exp_ids.add(cv_exp_id)

            # ------------------------------------------------------------------
            # B. Development Split Evaluation
            # ------------------------------------------------------------------
            if dev_exp_id not in completed_exp_ids:
                dev_res = run_development_regression(
                    X_train=X_train,
                    y_train=y_train,
                    X_dev=X_dev,
                    y_dev=y_dev,
                    dev_participant_ids=dev_pids,
                    representation=rep,
                    pooling_strategy=pooling,
                    model_name=model_name,
                    feature_names=train_data.feature_names,
                )

                # Save predictions
                pred_path = os.path.join(REG_PREDICTIONS_DIR, f"{dev_exp_id}__predictions.csv")
                dev_res.dev_predictions_df.to_csv(pred_path, index=False)

                # Generate diagnostic plots
                scatter_path = os.path.join(REG_PREDICTIONS_DIR, f"{dev_exp_id}__scatter.png")
                plot_predicted_vs_actual(
                    y_true=y_dev,
                    y_pred=dev_res.dev_predictions_df["y_pred"].to_numpy(),
                    experiment_id=dev_exp_id,
                    ccc=dev_res.metrics.ccc,
                    rmse=dev_res.metrics.rmse,
                    mae=dev_res.metrics.mae,
                    output_path=scatter_path,
                    protocol="dev",
                )
                resid_path = os.path.join(REG_PREDICTIONS_DIR, f"{dev_exp_id}__residuals.png")
                plot_residuals(
                    y_true=y_dev,
                    y_pred=dev_res.dev_predictions_df["y_pred"].to_numpy(),
                    experiment_id=dev_exp_id,
                    output_path=resid_path,
                    protocol="dev",
                )

                # Save config snapshot
                cfg_path = os.path.join(REG_CONFIGS_DIR, f"{dev_exp_id}__config.json")
                with open(cfg_path, "w") as f:
                    json.dump({
                        "experiment_id": dev_exp_id,
                        "representation": rep,
                        "pooling_strategy": pooling,
                        "model_name": model_name,
                        "target": TARGET_REGRESSION,
                        "protocol": "dev",
                        "feature_count": dev_res.feature_count,
                        "train_samples": dev_res.train_samples,
                        "dev_samples": dev_res.dev_samples,
                        "model_params": model_spec.default_params,
                        "requires_scaling": model_spec.requires_scaling,
                        "random_seed": RANDOM_SEED,
                    }, f, indent=2)

                # Append master row
                master_rows.append({
                    "experiment_id": dev_exp_id,
                    "representation": rep,
                    "pooling_strategy": pooling,
                    "model_name": model_name,
                    "model_display_name": model_spec.display_name,
                    "target": TARGET_REGRESSION,
                    "evaluation_protocol": "dev",
                    "feature_count": dev_res.feature_count,
                    "participant_count": len(y_dev),
                    "ccc": dev_res.metrics.ccc,
                    "rmse": dev_res.metrics.rmse,
                    "mae": dev_res.metrics.mae,
                    "r2": dev_res.metrics.r2,
                    "ccc_mean": np.nan,
                    "ccc_std": np.nan,
                    "rmse_mean": np.nan,
                    "rmse_std": np.nan,
                    "mae_mean": np.nan,
                    "mae_std": np.nan,
                    "r2_mean": np.nan,
                    "r2_std": np.nan,
                    "runtime_sec": dev_res.runtime_sec,
                    "status": dev_res.status,
                    "warnings": "; ".join(dev_res.metrics.warnings) or "none",
                })
                completed_exp_ids.add(dev_exp_id)

            # Checkpoint CSVs periodically
            pd.DataFrame(master_rows).to_csv(RESULTS_CSV, index=False)
            pd.DataFrame(fold_rows).to_csv(FOLD_RESULTS_CSV, index=False)

            elapsed = time.time() - t_start
            print(f"[{current_count}/{total_configs}] {rep} | {pooling} | {model_name} processed ({elapsed/60:.2f}m elapsed)")

    print(f"\nAll 480 experiments completed! Results saved to {RESULTS_CSV}")

    # Generate aggregate comparison plots
    df_all_results = pd.read_csv(RESULTS_CSV)
    generate_regression_comparison_plots(df_all_results)


if __name__ == "__main__":
    run_regression_benchmark()
