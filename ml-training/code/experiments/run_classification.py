"""
Benchmark Orchestration Module for E-DAIC Classification.

Runs the complete, reproducible classical classification benchmark across
all valid pooled feature datasets and models:
1. Dynamically discovers valid datasets from dataset/manifest.csv.
2. Evaluates enabled baseline classifiers using 5-fold Stratified CV on the training split.
3. Evaluates models on the development split (fit on full training split).
4. Strictly protects the official test split (never loaded or evaluated).
5. Saves predictions, metrics, curve coordinates, ROC/PR curves, and comparison figures.
6. Implements non-destructive resumability.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import json
import logging
import time
import pandas as pd
import numpy as np

code_root = str(Path(__file__).resolve().parent.parent)
if code_root not in sys.path:
    sys.path.insert(0, code_root)

from config import (
    DATASET_DIR,
    CLASSIFICATION_OUTPUT_DIR,
    PREDICTIONS_DIR,
    ROC_CURVES_DIR,
    PR_CURVES_DIR,
    CONFUSION_MATRICES_DIR,
    COMPARISONS_DIR,
    CURVE_DATA_DIR,
    CONFIGS_DIR,
    LOGS_DIR,
    RANDOM_SEED,
    TARGET_CLASSIFICATION,
    POSITIVE_CLASS_VALUE,
)
from split_loader import load_all_splits, DatasetSplits
from models.classification_models import CLASSIFIER_REGISTRY
from evaluation.cross_validation import run_cross_validation, CrossValidationResult
from evaluation.classification_evaluator import evaluate_on_development, DevelopmentResult
from visualization.classification_plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_model_pooling_heatmap,
)
from experiments.experiment_registry import (
    build_experiment_id,
    get_completed_experiment_ids,
    save_experiment_config,
)

# Setup logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "classification_benchmark.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def get_valid_datasets_from_manifest(manifest_path: Optional[Path] = None) -> List[Tuple[str, str]]:
    """
    Reads dataset/manifest.csv and returns list of (representation, pooling_strategy)
    tuples for all successfully generated datasets.
    """
    if manifest_path is None:
        manifest_path = DATASET_DIR / "manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    valid_combos = []

    for _, row in df_manifest.iterrows():
        rep = row["representation"]
        strat = row["pooling_strategy"]
        status = row.get("status", "SUCCESS")
        
        # Verify file actually exists on disk
        csv_file = DATASET_DIR / rep / f"{strat}.csv"
        if csv_file.exists() and "SUCCESS" in status.upper():
            valid_combos.append((rep, strat))
        else:
            logger.warning(f"Dataset {rep}/{strat} marked as {status} but file missing or invalid. Skipping.")

    logger.info(f"Discovered {len(valid_combos)} valid pooled datasets in manifest.")
    return valid_combos


def run_benchmark(
    representations_filter: Optional[List[str]] = None,
    pooling_filter: Optional[List[str]] = None,
    models_filter: Optional[List[str]] = None,
    force_rebuild: bool = False,
    generate_plots: bool = True,
) -> pd.DataFrame:
    """
    Executes the classification benchmark across all specified datasets and models.
    """
    # Create output directories
    for d in [
        CLASSIFICATION_OUTPUT_DIR,
        PREDICTIONS_DIR,
        ROC_CURVES_DIR,
        PR_CURVES_DIR,
        CONFUSION_MATRICES_DIR,
        COMPARISONS_DIR,
        CURVE_DATA_DIR,
        CONFIGS_DIR,
        LOGS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Discover valid datasets from manifest
    all_combos = get_valid_datasets_from_manifest()
    if representations_filter:
        all_combos = [c for c in all_combos if c[0] in representations_filter]
    if pooling_filter:
        all_combos = [c for c in all_combos if c[1] in pooling_filter]

    # 2. Filter models
    model_names = list(CLASSIFIER_REGISTRY.keys())
    if models_filter:
        model_names = [m for m in model_names if m in models_filter]

    # 3. Check existing completed experiments for resumability
    results_csv = CLASSIFICATION_OUTPUT_DIR / "results.csv"
    fold_results_csv = CLASSIFICATION_OUTPUT_DIR / "fold_results.csv"
    completed_ids = set() if force_rebuild else get_completed_experiment_ids(results_csv)

    logger.info(
        f"Starting Classification Benchmark: {len(all_combos)} datasets x {len(model_names)} models "
        f"({len(all_combos) * len(model_names)} pairs). Completed already: {len(completed_ids)}."
    )

    results_rows: List[Dict[str, Any]] = []
    fold_results_rows: List[Dict[str, Any]] = []

    # If results.csv exists and not force_rebuild, preserve previous rows
    if results_csv.exists() and not force_rebuild:
        try:
            prev_df = pd.read_csv(results_csv)
            results_rows = prev_df.to_dict("records")
        except Exception as e:
            logger.warning(f"Could not load previous results: {e}")

    if fold_results_csv.exists() and not force_rebuild:
        try:
            prev_fold_df = pd.read_csv(fold_results_csv)
            fold_results_rows = prev_fold_df.to_dict("records")
        except Exception as e:
            logger.warning(f"Could not load previous fold results: {e}")

    total_pairs = len(all_combos) * len(model_names)
    pair_count = 0

    for rep, strat in all_combos:
        # Load dataset splits once for this representation & pooling combination
        splits: Optional[DatasetSplits] = None

        for model_name in model_names:
            pair_count += 1
            model_spec = CLASSIFIER_REGISTRY[model_name]
            
            exp_id_cv = build_experiment_id(rep, strat, model_name, "cv5")
            exp_id_dev = build_experiment_id(rep, strat, model_name, "dev")

            need_cv = (exp_id_cv not in completed_ids) or force_rebuild
            need_dev = (exp_id_dev not in completed_ids) or force_rebuild

            if not need_cv and not need_dev:
                logger.info(f"[{pair_count}/{total_pairs}] SKIPPING (already completed): {rep} - {strat} - {model_name}")
                continue

            logger.info(f"[{pair_count}/{total_pairs}] RUNNING: {rep} - {strat} - {model_name}")

            # Lazy load splits only when needed
            if splits is None:
                try:
                    splits = load_all_splits(rep, strat)
                except Exception as e:
                    logger.error(f"Failed to load splits for {rep}/{strat}: {e}")
                    # Record failure for both
                    for eid, proto in [(exp_id_cv, "cv5"), (exp_id_dev, "dev")]:
                        results_rows.append({
                            "experiment_id": eid,
                            "representation": rep,
                            "pooling_strategy": strat,
                            "model_name": model_name,
                            "target": TARGET_CLASSIFICATION,
                            "evaluation_protocol": proto,
                            "feature_count": 0,
                            "random_seed": RANDOM_SEED,
                            "status": "FAILED",
                            "error": str(e),
                        })
                    continue

            feature_dim = splits.train.num_features

            # Save configuration fingerprint
            save_experiment_config(
                experiment_id=exp_id_cv,
                representation=rep,
                pooling_strategy=strat,
                model_name=model_name,
                evaluation_protocol="cv5",
                model_params=model_spec.default_params,
                feature_count=feature_dim,
            )
            save_experiment_config(
                experiment_id=exp_id_dev,
                representation=rep,
                pooling_strategy=strat,
                model_name=model_name,
                evaluation_protocol="dev",
                model_params=model_spec.default_params,
                feature_count=feature_dim,
            )

            # ==========================================================
            # A. Cross-Validation on Training Split (Strictly 163 PIDs)
            # ==========================================================
            if need_cv:
                try:
                    cv_res = run_cross_validation(splits.train, model_name=model_name, n_splits=5, random_state=RANDOM_SEED)
                    
                    # 1. Save OOF predictions
                    pred_file = PREDICTIONS_DIR / f"{exp_id_cv}__predictions.csv"
                    cv_res.oof_predictions_df.to_csv(pred_file, index=False)

                    # 2. Save curve data
                    with open(CURVE_DATA_DIR / f"{exp_id_cv}__roc.json", "w", encoding="utf-8") as f:
                        json.dump(cv_res.oof_roc_data, f, indent=2)
                    with open(CURVE_DATA_DIR / f"{exp_id_cv}__pr.json", "w", encoding="utf-8") as f:
                        json.dump(cv_res.oof_pr_data, f, indent=2)

                    # 3. Generate diagnostic figures
                    if generate_plots:
                        plot_roc_curve(
                            cv_res.oof_roc_data,
                            title=f"OOF ROC: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=ROC_CURVES_DIR / f"{exp_id_cv}__roc.png",
                        )
                        pos_prevalence = float(np.mean(splits.train.y_binary == POSITIVE_CLASS_VALUE))
                        plot_pr_curve(
                            cv_res.oof_pr_data,
                            title=f"OOF PR: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=PR_CURVES_DIR / f"{exp_id_cv}__precision_recall.png",
                            pos_prevalence=pos_prevalence,
                        )
                        plot_confusion_matrix(
                            tn=cv_res.oof_metrics.tn,
                            fp=cv_res.oof_metrics.fp,
                            fn=cv_res.oof_metrics.fn,
                            tp=cv_res.oof_metrics.tp,
                            title=f"OOF Confusion Matrix: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=CONFUSION_MATRICES_DIR / f"{exp_id_cv}__confusion.png",
                            normalize=False,
                        )
                        plot_confusion_matrix(
                            tn=cv_res.oof_metrics.tn,
                            fp=cv_res.oof_metrics.fp,
                            fn=cv_res.oof_metrics.fn,
                            tp=cv_res.oof_metrics.tp,
                            title=f"OOF Normalized CM: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=CONFUSION_MATRICES_DIR / f"{exp_id_cv}__confusion_normalized.png",
                            normalize=True,
                        )

                    # 4. Save fold-level metrics
                    for f_idx, f_m in enumerate(cv_res.fold_metrics, start=1):
                        f_row = {
                            "experiment_id": exp_id_cv,
                            "fold": f_idx,
                            "representation": rep,
                            "pooling_strategy": strat,
                            "model_name": model_name,
                        }
                        f_row.update(f_m.to_dict())
                        fold_results_rows.append(f_row)

                    # 5. Record summary row
                    summary_row = {
                        "experiment_id": exp_id_cv,
                        "representation": rep,
                        "pooling_strategy": strat,
                        "model_name": model_name,
                        "model_display_name": model_spec.display_name,
                        "target": TARGET_CLASSIFICATION,
                        "evaluation_protocol": "cv5",
                        "feature_count": feature_dim,
                        "participant_count": splits.train.num_participants,
                        "accuracy": cv_res.oof_metrics.accuracy,
                        "balanced_accuracy": cv_res.oof_metrics.balanced_accuracy,
                        "f1": cv_res.oof_metrics.f1,
                        "precision": cv_res.oof_metrics.precision,
                        "recall": cv_res.oof_metrics.recall,
                        "roc_auc": cv_res.oof_metrics.roc_auc,
                        "average_precision": cv_res.oof_metrics.average_precision,
                        "accuracy_mean": cv_res.mean_metrics.get("accuracy_mean"),
                        "accuracy_std": cv_res.std_metrics.get("accuracy_std"),
                        "f1_mean": cv_res.mean_metrics.get("f1_mean"),
                        "f1_std": cv_res.std_metrics.get("f1_std"),
                        "balanced_accuracy_mean": cv_res.mean_metrics.get("balanced_accuracy_mean"),
                        "balanced_accuracy_std": cv_res.std_metrics.get("balanced_accuracy_std"),
                        "roc_auc_mean": cv_res.mean_metrics.get("roc_auc_mean"),
                        "roc_auc_std": cv_res.std_metrics.get("roc_auc_std"),
                        "tn": cv_res.oof_metrics.tn,
                        "fp": cv_res.oof_metrics.fp,
                        "fn": cv_res.oof_metrics.fn,
                        "tp": cv_res.oof_metrics.tp,
                        "runtime_sec": cv_res.total_runtime_sec,
                        "status": "SUCCESS",
                        "warnings": "; ".join(cv_res.oof_metrics.warnings) if cv_res.oof_metrics.warnings else "none",
                    }
                    results_rows.append(summary_row)
                    completed_ids.add(exp_id_cv)
                except Exception as e:
                    logger.error(f"Error in CV for {exp_id_cv}: {e}")
                    results_rows.append({
                        "experiment_id": exp_id_cv,
                        "representation": rep,
                        "pooling_strategy": strat,
                        "model_name": model_name,
                        "target": TARGET_CLASSIFICATION,
                        "evaluation_protocol": "cv5",
                        "feature_count": feature_dim,
                        "status": "FAILED",
                        "error": str(e),
                    })

            # ==========================================================
            # B. Development Split Evaluation (Strictly 56 PIDs)
            # ==========================================================
            if need_dev:
                try:
                    dev_res = evaluate_on_development(splits.train, splits.dev, model_name=model_name)

                    # 1. Save Dev predictions
                    pred_file = PREDICTIONS_DIR / f"{exp_id_dev}__predictions.csv"
                    dev_res.predictions_df.to_csv(pred_file, index=False)

                    # 2. Save curve data
                    with open(CURVE_DATA_DIR / f"{exp_id_dev}__roc.json", "w", encoding="utf-8") as f:
                        json.dump(dev_res.roc_data, f, indent=2)
                    with open(CURVE_DATA_DIR / f"{exp_id_dev}__pr.json", "w", encoding="utf-8") as f:
                        json.dump(dev_res.pr_data, f, indent=2)

                    # 3. Generate diagnostic figures
                    if generate_plots:
                        plot_roc_curve(
                            dev_res.roc_data,
                            title=f"Dev ROC: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=ROC_CURVES_DIR / f"{exp_id_dev}__roc.png",
                        )
                        pos_prevalence = float(np.mean(splits.dev.y_binary == POSITIVE_CLASS_VALUE))
                        plot_pr_curve(
                            dev_res.pr_data,
                            title=f"Dev PR: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=PR_CURVES_DIR / f"{exp_id_dev}__precision_recall.png",
                            pos_prevalence=pos_prevalence,
                        )
                        plot_confusion_matrix(
                            tn=dev_res.metrics.tn,
                            fp=dev_res.metrics.fp,
                            fn=dev_res.metrics.fn,
                            tp=dev_res.metrics.tp,
                            title=f"Dev Confusion Matrix: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=CONFUSION_MATRICES_DIR / f"{exp_id_dev}__confusion.png",
                            normalize=False,
                        )
                        plot_confusion_matrix(
                            tn=dev_res.metrics.tn,
                            fp=dev_res.metrics.fp,
                            fn=dev_res.metrics.fn,
                            tp=dev_res.metrics.tp,
                            title=f"Dev Normalized CM: {model_spec.display_name}\n({rep} - {strat})",
                            save_path=CONFUSION_MATRICES_DIR / f"{exp_id_dev}__confusion_normalized.png",
                            normalize=True,
                        )

                    # 4. Record Dev summary row
                    summary_row = {
                        "experiment_id": exp_id_dev,
                        "representation": rep,
                        "pooling_strategy": strat,
                        "model_name": model_name,
                        "model_display_name": model_spec.display_name,
                        "target": TARGET_CLASSIFICATION,
                        "evaluation_protocol": "dev",
                        "feature_count": feature_dim,
                        "participant_count": splits.dev.num_participants,
                        "accuracy": dev_res.metrics.accuracy,
                        "balanced_accuracy": dev_res.metrics.balanced_accuracy,
                        "f1": dev_res.metrics.f1,
                        "precision": dev_res.metrics.precision,
                        "recall": dev_res.metrics.recall,
                        "roc_auc": dev_res.metrics.roc_auc,
                        "average_precision": dev_res.metrics.average_precision,
                        "tn": dev_res.metrics.tn,
                        "fp": dev_res.metrics.fp,
                        "fn": dev_res.metrics.fn,
                        "tp": dev_res.metrics.tp,
                        "runtime_sec": dev_res.runtime_sec,
                        "status": "SUCCESS",
                        "warnings": "; ".join(dev_res.metrics.warnings) if dev_res.metrics.warnings else "none",
                    }
                    results_rows.append(summary_row)
                    completed_ids.add(exp_id_dev)
                except Exception as e:
                    logger.error(f"Error in Dev eval for {exp_id_dev}: {e}")
                    results_rows.append({
                        "experiment_id": exp_id_dev,
                        "representation": rep,
                        "pooling_strategy": strat,
                        "model_name": model_name,
                        "target": TARGET_CLASSIFICATION,
                        "evaluation_protocol": "dev",
                        "feature_count": feature_dim,
                        "status": "FAILED",
                        "error": str(e),
                    })

        # Save progress after each dataset representation/pooling combo
        df_current_results = pd.DataFrame(results_rows).drop_duplicates(subset=["experiment_id"], keep="last")
        df_current_results.to_csv(results_csv, index=False)

        if fold_results_rows:
            df_curr_fold = pd.DataFrame(fold_results_rows).drop_duplicates(subset=["experiment_id", "fold"], keep="last")
            df_curr_fold.to_csv(fold_results_csv, index=False)

    # 4. Generate comparison plots across all completed experiments
    logger.info("Generating global benchmark comparison plots...")
    final_results_df = pd.DataFrame(results_rows).drop_duplicates(subset=["experiment_id"], keep="last")
    final_results_df.to_csv(results_csv, index=False)

    cv_df = final_results_df[final_results_df["evaluation_protocol"] == "cv5"]
    dev_df = final_results_df[final_results_df["evaluation_protocol"] == "dev"]

    if not cv_df.empty:
        plot_model_comparison(cv_df, "f1", "Top Models by F1-Score (5-Fold CV OOF)", COMPARISONS_DIR / "top_models_f1_cv5.png")
        plot_model_comparison(cv_df, "balanced_accuracy", "Top Models by Balanced Accuracy (5-Fold CV OOF)", COMPARISONS_DIR / "top_models_balanced_acc_cv5.png")
        plot_model_comparison(cv_df, "roc_auc", "Top Models by ROC-AUC (5-Fold CV OOF)", COMPARISONS_DIR / "top_models_roc_auc_cv5.png")
        plot_model_pooling_heatmap(cv_df, "f1", "F1-Score Heatmap across Representations & Models (5-Fold CV OOF)", COMPARISONS_DIR / "heatmap_f1_cv5.png")
        plot_model_pooling_heatmap(cv_df, "roc_auc", "ROC-AUC Heatmap across Representations & Models (5-Fold CV OOF)", COMPARISONS_DIR / "heatmap_roc_auc_cv5.png")

    if not dev_df.empty:
        plot_model_comparison(dev_df, "f1", "Top Models by F1-Score (Development Split)", COMPARISONS_DIR / "top_models_f1_dev.png")
        plot_model_comparison(dev_df, "balanced_accuracy", "Top Models by Balanced Accuracy (Development Split)", COMPARISONS_DIR / "top_models_balanced_acc_dev.png")
        plot_model_comparison(dev_df, "roc_auc", "Top Models by ROC-AUC (Development Split)", COMPARISONS_DIR / "top_models_roc_auc_dev.png")
        plot_model_pooling_heatmap(dev_df, "f1", "F1-Score Heatmap across Representations & Models (Development Split)", COMPARISONS_DIR / "heatmap_f1_dev.png")
        plot_model_pooling_heatmap(dev_df, "roc_auc", "ROC-AUC Heatmap across Representations & Models (Development Split)", COMPARISONS_DIR / "heatmap_roc_auc_dev.png")

    logger.info("Classification benchmark completed successfully!")
    return final_results_df


if __name__ == "__main__":
    run_benchmark()
