"""
Controlled Classical E-DAIC Benchmark Runner.

Executes a controlled comparison of classical machine learning models using a single
pooling strategy, 'mean_std', across four acoustic feature representations:
  - raw_egemaps (46 features)
  - raw_mfcc (78 features)
  - boaw_egemaps (200 features)
  - boaw_mfcc (200 features)

Tasks:
1. Controlled Classification Benchmark (Target: PHQ_Binary)
   - 4 configurations using LinearSVC with StandardScaler inside pipeline.
   - Evaluated via 5-fold Stratified CV, 50-fold Repeated Stratified CV, and Dev split (n=56).
2. Controlled Regression Benchmark (Target: PHQ_Score)
   - 5 configurations: 4 with Ridge (with StandardScaler) + 1 with XGBoost (no scaling).
   - Plus 1 reference baseline: DummyRegressor (Mean).
   - Evaluated via 5-fold CV, 50-fold Repeated CV, and Dev split (n=56).

Guarantees:
- Strict output isolation under outputs/classical_controlled/.
- Zero overwriting of historical outputs.
- Predefined identical fold assignments shared across all configurations within each task.
- Zero data leakage: scalers fit strictly on training fold portions.
- Official test split is completely untouched (never loaded or evaluated).
- Zero model persistence (no pickle or weights saved).
- Raw regression predictions preserved for primary metrics (no silent clipping).
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold, RepeatedKFold

# Ensure code root is in sys.path
CODE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import (
    DATASET_DIR,
    LABELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    POSITIVE_CLASS_VALUE,
    FEATURE_REGISTRY,
)
from split_loader import load_split_dataset, load_official_splits, SplitData
from models.classification_models import get_classifier, extract_continuous_scores
from models.regression_models import get_regressor
from utils.metrics import (
    compute_classification_metrics,
    ClassificationMetrics,
    compute_regression_metrics,
    RegressionMetrics,
    concordance_correlation_coefficient,
)

# Setup Output Paths
CONTROLLED_DIR = OUTPUTS_DIR / "classical_controlled"
CLASS_DIR = CONTROLLED_DIR / "classification"
REG_DIR = CONTROLLED_DIR / "regression"
PRED_DIR = CONTROLLED_DIR / "predictions"
CONFIG_DIR = CONTROLLED_DIR / "configs"
LOG_DIR = CONTROLLED_DIR / "logs"
AUDIT_DIR = CONTROLLED_DIR / "audit"
SUMMARY_DIR = CONTROLLED_DIR / "summaries"

for d in [CONTROLLED_DIR, CLASS_DIR, REG_DIR, PRED_DIR, CONFIG_DIR, LOG_DIR, AUDIT_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logger setup
logger = logging.getLogger("controlled_benchmark")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Avoid duplicate handlers on re-run
if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(log_formatter)
    logger.addHandler(c_handler)
    
    f_handler = logging.FileHandler(LOG_DIR / "controlled_benchmark.log", mode="a", encoding="utf-8")
    f_handler.setFormatter(log_formatter)
    logger.addHandler(f_handler)

POOLING_STRATEGY = "mean_std"
REPRESENTATIONS = ["raw_egemaps", "raw_mfcc", "boaw_egemaps", "boaw_mfcc"]

CLASSIFICATION_CONFIGS = [
    {"rep": rep, "model": "linear_svc", "pooling": POOLING_STRATEGY}
    for rep in REPRESENTATIONS
]

REGRESSION_CONFIGS = [
    {"rep": "raw_egemaps", "model": "ridge", "pooling": POOLING_STRATEGY, "is_reference": False},
    {"rep": "raw_mfcc", "model": "ridge", "pooling": POOLING_STRATEGY, "is_reference": False},
    {"rep": "boaw_egemaps", "model": "ridge", "pooling": POOLING_STRATEGY, "is_reference": False},
    {"rep": "boaw_mfcc", "model": "ridge", "pooling": POOLING_STRATEGY, "is_reference": False},
    {"rep": "boaw_mfcc", "model": "xgboost", "pooling": POOLING_STRATEGY, "is_reference": False},
    # Reference Baseline
    {"rep": "raw_mfcc", "model": "dummy", "pooling": POOLING_STRATEGY, "is_reference": True},
]

N_SPLITS = 5
N_REPEATS = 10
TOTAL_REPEATED_FOLDS = N_SPLITS * N_REPEATS  # 50


def run_controlled_classification() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the controlled classification benchmark (4 configurations).
    Returns (summary_df, fold_results_df, repeated_cv_df, dev_results_df).
    """
    logger.info("=" * 70)
    logger.info("STARTING CONTROLLED CLASSIFICATION BENCHMARK (PHQ_Binary)")
    logger.info(f"Target: {TARGET_CLASSIFICATION}, Pooling: {POOLING_STRATEGY}, Model: LinearSVC")
    logger.info(f"Configurations to run: {len(CLASSIFICATION_CONFIGS)}")
    logger.info("=" * 70)

    # 1. Load train dataset once to establish ground truth participant IDs and predetermine folds
    ref_train = load_split_dataset(REPRESENTATIONS[0], POOLING_STRATEGY, split="train")
    train_pids = ref_train.participant_ids
    y_train_binary = ref_train.y_binary
    n_train = len(train_pids)
    assert n_train == 163, f"Expected 163 training participants, got {n_train}"

    # Verify participant alignment across all representations
    for rep in REPRESENTATIONS[1:]:
        t_data = load_split_dataset(rep, POOLING_STRATEGY, split="train")
        assert np.array_equal(t_data.participant_ids, train_pids), f"Participant ID mismatch in train for {rep}"
        assert np.array_equal(t_data.y_binary, y_train_binary), f"Target mismatch in train for {rep}"

    # 2. Precompute and freeze identical fold splits for all classification configurations
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv5_folds = list(skf.split(np.zeros(n_train), y_train_binary))

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    repeated_folds = list(rskf.split(np.zeros(n_train), y_train_binary))

    # Save fold assignments for full auditability
    fold_assignment_records = []
    for f_idx, (tr_idx, val_idx) in enumerate(cv5_folds, start=1):
        for pid in train_pids[val_idx]:
            fold_assignment_records.append({"Participant_ID": int(pid), "fold": f_idx})
    df_fold_assign = pd.DataFrame(fold_assignment_records).sort_values("Participant_ID")
    df_fold_assign.to_csv(CLASS_DIR / "cv5_fold_assignments.csv", index=False)

    summary_records = []
    fold_records = []
    repeated_records = []
    dev_records = []

    for cfg in CLASSIFICATION_CONFIGS:
        rep = cfg["rep"]
        model_name = cfg["model"]
        pooling = cfg["pooling"]
        config_id = f"{rep}__{pooling}__{model_name}"
        logger.info(f"\n--- Running Classification: {config_id} ---")

        # Load train & dev split data (NEVER load test!)
        train_data = load_split_dataset(rep, pooling, split="train")
        dev_data = load_split_dataset(rep, pooling, split="dev")
        feature_dim = train_data.num_features
        logger.info(f"Loaded {rep}: feature dimension = {feature_dim}, train n={train_data.num_participants}, dev n={dev_data.num_participants}")

        t0_total = time.time()

        # ----------------------------------------------------------------------
        # A. 5-Fold Stratified Cross-Validation
        # ----------------------------------------------------------------------
        oof_preds = np.zeros(n_train, dtype=np.int64)
        oof_scores = np.zeros(n_train, dtype=np.float64)
        oof_assigned = np.zeros(n_train, dtype=bool)
        oof_folds_col = np.zeros(n_train, dtype=int)

        cfg_fold_metrics: List[ClassificationMetrics] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(cv5_folds, start=1):
            t0_f = time.time()
            X_tr, y_tr = train_data.X[tr_idx], train_data.y_binary[tr_idx]
            X_va, y_va = train_data.X[val_idx], train_data.y_binary[val_idx]
            val_pids = train_data.participant_ids[val_idx]

            # Participant leakage check
            assert len(set(train_data.participant_ids[tr_idx]).intersection(set(val_pids))) == 0, "Leakage in fold!"

            pipe = get_classifier(model_name)
            pipe.fit(X_tr, y_tr)

            y_pred_va = pipe.predict(X_va)
            y_score_va = extract_continuous_scores(pipe, X_va)

            oof_preds[val_idx] = y_pred_va
            oof_scores[val_idx] = y_score_va
            oof_assigned[val_idx] = True
            oof_folds_col[val_idx] = fold_idx

            f_metrics = compute_classification_metrics(y_va, y_pred_va, y_score=y_score_va)
            cfg_fold_metrics.append(f_metrics)
            f_runtime = time.time() - t0_f

            fold_records.append({
                "config_id": config_id,
                "representation": rep,
                "pooling_strategy": pooling,
                "model_name": model_name,
                "fold": fold_idx,
                "train_samples": len(tr_idx),
                "val_samples": len(val_idx),
                "accuracy": f_metrics.accuracy,
                "balanced_accuracy": f_metrics.balanced_accuracy,
                "f1": f_metrics.f1,
                "precision": f_metrics.precision,
                "recall": f_metrics.recall,
                "roc_auc": f_metrics.roc_auc,
                "average_precision": f_metrics.average_precision,
                "tn": f_metrics.tn,
                "fp": f_metrics.fp,
                "fn": f_metrics.fn,
                "tp": f_metrics.tp,
                "runtime_sec": f_runtime,
            })
            del pipe

        assert np.all(oof_assigned), f"Incomplete OOF coverage for {config_id}"

        # Compute Pooled OOF Metrics across all 163 participants
        pooled_oof_metrics = compute_classification_metrics(
            y_true=train_data.y_binary,
            y_pred=oof_preds,
            y_score=oof_scores,
        )

        # Save OOF predictions DataFrame
        df_oof = pd.DataFrame({
            "Participant_ID": train_data.participant_ids,
            "fold": oof_folds_col,
            "y_true": train_data.y_binary,
            "y_pred": oof_preds,
            "y_score": oof_scores,
        }).sort_values("Participant_ID").reset_index(drop=True)

        oof_pred_path = PRED_DIR / f"{config_id}__cv5_oof__predictions.csv"
        df_oof.to_csv(oof_pred_path, index=False)

        # Fold-averaged metrics
        f1_mean = float(np.mean([m.f1 for m in cfg_fold_metrics]))
        f1_std = float(np.std([m.f1 for m in cfg_fold_metrics], ddof=1))
        bal_acc_mean = float(np.mean([m.balanced_accuracy for m in cfg_fold_metrics]))
        bal_acc_std = float(np.std([m.balanced_accuracy for m in cfg_fold_metrics], ddof=1))
        auc_vals = [m.roc_auc for m in cfg_fold_metrics if m.roc_auc is not None]
        auc_mean = float(np.mean(auc_vals)) if auc_vals else np.nan
        auc_std = float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else np.nan

        # ----------------------------------------------------------------------
        # B. 50-Fold Repeated Cross-Validation (10 repeats x 5 folds)
        # ----------------------------------------------------------------------
        logger.info(f"Running 50-fold repeated CV for {config_id}...")
        rep_f1s = []
        rep_aucs = []
        rep_bal_accs = []
        repeat_f1_means = []

        for r_idx in range(N_REPEATS):
            r_fold_f1s = []
            for f_idx in range(N_SPLITS):
                overall_idx = r_idx * N_SPLITS + f_idx
                tr_idx, val_idx = repeated_folds[overall_idx]

                X_tr, y_tr = train_data.X[tr_idx], train_data.y_binary[tr_idx]
                X_va, y_va = train_data.X[val_idx], train_data.y_binary[val_idx]

                pipe = get_classifier(model_name)
                pipe.fit(X_tr, y_tr)
                y_pred_va = pipe.predict(X_va)
                y_score_va = extract_continuous_scores(pipe, X_va)

                r_metrics = compute_classification_metrics(y_va, y_pred_va, y_score=y_score_va)
                rep_f1s.append(r_metrics.f1)
                r_fold_f1s.append(r_metrics.f1)
                if r_metrics.roc_auc is not None:
                    rep_aucs.append(r_metrics.roc_auc)
                rep_bal_accs.append(r_metrics.balanced_accuracy)

                repeated_records.append({
                    "config_id": config_id,
                    "representation": rep,
                    "pooling_strategy": pooling,
                    "model_name": model_name,
                    "repeat": r_idx + 1,
                    "fold": f_idx + 1,
                    "overall_fold_idx": overall_idx + 1,
                    "f1": r_metrics.f1,
                    "balanced_accuracy": r_metrics.balanced_accuracy,
                    "roc_auc": r_metrics.roc_auc,
                    "precision": r_metrics.precision,
                    "recall": r_metrics.recall,
                })
                del pipe
            repeat_f1_means.append(float(np.mean(r_fold_f1s)))

        rep_f1_mean = float(np.mean(rep_f1s))
        rep_f1_std_fold = float(np.std(rep_f1s, ddof=1))
        rep_f1_std_repeat = float(np.std(repeat_f1_means, ddof=1))
        rep_auc_mean = float(np.mean(rep_aucs)) if rep_aucs else np.nan
        rep_auc_std_fold = float(np.std(rep_aucs, ddof=1)) if len(rep_aucs) > 1 else np.nan

        # ----------------------------------------------------------------------
        # C. Development Set Evaluation (Fit on Train n=163, Evaluate on Dev n=56)
        # ----------------------------------------------------------------------
        logger.info(f"Evaluating on official Development split (n=56)...")
        pipe_dev = get_classifier(model_name)
        pipe_dev.fit(train_data.X, train_data.y_binary)

        y_pred_dev = pipe_dev.predict(dev_data.X)
        y_score_dev = extract_continuous_scores(pipe_dev, dev_data.X)

        dev_metrics = compute_classification_metrics(
            y_true=dev_data.y_binary,
            y_pred=y_pred_dev,
            y_score=y_score_dev,
        )

        df_dev_pred = pd.DataFrame({
            "Participant_ID": dev_data.participant_ids,
            "y_true": dev_data.y_binary,
            "y_pred": y_pred_dev,
            "y_score": y_score_dev,
        }).sort_values("Participant_ID").reset_index(drop=True)

        dev_pred_path = PRED_DIR / f"{config_id}__dev__predictions.csv"
        df_dev_pred.to_csv(dev_pred_path, index=False)
        del pipe_dev

        dev_records.append({
            "config_id": config_id,
            "representation": rep,
            "pooling_strategy": pooling,
            "model_name": model_name,
            "accuracy": dev_metrics.accuracy,
            "balanced_accuracy": dev_metrics.balanced_accuracy,
            "f1": dev_metrics.f1,
            "precision": dev_metrics.precision,
            "recall": dev_metrics.recall,
            "roc_auc": dev_metrics.roc_auc,
            "average_precision": dev_metrics.average_precision,
            "tn": dev_metrics.tn,
            "fp": dev_metrics.fp,
            "fn": dev_metrics.fn,
            "tp": dev_metrics.tp,
        })

        total_runtime = time.time() - t0_total

        # ----------------------------------------------------------------------
        # D. Save Config JSON Snapshot
        # ----------------------------------------------------------------------
        config_snapshot = {
            "config_id": config_id,
            "task": "classification",
            "representation": rep,
            "pooling_strategy": pooling,
            "model_name": model_name,
            "target": TARGET_CLASSIFICATION,
            "feature_dim": feature_dim,
            "train_samples": n_train,
            "dev_samples": len(dev_data.participant_ids),
            "pipeline": ["StandardScaler", "LinearSVC"],
            "model_params": {"max_iter": 2000, "random_state": RANDOM_SEED, "dual": "auto"},
            "random_seed": RANDOM_SEED,
            "runtime_sec": round(total_runtime, 3),
        }
        with open(CONFIG_DIR / f"{config_id}__config.json", "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, indent=2)

        # ----------------------------------------------------------------------
        # E. Summary Record
        # ----------------------------------------------------------------------
        summary_records.append({
            "config_id": config_id,
            "representation": rep,
            "model": model_name,
            "pooling": pooling,
            "feature_dim": feature_dim,
            # Pooled OOF Metrics (163 samples)
            "oof_f1": round(pooled_oof_metrics.f1, 4),
            "oof_balanced_accuracy": round(pooled_oof_metrics.balanced_accuracy, 4),
            "oof_roc_auc": round(pooled_oof_metrics.roc_auc, 4) if pooled_oof_metrics.roc_auc else np.nan,
            "oof_precision": round(pooled_oof_metrics.precision, 4),
            "oof_recall": round(pooled_oof_metrics.recall, 4),
            "oof_accuracy": round(pooled_oof_metrics.accuracy, 4),
            "oof_tn": pooled_oof_metrics.tn,
            "oof_fp": pooled_oof_metrics.fp,
            "oof_fn": pooled_oof_metrics.fn,
            "oof_tp": pooled_oof_metrics.tp,
            # 5-Fold Cross-Validation (Mean ± Std)
            "cv5_f1_mean": round(f1_mean, 4),
            "cv5_f1_std": round(f1_std, 4),
            "cv5_bal_acc_mean": round(bal_acc_mean, 4),
            "cv5_bal_acc_std": round(bal_acc_std, 4),
            "cv5_roc_auc_mean": round(auc_mean, 4),
            "cv5_roc_auc_std": round(auc_std, 4),
            # 50-Fold Repeated CV (Stability)
            "rep_f1_mean": round(rep_f1_mean, 4),
            "rep_f1_std_fold": round(rep_f1_std_fold, 4),
            "rep_f1_std_repeat": round(rep_f1_std_repeat, 4),
            "rep_roc_auc_mean": round(rep_auc_mean, 4),
            "rep_roc_auc_std_fold": round(rep_auc_std_fold, 4),
            # Development Set Metrics (56 samples)
            "dev_f1": round(dev_metrics.f1, 4),
            "dev_balanced_accuracy": round(dev_metrics.balanced_accuracy, 4),
            "dev_roc_auc": round(dev_metrics.roc_auc, 4) if dev_metrics.roc_auc else np.nan,
            "dev_precision": round(dev_metrics.precision, 4),
            "dev_recall": round(dev_metrics.recall, 4),
            "dev_accuracy": round(dev_metrics.accuracy, 4),
            "dev_tn": dev_metrics.tn,
            "dev_fp": dev_metrics.fp,
            "dev_fn": dev_metrics.fn,
            "dev_tp": dev_metrics.tp,
            "runtime_sec": round(total_runtime, 2),
        })

        logger.info(
            f"Result for {config_id}: OOF F1={pooled_oof_metrics.f1:.4f}, OOF AUC={pooled_oof_metrics.roc_auc:.4f} | "
            f"Dev F1={dev_metrics.f1:.4f}, Dev AUC={dev_metrics.roc_auc:.4f} | Runtime={total_runtime:.2f}s"
        )

    # Save CSVs
    df_summary = pd.DataFrame(summary_records)
    df_folds = pd.DataFrame(fold_records)
    df_repeated = pd.DataFrame(repeated_records)
    df_dev = pd.DataFrame(dev_records)

    df_summary.to_csv(CLASS_DIR / "summary.csv", index=False)
    df_folds.to_csv(CLASS_DIR / "fold_results.csv", index=False)
    df_repeated.to_csv(CLASS_DIR / "repeated_cv_results.csv", index=False)
    df_dev.to_csv(CLASS_DIR / "dev_results.csv", index=False)

    logger.info("Controlled Classification Benchmark finished successfully.")
    return df_summary, df_folds, df_repeated, df_dev


def run_controlled_regression() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the controlled regression benchmark (5 primary configurations + 1 dummy baseline).
    Returns (summary_df, fold_results_df, repeated_cv_df, dev_results_df).
    """
    logger.info("=" * 70)
    logger.info("STARTING CONTROLLED REGRESSION BENCHMARK (PHQ_Score, 0-24)")
    logger.info(f"Target: {TARGET_REGRESSION}, Pooling: {POOLING_STRATEGY}")
    logger.info(f"Configurations to run: {len(REGRESSION_CONFIGS)} (5 primary + 1 reference baseline)")
    logger.info("=" * 70)

    # 1. Load train dataset once to establish ground truth participant IDs and predetermine folds
    ref_train = load_split_dataset(REPRESENTATIONS[0], POOLING_STRATEGY, split="train")
    train_pids = ref_train.participant_ids
    y_train_score = ref_train.y_score.astype(np.float64)
    n_train = len(train_pids)
    assert n_train == 163, f"Expected 163 training participants, got {n_train}"

    # Verify participant alignment across all representations
    for rep in REPRESENTATIONS[1:]:
        t_data = load_split_dataset(rep, POOLING_STRATEGY, split="train")
        assert np.array_equal(t_data.participant_ids, train_pids), f"Participant ID mismatch in train for {rep}"
        assert np.array_equal(t_data.y_score, y_train_score), f"Target mismatch in train for {rep}"

    # 2. Precompute and freeze identical fold splits for all regression configurations
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv5_folds = list(kf.split(np.zeros(n_train), y_train_score))

    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    repeated_folds = list(rkf.split(np.zeros(n_train), y_train_score))

    # Save fold assignments for full auditability
    fold_assignment_records = []
    for f_idx, (tr_idx, val_idx) in enumerate(cv5_folds, start=1):
        for pid in train_pids[val_idx]:
            fold_assignment_records.append({"Participant_ID": int(pid), "fold": f_idx})
    df_fold_assign = pd.DataFrame(fold_assignment_records).sort_values("Participant_ID")
    df_fold_assign.to_csv(REG_DIR / "cv5_fold_assignments.csv", index=False)

    summary_records = []
    fold_records = []
    repeated_records = []
    dev_records = []

    for cfg in REGRESSION_CONFIGS:
        rep = cfg["rep"]
        model_name = cfg["model"]
        pooling = cfg["pooling"]
        is_reference = cfg["is_reference"]
        config_id = f"{rep}__{pooling}__{model_name}"
        if is_reference:
            config_id = f"reference_baseline__{config_id}"

        logger.info(f"\n--- Running Regression: {config_id} ---")

        # Load train & dev split data (NEVER load test!)
        train_data = load_split_dataset(rep, pooling, split="train")
        dev_data = load_split_dataset(rep, pooling, split="dev")
        feature_dim = train_data.num_features
        logger.info(f"Loaded {rep}: feature dimension = {feature_dim}, train n={train_data.num_participants}, dev n={dev_data.num_participants}")

        t0_total = time.time()

        # ----------------------------------------------------------------------
        # A. 5-Fold Cross-Validation (K-Fold)
        # ----------------------------------------------------------------------
        oof_preds = np.zeros(n_train, dtype=np.float64)
        oof_assigned = np.zeros(n_train, dtype=bool)
        oof_folds_col = np.zeros(n_train, dtype=int)

        cfg_fold_metrics: List[RegressionMetrics] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(cv5_folds, start=1):
            t0_f = time.time()
            X_tr, y_tr = train_data.X[tr_idx], train_data.y_score.astype(np.float64)[tr_idx]
            X_va, y_va = train_data.X[val_idx], train_data.y_score.astype(np.float64)[val_idx]
            val_pids = train_data.participant_ids[val_idx]

            # Participant leakage check
            assert len(set(train_data.participant_ids[tr_idx]).intersection(set(val_pids))) == 0, "Leakage in fold!"

            pipe = get_regressor(model_name)
            pipe.fit(X_tr, y_tr)

            # Raw unclipped predictions
            y_pred_va = pipe.predict(X_va).astype(np.float64)

            oof_preds[val_idx] = y_pred_va
            oof_assigned[val_idx] = True
            oof_folds_col[val_idx] = fold_idx

            f_metrics = compute_regression_metrics(y_va, y_pred_va)
            cfg_fold_metrics.append(f_metrics)
            f_runtime = time.time() - t0_f

            fold_records.append({
                "config_id": config_id,
                "representation": rep,
                "pooling_strategy": pooling,
                "model_name": model_name,
                "is_reference": is_reference,
                "fold": fold_idx,
                "train_samples": len(tr_idx),
                "val_samples": len(val_idx),
                "ccc": f_metrics.ccc,
                "rmse": f_metrics.rmse,
                "mae": f_metrics.mae,
                "r2": f_metrics.r2,
                "runtime_sec": f_runtime,
            })
            del pipe

        assert np.all(oof_assigned), f"Incomplete OOF coverage for {config_id}"

        # Compute Pooled OOF Metrics across all 163 participants
        pooled_oof_metrics = compute_regression_metrics(
            y_true=train_data.y_score.astype(np.float64),
            y_pred=oof_preds,
        )

        # Save OOF predictions DataFrame
        df_oof = pd.DataFrame({
            "Participant_ID": train_data.participant_ids,
            "fold": oof_folds_col,
            "y_true": train_data.y_score.astype(np.float64),
            "y_pred": oof_preds,
            "residual": train_data.y_score.astype(np.float64) - oof_preds,
        }).sort_values("Participant_ID").reset_index(drop=True)

        oof_pred_path = PRED_DIR / f"{config_id}__cv5_oof__predictions.csv"
        df_oof.to_csv(oof_pred_path, index=False)

        # Fold-averaged metrics
        ccc_mean = float(np.mean([m.ccc for m in cfg_fold_metrics]))
        ccc_std = float(np.std([m.ccc for m in cfg_fold_metrics], ddof=1))
        rmse_mean = float(np.mean([m.rmse for m in cfg_fold_metrics]))
        rmse_std = float(np.std([m.rmse for m in cfg_fold_metrics], ddof=1))
        mae_mean = float(np.mean([m.mae for m in cfg_fold_metrics]))
        mae_std = float(np.std([m.mae for m in cfg_fold_metrics], ddof=1))
        r2_mean = float(np.mean([m.r2 for m in cfg_fold_metrics]))
        r2_std = float(np.std([m.r2 for m in cfg_fold_metrics], ddof=1))

        # ----------------------------------------------------------------------
        # B. 50-Fold Repeated Cross-Validation (10 repeats x 5 folds)
        # ----------------------------------------------------------------------
        logger.info(f"Running 50-fold repeated CV for {config_id}...")
        rep_cccs = []
        rep_rmses = []
        rep_maes = []
        rep_r2s = []
        repeat_ccc_means = []

        for r_idx in range(N_REPEATS):
            r_fold_cccs = []
            for f_idx in range(N_SPLITS):
                overall_idx = r_idx * N_SPLITS + f_idx
                tr_idx, val_idx = repeated_folds[overall_idx]

                X_tr, y_tr = train_data.X[tr_idx], train_data.y_score.astype(np.float64)[tr_idx]
                X_va, y_va = train_data.X[val_idx], train_data.y_score.astype(np.float64)[val_idx]

                pipe = get_regressor(model_name)
                pipe.fit(X_tr, y_tr)
                y_pred_va = pipe.predict(X_va).astype(np.float64)

                r_metrics = compute_regression_metrics(y_va, y_pred_va)
                rep_cccs.append(r_metrics.ccc)
                r_fold_cccs.append(r_metrics.ccc)
                rep_rmses.append(r_metrics.rmse)
                rep_maes.append(r_metrics.mae)
                rep_r2s.append(r_metrics.r2)

                repeated_records.append({
                    "config_id": config_id,
                    "representation": rep,
                    "pooling_strategy": pooling,
                    "model_name": model_name,
                    "is_reference": is_reference,
                    "repeat": r_idx + 1,
                    "fold": f_idx + 1,
                    "overall_fold_idx": overall_idx + 1,
                    "ccc": r_metrics.ccc,
                    "rmse": r_metrics.rmse,
                    "mae": r_metrics.mae,
                    "r2": r_metrics.r2,
                })
                del pipe
            repeat_ccc_means.append(float(np.mean(r_fold_cccs)))

        rep_ccc_mean = float(np.mean(rep_cccs))
        rep_ccc_std_fold = float(np.std(rep_cccs, ddof=1))
        rep_ccc_std_repeat = float(np.std(repeat_ccc_means, ddof=1))
        rep_rmse_mean = float(np.mean(rep_rmses))
        rep_rmse_std_fold = float(np.std(rep_rmses, ddof=1))
        rep_mae_mean = float(np.mean(rep_maes))
        rep_mae_std_fold = float(np.std(rep_maes, ddof=1))

        # ----------------------------------------------------------------------
        # C. Development Set Evaluation (Fit on Train n=163, Evaluate on Dev n=56)
        # ----------------------------------------------------------------------
        logger.info(f"Evaluating on official Development split (n=56)...")
        pipe_dev = get_regressor(model_name)
        pipe_dev.fit(train_data.X, train_data.y_score.astype(np.float64))

        y_pred_dev = pipe_dev.predict(dev_data.X).astype(np.float64)  # Raw unclipped!
        y_dev_true = dev_data.y_score.astype(np.float64)

        dev_metrics = compute_regression_metrics(y_dev_true, y_pred_dev)

        df_dev_pred = pd.DataFrame({
            "Participant_ID": dev_data.participant_ids,
            "y_true": y_dev_true,
            "y_pred": y_pred_dev,
            "residual": y_dev_true - y_pred_dev,
        }).sort_values("Participant_ID").reset_index(drop=True)

        dev_pred_path = PRED_DIR / f"{config_id}__dev__predictions.csv"
        df_dev_pred.to_csv(dev_pred_path, index=False)
        del pipe_dev

        dev_records.append({
            "config_id": config_id,
            "representation": rep,
            "pooling_strategy": pooling,
            "model_name": model_name,
            "is_reference": is_reference,
            "ccc": dev_metrics.ccc,
            "rmse": dev_metrics.rmse,
            "mae": dev_metrics.mae,
            "r2": dev_metrics.r2,
        })

        total_runtime = time.time() - t0_total

        # ----------------------------------------------------------------------
        # D. Save Config JSON Snapshot
        # ----------------------------------------------------------------------
        config_snapshot = {
            "config_id": config_id,
            "task": "regression",
            "representation": rep,
            "pooling_strategy": pooling,
            "model_name": model_name,
            "is_reference": is_reference,
            "target": TARGET_REGRESSION,
            "feature_dim": feature_dim,
            "train_samples": n_train,
            "dev_samples": len(dev_data.participant_ids),
            "pipeline": ["StandardScaler", "Ridge"] if model_name == "ridge" else ([model_name]),
            "random_seed": RANDOM_SEED,
            "runtime_sec": round(total_runtime, 3),
        }
        with open(CONFIG_DIR / f"{config_id}__config.json", "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, indent=2)

        # ----------------------------------------------------------------------
        # E. Summary Record
        # ----------------------------------------------------------------------
        summary_records.append({
            "config_id": config_id,
            "representation": rep,
            "model": model_name,
            "pooling": pooling,
            "is_reference": is_reference,
            "feature_dim": feature_dim,
            # Pooled OOF Metrics (163 samples)
            "oof_ccc": round(pooled_oof_metrics.ccc, 4),
            "oof_rmse": round(pooled_oof_metrics.rmse, 4),
            "oof_mae": round(pooled_oof_metrics.mae, 4),
            "oof_r2": round(pooled_oof_metrics.r2, 4),
            # 5-Fold Cross-Validation (Mean ± Std)
            "cv5_ccc_mean": round(ccc_mean, 4),
            "cv5_ccc_std": round(ccc_std, 4),
            "cv5_rmse_mean": round(rmse_mean, 4),
            "cv5_rmse_std": round(rmse_std, 4),
            "cv5_mae_mean": round(mae_mean, 4),
            "cv5_mae_std": round(mae_std, 4),
            "cv5_r2_mean": round(r2_mean, 4),
            "cv5_r2_std": round(r2_std, 4),
            # 50-Fold Repeated CV (Stability)
            "rep_ccc_mean": round(rep_ccc_mean, 4),
            "rep_ccc_std_fold": round(rep_ccc_std_fold, 4),
            "rep_ccc_std_repeat": round(rep_ccc_std_repeat, 4),
            "rep_rmse_mean": round(rep_rmse_mean, 4),
            "rep_rmse_std_fold": round(rep_rmse_std_fold, 4),
            "rep_mae_mean": round(rep_mae_mean, 4),
            "rep_mae_std_fold": round(rep_mae_std_fold, 4),
            # Development Set Metrics (56 samples)
            "dev_ccc": round(dev_metrics.ccc, 4),
            "dev_rmse": round(dev_metrics.rmse, 4),
            "dev_mae": round(dev_metrics.mae, 4),
            "dev_r2": round(dev_metrics.r2, 4),
            "runtime_sec": round(total_runtime, 2),
        })

        logger.info(
            f"Result for {config_id}: OOF CCC={pooled_oof_metrics.ccc:.4f}, OOF RMSE={pooled_oof_metrics.rmse:.4f} | "
            f"Dev CCC={dev_metrics.ccc:.4f}, Dev RMSE={dev_metrics.rmse:.4f} | Runtime={total_runtime:.2f}s"
        )

    # Save CSVs
    df_summary = pd.DataFrame(summary_records)
    df_folds = pd.DataFrame(fold_records)
    df_repeated = pd.DataFrame(repeated_records)
    df_dev = pd.DataFrame(dev_records)

    df_summary.to_csv(REG_DIR / "summary.csv", index=False)
    df_folds.to_csv(REG_DIR / "fold_results.csv", index=False)
    df_repeated.to_csv(REG_DIR / "repeated_cv_results.csv", index=False)
    df_dev.to_csv(REG_DIR / "dev_results.csv", index=False)

    logger.info("Controlled Regression Benchmark finished successfully.")
    return df_summary, df_folds, df_repeated, df_dev


def generate_master_comparison_tables(df_class_summary: pd.DataFrame, df_reg_summary: pd.DataFrame):
    """
    Generates unified master CSV and Markdown tables combining classification and regression summaries.
    """
    master_csv_path = SUMMARY_DIR / "master_comparison.csv"
    master_md_path = SUMMARY_DIR / "master_comparison.md"

    # Save classification and regression summary tables into markdown
    md_content = [
        "# Controlled Classical E-DAIC Benchmark — Master Results Summary\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "**Protocol**: Temporal Pooling = `mean_std` across all models and acoustic representations.\n",
        "**Splits**: Training ($n=163$), Development ($n=56$), Official Test ($n=56$, strictly unblinded/untouched).\n",
        "\n---\n",
        "## 1. Controlled Classification Benchmark (`PHQ_Binary`)\n",
        r"Target: Binary depression status ($\text{PHQ} \ge 10$). Standard Decision Threshold: $0.0$ (via decision function).\n\n",
        "| Configuration ID | Feature Dim | OOF F1 | OOF Bal. Acc | OOF ROC-AUC | OOF Prec / Rec | 5-Fold F1 (Mean ± Std) | Rep-CV F1 (Mean ± Std) | Dev F1 | Dev Bal. Acc | Dev ROC-AUC | Runtime (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df_class_summary.iterrows():
        md_content.append(
            f"| `{r['config_id']}` | {r['feature_dim']} | **{r['oof_f1']:.4f}** | {r['oof_balanced_accuracy']:.4f} | "
            f"{r['oof_roc_auc']:.4f} | {r['oof_precision']:.4f} / {r['oof_recall']:.4f} | "
            f"{r['cv5_f1_mean']:.4f} ± {r['cv5_f1_std']:.4f} | {r['rep_f1_mean']:.4f} ± {r['rep_f1_std_fold']:.4f} | "
            f"**{r['dev_f1']:.4f}** | {r['dev_balanced_accuracy']:.4f} | {r['dev_roc_auc']:.4f} | {r['runtime_sec']:.2f} |"
        )

    md_content.extend([
        "\n---\n",
        "## 2. Controlled Regression Benchmark (`PHQ_Score`)\n",
        "Target: Continuous PHQ-8 score (0–24). Primary Metric: Concordance Correlation Coefficient (CCC).\n\n",
        "| Configuration ID | Feature Dim | OOF CCC | OOF RMSE | OOF MAE | OOF R² | 5-Fold CCC (Mean ± Std) | Rep-CV CCC (Mean ± Std) | Dev CCC | Dev RMSE | Dev MAE | Dev R² | Runtime (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, r in df_reg_summary.iterrows():
        ref_mark = " *(Reference Baseline)*" if r.get("is_reference", False) else ""
        md_content.append(
            f"| `{r['config_id']}`{ref_mark} | {r['feature_dim']} | **{r['oof_ccc']:.4f}** | {r['oof_rmse']:.4f} | "
            f"{r['oof_mae']:.4f} | {r['oof_r2']:.4f} | {r['cv5_ccc_mean']:.4f} ± {r['cv5_ccc_std']:.4f} | "
            f"{r['rep_ccc_mean']:.4f} ± {r['rep_ccc_std_fold']:.4f} | **{r['dev_ccc']:.4f}** | {r['dev_rmse']:.4f} | "
            f"{r['dev_mae']:.4f} | {r['dev_r2']:.4f} | {r['runtime_sec']:.2f} |"
        )

    with open(master_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    # Also build a combined CSV
    df_class_copy = df_class_summary.copy()
    df_class_copy["benchmark_task"] = "classification"
    df_reg_copy = df_reg_summary.copy()
    df_reg_copy["benchmark_task"] = "regression"

    df_combined = pd.concat([df_class_copy, df_reg_copy], ignore_index=True)
    df_combined.to_csv(master_csv_path, index=False)
    logger.info(f"Master comparison tables written to {master_md_path} and {master_csv_path}")


def main():
    logger.info("Initializing Controlled Classical Benchmark Suite...")
    t_start = time.time()

    # Classification
    df_class_summary, df_class_folds, df_class_repeated, df_class_dev = run_controlled_classification()

    # Regression
    df_reg_summary, df_reg_folds, df_reg_repeated, df_reg_dev = run_controlled_regression()

    # Master tables
    generate_master_comparison_tables(df_class_summary, df_reg_summary)

    total_time = time.time() - t_start
    logger.info(f"All controlled benchmarks completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
