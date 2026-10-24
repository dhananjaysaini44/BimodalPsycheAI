"""
Phase 18B: Comprehensive Benchmarking Script for DistilBERT Segment-Level Representations.

Executes:
1. Controlled LinearSVC classification with in-fold scaling and Train OOF threshold calibration.
2. Controlled Ridge regression and RBF SVR with in-fold scaling and Train CV CCC hyperparameter selection.
3. Benchmarking across:
   - Whole-interview baselines (Full cohort N=163/56 and Matched cohort N=157/54)
   - Single-segment models (Early, Middle, Late x 4 aggregations)
   - Three-segment concatenation (Early + Middle + Late x 4 aggregations)
4. Non-parametric participant-level bootstrap confidence intervals (95% CI, B=1,000).
5. Paired participant-level bootstrap difference tests against Phase 18A MiniLM champions.
6. Diagnostic visualization generation and comprehensive research reporting.
"""

import copy
import datetime
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase18B_Benchmark")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "ml-training" / "dataset" / "text_distilbert_v1"
FEATURES_DIR = DATASET_DIR / "features"
OUTPUT_DIR = REPO_ROOT / "ml-training" / "outputs" / "text_distilbert_v1"
LABELS_DIR = REPO_ROOT / "ml-training" / "edaic-woz" / "labels"

PHASE_18A_PRED_DIR = REPO_ROOT / "ml-training" / "outputs" / "text_segments_v1" / "predictions"
PHASE_17A_PRED_DIR = REPO_ROOT / "ml-training" / "outputs" / "text_baselines_final_v1" / "development"

RANDOM_SEED = 42


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Lin's Concordance Correlation Coefficient using ddof=0 population moments."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    if np.array_equal(y_true, y_pred):
        return 1.0
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true, ddof=0))
    var_pred = float(np.var(y_pred, ddof=0))
    covariance = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    numerator = 2.0 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """Computes standard binary classification metrics."""
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    if len(np.unique(y_true)) > 1:
        auc_val = float(roc_auc_score(y_true, y_score))
        ap_val = float(average_precision_score(y_true, y_score))
    else:
        auc_val = 0.5
        ap_val = float(np.mean(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "roc_auc": auc_val,
        "pr_auc": ap_val,
        "precision": prec,
        "recall": rec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": f"{tn}/{fp}/{fn}/{tp}",
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Computes continuous regression metrics."""
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    var_t = float(np.var(y_true, ddof=0))
    var_p = float(np.var(y_pred, ddof=0))
    if var_t > 0 and var_p > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "ccc": ccc,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """Computes 95% non-parametric bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    boot_vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_sample = y_true[idx]
        yp_sample = y_pred[idx]
        # In classification, ensure sample contains at least one positive and one negative
        if len(np.unique(yt_sample)) < 2:
            continue
        try:
            val = metric_fn(yt_sample, yp_sample)
            if np.isfinite(val):
                boot_vals.append(val)
        except Exception:
            continue
    if len(boot_vals) < 50:
        return 0.0, 0.0
    return float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))


def paired_bootstrap_difference(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred_distilbert: np.ndarray,
    y_pred_minilm: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Computes paired bootstrap difference between DistilBERT and MiniLM (Delta = DistilBERT - MiniLM)."""
    assert len(y_true) == len(y_pred_distilbert) == len(y_pred_minilm)
    n = len(y_true)
    diffs = []
    rng = np.random.RandomState(seed)

    point_distilbert = metric_fn(y_true, y_pred_distilbert)
    point_minilm = metric_fn(y_true, y_pred_minilm)
    point_diff = point_distilbert - point_minilm

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yd = y_pred_distilbert[idx]
        ym = y_pred_minilm[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            val_d = metric_fn(yt, yd)
            val_m = metric_fn(yt, ym)
            if np.isfinite(val_d) and np.isfinite(val_m):
                diffs.append(val_d - val_m)
        except Exception:
            continue

    if len(diffs) >= 50:
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
        p_distilbert_wins = float(np.mean(np.array(diffs) > 0))
    else:
        ci_lower, ci_upper, p_distilbert_wins = 0.0, 0.0, 0.5

    return {
        "point_diff": float(point_diff),
        "point_distilbert": float(point_distilbert),
        "point_minilm": float(point_minilm),
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "p_distilbert_wins": p_distilbert_wins,
    }


def load_dataset_labels() -> Dict[str, Any]:
    """Loads official participant labels for train and dev splits."""
    train_labels = pd.read_csv(LABELS_DIR / "train_split.csv").sort_values("Participant_ID")
    dev_labels = pd.read_csv(LABELS_DIR / "dev_split.csv").sort_values("Participant_ID")

    train_label_map_cls = dict(zip(train_labels["Participant_ID"], train_labels["PHQ_Binary"]))
    train_label_map_reg = dict(zip(train_labels["Participant_ID"], train_labels["PHQ_Score"]))

    dev_label_map_cls = dict(zip(dev_labels["Participant_ID"], dev_labels["PHQ_Binary"]))
    dev_label_map_reg = dict(zip(dev_labels["Participant_ID"], dev_labels["PHQ_Score"]))

    return {
        "train_labels": train_labels,
        "dev_labels": dev_labels,
        "train_cls": train_label_map_cls,
        "train_reg": train_label_map_reg,
        "dev_cls": dev_label_map_cls,
        "dev_reg": dev_label_map_reg,
    }


def load_feature_split(
    rep_name: str,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads X_train, y_train_cls, y_train_reg, X_dev, y_dev_cls, y_dev_reg, train_pids, dev_pids."""
    train_mat = np.load(FEATURES_DIR / "train" / f"{rep_name}.npy")
    dev_mat = np.load(FEATURES_DIR / "dev" / f"{rep_name}.npy")

    train_pids = pd.read_csv(FEATURES_DIR / "train" / f"{rep_name}__pids.csv")["Participant_ID"].values
    dev_pids = pd.read_csv(FEATURES_DIR / "dev" / f"{rep_name}__pids.csv")["Participant_ID"].values

    y_train_cls = np.array([data["train_cls"][p] for p in train_pids], dtype=int)
    y_train_reg = np.array([data["train_reg"][p] for p in train_pids], dtype=float)

    y_dev_cls = np.array([data["dev_cls"][p] for p in dev_pids], dtype=int)
    y_dev_reg = np.array([data["dev_reg"][p] for p in dev_pids], dtype=float)

    return train_mat, y_train_cls, y_train_reg, dev_mat, y_dev_cls, y_dev_reg, train_pids, dev_pids


def train_eval_classification_linear_svc(
    rep_name: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs controlled LinearSVC classification tuning and evaluation."""
    X_train, y_train, _, X_dev, y_dev, _, train_pids, dev_pids = load_feature_split(rep_name, data)

    c_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    cw_grid = [None, "balanced"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    best_config = None
    best_train_f1 = -1.0
    best_train_auc = -1.0
    best_oof_scores = None
    best_calibrated_tau = 0.0
    grid_records = []

    for C in c_grid:
        for cw in cw_grid:
            oof_scores = np.zeros(len(y_train), dtype=np.float64)

            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_va = scaler.transform(X_train[val_idx])

                clf = LinearSVC(
                    C=C,
                    class_weight=cw,
                    dual="auto",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                )
                clf.fit(X_tr, y_train[tr_idx])
                oof_scores[val_idx] = clf.decision_function(X_va)

            # Train OOF threshold calibration
            min_s, max_s = np.percentile(oof_scores, 1.0), np.percentile(oof_scores, 99.0)
            candidates = np.linspace(min_s, max_s, 300)
            cur_best_tau = 0.0
            cur_best_f1 = -1.0

            for tau in candidates:
                f = f1_score(y_train, (oof_scores > tau).astype(int), pos_label=1, zero_division=0)
                if f > cur_best_f1:
                    cur_best_f1 = f
                    cur_best_tau = float(tau)

            cur_pred = (oof_scores > cur_best_tau).astype(int)
            cur_metrics = compute_classification_metrics(y_train, cur_pred, oof_scores)

            record = {
                "representation": rep_name,
                "C": C,
                "class_weight": str(cw),
                "calibrated_tau": cur_best_tau,
                "train_oof_f1": cur_metrics["f1"],
                "train_oof_bal_acc": cur_metrics["balanced_accuracy"],
                "train_oof_auc": cur_metrics["roc_auc"],
            }
            grid_records.append(record)

            # Tie-break rule: highest F1, then highest ROC-AUC
            if (cur_metrics["f1"] > best_train_f1) or (
                np.isclose(cur_metrics["f1"], best_train_f1, atol=1e-5) and cur_metrics["roc_auc"] > best_train_auc
            ):
                best_train_f1 = cur_metrics["f1"]
                best_train_auc = cur_metrics["roc_auc"]
                best_config = (C, cw)
                best_oof_scores = oof_scores.copy()
                best_calibrated_tau = cur_best_tau

    opt_C, opt_cw = best_config

    # Refit optimal model on full Train and evaluate once on Development
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_dev_scaled = scaler_full.transform(X_dev)

    clf_opt = LinearSVC(
        C=opt_C,
        class_weight=opt_cw,
        dual="auto",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    clf_opt.fit(X_train_scaled, y_train)

    dev_scores = clf_opt.decision_function(X_dev_scaled)
    dev_pred = (dev_scores > best_calibrated_tau).astype(int)
    dev_metrics = compute_classification_metrics(y_dev, dev_pred, dev_scores)

    # Compute 95% bootstrap CIs
    ci_f1 = bootstrap_ci(lambda yt, yp: float(f1_score(yt, yp, pos_label=1, zero_division=0)), y_dev, dev_pred)
    ci_auc = bootstrap_ci(lambda yt, ys: float(roc_auc_score(yt, ys)), y_dev, dev_scores)
    ci_bal_acc = bootstrap_ci(lambda yt, yp: float(balanced_accuracy_score(yt, yp)), y_dev, dev_pred)

    dev_metrics["f1_ci95"] = list(ci_f1)
    dev_metrics["roc_auc_ci95"] = list(ci_auc)
    dev_metrics["bal_acc_ci95"] = list(ci_bal_acc)

    train_oof_pred = (best_oof_scores > best_calibrated_tau).astype(int)
    train_oof_metrics = compute_classification_metrics(y_train, train_oof_pred, best_oof_scores)

    # Save participant predictions
    df_oof = pd.DataFrame({
        "Participant_ID": train_pids,
        "split": "train",
        "representation": rep_name,
        "y_true": y_train,
        "y_pred": train_oof_pred,
        "y_score": best_oof_scores,
        "calibrated_tau": best_calibrated_tau,
    })
    df_oof.to_csv(OUTPUT_DIR / "predictions" / f"oof_cls__{rep_name}__linear_svc.csv", index=False)

    df_dev = pd.DataFrame({
        "Participant_ID": dev_pids,
        "split": "dev",
        "representation": rep_name,
        "y_true": y_dev,
        "y_pred": dev_pred,
        "y_score": dev_scores,
        "calibrated_tau": best_calibrated_tau,
    })
    df_dev.to_csv(OUTPUT_DIR / "predictions" / f"dev_cls__{rep_name}__linear_svc.csv", index=False)

    pd.DataFrame(grid_records).to_csv(OUTPUT_DIR / "cv_results" / f"grid_cls__{rep_name}__linear_svc.csv", index=False)

    return {
        "representation": rep_name,
        "task": "classification",
        "model": "LinearSVC",
        "n_train": len(train_pids),
        "n_dev": len(dev_pids),
        "dimension": X_train.shape[1],
        "opt_C": opt_C,
        "opt_class_weight": str(opt_cw),
        "calibrated_tau": best_calibrated_tau,
        "train_oof_metrics": train_oof_metrics,
        "dev_metrics": dev_metrics,
        "dev_scores": dev_scores,
        "dev_pred": dev_pred,
        "y_dev": y_dev,
    }


def train_eval_regression_ridge(
    rep_name: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs controlled Ridge regression tuning and evaluation."""
    X_train, _, y_train, X_dev, _, y_dev, train_pids, dev_pids = load_feature_split(rep_name, data)

    alpha_grid = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    best_alpha = 1.0
    best_train_ccc = -999.0
    best_oof_preds = None
    grid_records = []

    for alpha in alpha_grid:
        oof_preds = np.zeros(len(y_train), dtype=np.float64)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train[tr_idx])
            X_va = scaler.transform(X_train[val_idx])

            reg = Ridge(alpha=alpha, solver="auto", random_state=RANDOM_SEED)
            reg.fit(X_tr, y_train[tr_idx])
            oof_preds[val_idx] = reg.predict(X_va)

        cur_metrics = compute_regression_metrics(y_train, oof_preds)
        grid_records.append({
            "representation": rep_name,
            "alpha": alpha,
            "train_cv_ccc": cur_metrics["ccc"],
            "train_cv_rmse": cur_metrics["rmse"],
            "train_cv_r2": cur_metrics["r2"],
        })

        if cur_metrics["ccc"] > best_train_ccc:
            best_train_ccc = cur_metrics["ccc"]
            best_alpha = alpha
            best_oof_preds = oof_preds.copy()

    # Refit optimal model on full Train and evaluate once on Development
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_dev_scaled = scaler_full.transform(X_dev)

    reg_opt = Ridge(alpha=best_alpha, solver="auto", random_state=RANDOM_SEED)
    reg_opt.fit(X_train_scaled, y_train)

    dev_preds = reg_opt.predict(X_dev_scaled)
    dev_metrics = compute_regression_metrics(y_dev, dev_preds)

    ci_ccc = bootstrap_ci(concordance_correlation_coefficient, y_dev, dev_preds)
    ci_rmse = bootstrap_ci(lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))), y_dev, dev_preds)

    dev_metrics["ccc_ci95"] = list(ci_ccc)
    dev_metrics["rmse_ci95"] = list(ci_rmse)

    train_oof_metrics = compute_regression_metrics(y_train, best_oof_preds)

    # Save participant predictions
    df_oof = pd.DataFrame({
        "Participant_ID": train_pids,
        "split": "train",
        "representation": rep_name,
        "y_true": y_train,
        "y_pred": best_oof_preds,
    })
    df_oof.to_csv(OUTPUT_DIR / "predictions" / f"oof_reg__{rep_name}__ridge.csv", index=False)

    df_dev = pd.DataFrame({
        "Participant_ID": dev_pids,
        "split": "dev",
        "representation": rep_name,
        "y_true": y_dev,
        "y_pred": dev_preds,
    })
    df_dev.to_csv(OUTPUT_DIR / "predictions" / f"dev_reg__{rep_name}__ridge.csv", index=False)

    pd.DataFrame(grid_records).to_csv(OUTPUT_DIR / "cv_results" / f"grid_reg__{rep_name}__ridge.csv", index=False)

    return {
        "representation": rep_name,
        "task": "regression",
        "model": "Ridge",
        "n_train": len(train_pids),
        "n_dev": len(dev_pids),
        "dimension": X_train.shape[1],
        "opt_alpha": best_alpha,
        "train_oof_metrics": train_oof_metrics,
        "dev_metrics": dev_metrics,
        "dev_preds": dev_preds,
        "y_dev": y_dev,
    }


def train_eval_regression_svr(
    rep_name: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs controlled RBF SVR tuning and evaluation on key representations."""
    X_train, _, y_train, X_dev, _, y_dev, train_pids, dev_pids = load_feature_split(rep_name, data)

    c_grid = [0.1, 1.0, 5.0, 10.0, 50.0]
    gamma_grid = ["scale", "auto", 0.001, 0.01, 0.1]
    epsilon_grid = [0.01, 0.1, 0.5, 1.0]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    best_config = None
    best_train_ccc = -999.0
    best_oof_preds = None
    grid_records = []

    for C in c_grid:
        for gamma in gamma_grid:
            for eps in epsilon_grid:
                oof_preds = np.zeros(len(y_train), dtype=np.float64)

                for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train[tr_idx])
                    X_va = scaler.transform(X_train[val_idx])

                    svr = SVR(C=C, gamma=gamma, epsilon=eps)
                    svr.fit(X_tr, y_train[tr_idx])
                    oof_preds[val_idx] = svr.predict(X_va)

                cur_metrics = compute_regression_metrics(y_train, oof_preds)
                grid_records.append({
                    "representation": rep_name,
                    "C": C,
                    "gamma": str(gamma),
                    "epsilon": eps,
                    "train_cv_ccc": cur_metrics["ccc"],
                    "train_cv_rmse": cur_metrics["rmse"],
                    "train_cv_r2": cur_metrics["r2"],
                })

                if cur_metrics["ccc"] > best_train_ccc:
                    best_train_ccc = cur_metrics["ccc"]
                    best_config = (C, gamma, eps)
                    best_oof_preds = oof_preds.copy()

    opt_C, opt_gamma, opt_eps = best_config

    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_dev_scaled = scaler_full.transform(X_dev)

    svr_opt = SVR(C=opt_C, gamma=opt_gamma, epsilon=opt_eps)
    svr_opt.fit(X_train_scaled, y_train)

    dev_preds = svr_opt.predict(X_dev_scaled)
    dev_metrics = compute_regression_metrics(y_dev, dev_preds)

    ci_ccc = bootstrap_ci(concordance_correlation_coefficient, y_dev, dev_preds)
    ci_rmse = bootstrap_ci(lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))), y_dev, dev_preds)

    dev_metrics["ccc_ci95"] = list(ci_ccc)
    dev_metrics["rmse_ci95"] = list(ci_rmse)

    train_oof_metrics = compute_regression_metrics(y_train, best_oof_preds)

    # Save participant predictions
    df_oof = pd.DataFrame({
        "Participant_ID": train_pids,
        "split": "train",
        "representation": rep_name,
        "y_true": y_train,
        "y_pred": best_oof_preds,
    })
    df_oof.to_csv(OUTPUT_DIR / "predictions" / f"oof_reg__{rep_name}__rbf_svr.csv", index=False)

    df_dev = pd.DataFrame({
        "Participant_ID": dev_pids,
        "split": "dev",
        "representation": rep_name,
        "y_true": y_dev,
        "y_pred": dev_preds,
    })
    df_dev.to_csv(OUTPUT_DIR / "predictions" / f"dev_reg__{rep_name}__rbf_svr.csv", index=False)

    pd.DataFrame(grid_records).to_csv(OUTPUT_DIR / "cv_results" / f"grid_reg__{rep_name}__rbf_svr.csv", index=False)

    return {
        "representation": rep_name,
        "task": "regression",
        "model": "RBF_SVR",
        "n_train": len(train_pids),
        "n_dev": len(dev_pids),
        "dimension": X_train.shape[1],
        "opt_C": opt_C,
        "opt_gamma": str(opt_gamma),
        "opt_epsilon": opt_eps,
        "train_oof_metrics": train_oof_metrics,
        "dev_metrics": dev_metrics,
        "dev_preds": dev_preds,
        "y_dev": y_dev,
    }


def generate_diagnostic_plots(
    df_results: pd.DataFrame,
    paired_diffs: List[Dict[str, Any]],
    cls_results: Dict[str, Dict[str, Any]],
) -> None:
    """Generates all required diagnostic visualizations for Phase 18B."""
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. DistilBERT vs MiniLM Classification Comparison
    # Compare matching representations where Phase 18A results exist
    fig, ax = plt.subplots(figsize=(12, 6))
    cls_df = df_results[df_results["model"] == "LinearSVC"].copy()
    
    # We plot F1 and ROC-AUC for DistilBERT across representations
    x_labels = cls_df["representation"].tolist()
    x_pos = np.arange(len(x_labels))
    width = 0.35

    ax.bar(x_pos - width/2, cls_df["dev_f1"], width, label="Dev F1 (DistilBERT)", color="#2b5c8f", alpha=0.85)
    ax.bar(x_pos + width/2, cls_df["dev_roc_auc"], width, label="Dev ROC-AUC (DistilBERT)", color="#d95f02", alpha=0.85)

    # Add reference lines for Phase 18A champions
    ax.axhline(0.5161, color="#1b9e77", linestyle="--", linewidth=1.5, label="Phase 18A MiniLM middle__mean F1 (0.5161)")
    ax.axhline(0.4571, color="#7570b3", linestyle=":", linewidth=1.5, label="Phase 17A char_tfidf F1 (0.4571)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Metric Value")
    ax.set_title("Phase 18B: DistilBERT LinearSVC Classification Performance across Representations (Dev Split)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "distilbert_classification_benchmark.png", dpi=300)
    plt.close()

    # 2. DistilBERT Regression Performance across Representations (Ridge)
    fig, ax = plt.subplots(figsize=(12, 6))
    reg_df = df_results[df_results["model"] == "Ridge"].copy()
    
    x_labels_reg = reg_df["representation"].tolist()
    x_pos_reg = np.arange(len(x_labels_reg))

    ax.bar(x_pos_reg - width/2, reg_df["dev_ccc"], width, label="Dev CCC (DistilBERT Ridge)", color="#1b9e77", alpha=0.85)
    ax.bar(x_pos_reg + width/2, reg_df["dev_rmse"] / 10.0, width, label="Dev RMSE / 10 (DistilBERT Ridge)", color="#e7298a", alpha=0.85)

    ax.axhline(0.3958, color="#2b5c8f", linestyle="--", linewidth=1.5, label="Phase 18A MiniLM whole__median CCC (0.3958)")
    ax.axhline(0.3707, color="#d95f02", linestyle=":", linewidth=1.5, label="Phase 17B.1 MiniLM RBF SVR CCC (0.3707)")

    ax.set_xticks(x_pos_reg)
    ax.set_xticklabels(x_labels_reg, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Metric Value")
    ax.set_title("Phase 18B: DistilBERT Ridge Regression Performance across Representations (Dev Split)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "distilbert_regression_benchmark.png", dpi=300)
    plt.close()

    # 3. Early vs Middle vs Late vs Whole Chronological Trajectory
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    segments = ["whole", "early", "middle", "late", "concat"]
    aggs = ["mean", "median", "trimmed", "mean_std"]
    
    for agg in aggs:
        f1_vals = []
        ccc_vals = []
        for seg in segments:
            rep = f"{seg}__{agg}"
            sub_c = cls_df[cls_df["representation"] == rep]
            sub_r = reg_df[reg_df["representation"] == rep]
            f1_vals.append(sub_c["dev_f1"].values[0] if len(sub_c) > 0 else np.nan)
            ccc_vals.append(sub_r["dev_ccc"].values[0] if len(sub_r) > 0 else np.nan)
        ax1.plot(segments, f1_vals, marker="o", label=f"Agg: {agg}")
        ax2.plot(segments, ccc_vals, marker="s", label=f"Agg: {agg}")

    ax1.set_title("Classification F1 across Chronological Segments (LinearSVC)", fontweight="bold")
    ax1.set_ylabel("Development F1 Score")
    ax1.set_xlabel("Segment / Representation")
    ax1.legend()

    ax2.set_title("Regression CCC across Chronological Segments (Ridge)", fontweight="bold")
    ax2.set_ylabel("Development Lin's CCC")
    ax2.set_xlabel("Segment / Representation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "chronological_trajectory_distilbert.png", dpi=300)
    plt.close()

    # 4. Confusion Matrices for Selected Models
    key_models = ["middle__mean", "whole__mean", "whole__mean_std", "concat__mean"]
    available_keys = [k for k in key_models if k in cls_results]
    n_cm = len(available_keys)
    if n_cm > 0:
        fig, axes = plt.subplots(1, n_cm, figsize=(4 * n_cm, 3.5))
        if n_cm == 1:
            axes = [axes]
        for ax, k in zip(axes, available_keys):
            res = cls_results[k]
            y_true = res["y_dev"]
            y_pred = res["dev_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"{k}\n(F1={res['dev_metrics']['f1']:.4f})", fontsize=10, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Non-Dep (0)", "Dep (1)"])
            ax.set_yticklabels(["Non-Dep (0)", "Dep (1)"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrices_distilbert.png", dpi=300)
        plt.close()

    # 5. Paired Bootstrap Differences Forest Plot
    if len(paired_diffs) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(paired_diffs) * 0.6)))
        y_ticks = np.arange(len(paired_diffs))
        labels = [d["comparison"] for d in paired_diffs]
        point_diffs = [d["point_diff"] for d in paired_diffs]
        ci_lowers = [d["ci95_lower"] for d in paired_diffs]
        ci_uppers = [d["ci95_upper"] for d in paired_diffs]
        err_low = [p - l for p, l in zip(point_diffs, ci_lowers)]
        err_high = [u - p for p, u in zip(point_diffs, ci_uppers)]

        ax.errorbar(point_diffs, y_ticks, xerr=[err_low, err_high], fmt="o", color="#2b5c8f", ecolor="#d95f02", elinewidth=2, capsize=4)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Paired Difference: Metric(DistilBERT) - Metric(MiniLM) (95% CI)")
        ax.set_title("Paired Bootstrap Differences: DistilBERT vs MiniLM on Matched Development Split", fontweight="bold")
        plt.tight_layout()
        plt.savefig(plots_dir / "paired_bootstrap_differences.png", dpi=300)
        plt.close()

    logger.info(f"Diagnostic visualizations saved to: {plots_dir}")


def run_benchmark() -> None:
    set_seed(RANDOM_SEED)
    t0_total = time.time()
    logger.info("Starting Phase 18B DistilBERT controlled benchmarking suite...")

    (OUTPUT_DIR / "cv_results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    data = load_dataset_labels()
    logger.info(f"Loaded labels: Train N={len(data['train_labels'])}, Dev N={len(data['dev_labels'])}")

    # All 20 representations
    representations = [
        "whole__mean", "whole__median", "whole__trimmed", "whole__mean_std",
        "early__mean", "early__median", "early__trimmed", "early__mean_std",
        "middle__mean", "middle__median", "middle__trimmed", "middle__mean_std",
        "late__mean", "late__median", "late__trimmed", "late__mean_std",
        "concat__mean", "concat__median", "concat__trimmed", "concat__mean_std",
    ]

    all_results = []
    cls_result_map = {}
    reg_result_map = {}

    # 1. LinearSVC Classification across all 20 representations
    logger.info("\n=======================================================")
    logger.info("1. LINEARSVC CLASSIFICATION BENCHMARK (20 REPRESENTATIONS)")
    logger.info("=======================================================")

    for rep in representations:
        t_start = time.time()
        logger.info(f"Training LinearSVC on representation: {rep}...")
        res = train_eval_classification_linear_svc(rep, data)
        dev_m = res["dev_metrics"]
        train_m = res["train_oof_metrics"]
        logger.info(
            f"[{rep}] LinearSVC -> Opt C={res['opt_C']}, cw={res['opt_class_weight']}, tau={res['calibrated_tau']:.4f} | "
            f"Train OOF F1={train_m['f1']:.4f}, AUC={train_m['roc_auc']:.4f} | "
            f"Dev F1={dev_m['f1']:.4f} ({dev_m['f1_ci95'][0]:.4f}, {dev_m['f1_ci95'][1]:.4f}), "
            f"BalAcc={dev_m['balanced_accuracy']:.4f}, AUC={dev_m['roc_auc']:.4f}, CM={dev_m['confusion_matrix']} ({time.time()-t_start:.1f}s)"
        )
        all_results.append({
            "representation": rep,
            "task": "classification",
            "model": "LinearSVC",
            "n_train": res["n_train"],
            "n_dev": res["n_dev"],
            "dimension": res["dimension"],
            "opt_hyperparams": f"C={res['opt_C']}, class_weight={res['opt_class_weight']}, tau={res['calibrated_tau']:.4f}",
            "train_oof_f1": train_m["f1"],
            "train_oof_bal_acc": train_m["balanced_accuracy"],
            "train_oof_roc_auc": train_m["roc_auc"],
            "dev_f1": dev_m["f1"],
            "dev_f1_ci95_low": dev_m["f1_ci95"][0],
            "dev_f1_ci95_high": dev_m["f1_ci95"][1],
            "dev_balanced_accuracy": dev_m["balanced_accuracy"],
            "dev_accuracy": dev_m["accuracy"],
            "dev_roc_auc": dev_m["roc_auc"],
            "dev_roc_auc_ci95_low": dev_m["roc_auc_ci95"][0],
            "dev_roc_auc_ci95_high": dev_m["roc_auc_ci95"][1],
            "dev_pr_auc": dev_m["pr_auc"],
            "dev_precision": dev_m["precision"],
            "dev_recall": dev_m["recall"],
            "dev_confusion_matrix": dev_m["confusion_matrix"],
            "dev_ccc": np.nan,
            "dev_rmse": np.nan,
            "dev_mae": np.nan,
            "dev_r2": np.nan,
            "dev_pearson_r": np.nan,
        })
        cls_result_map[rep] = res

    # 2. Ridge Regression across all 20 representations
    logger.info("\n=======================================================")
    logger.info("2. RIDGE REGRESSION BENCHMARK (20 REPRESENTATIONS)")
    logger.info("=======================================================")

    for rep in representations:
        t_start = time.time()
        logger.info(f"Training Ridge regression on representation: {rep}...")
        res = train_eval_regression_ridge(rep, data)
        dev_m = res["dev_metrics"]
        train_m = res["train_oof_metrics"]
        logger.info(
            f"[{rep}] Ridge -> Opt alpha={res['opt_alpha']} | "
            f"Train OOF CCC={train_m['ccc']:.4f}, RMSE={train_m['rmse']:.4f} | "
            f"Dev CCC={dev_m['ccc']:.4f} ({dev_m['ccc_ci95'][0]:.4f}, {dev_m['ccc_ci95'][1]:.4f}), "
            f"RMSE={dev_m['rmse']:.4f}, R2={dev_m['r2']:.4f}, r={dev_m['pearson_r']:.4f} ({time.time()-t_start:.1f}s)"
        )
        all_results.append({
            "representation": rep,
            "task": "regression",
            "model": "Ridge",
            "n_train": res["n_train"],
            "n_dev": res["n_dev"],
            "dimension": res["dimension"],
            "opt_hyperparams": f"alpha={res['opt_alpha']}",
            "train_oof_f1": np.nan,
            "train_oof_bal_acc": np.nan,
            "train_oof_roc_auc": np.nan,
            "dev_f1": np.nan,
            "dev_f1_ci95_low": np.nan,
            "dev_f1_ci95_high": np.nan,
            "dev_balanced_accuracy": np.nan,
            "dev_accuracy": np.nan,
            "dev_roc_auc": np.nan,
            "dev_roc_auc_ci95_low": np.nan,
            "dev_roc_auc_ci95_high": np.nan,
            "dev_pr_auc": np.nan,
            "dev_precision": np.nan,
            "dev_recall": np.nan,
            "dev_confusion_matrix": "",
            "dev_ccc": dev_m["ccc"],
            "dev_ccc_ci95_low": dev_m["ccc_ci95"][0],
            "dev_ccc_ci95_high": dev_m["ccc_ci95"][1],
            "dev_rmse": dev_m["rmse"],
            "dev_rmse_ci95_low": dev_m["rmse_ci95"][0],
            "dev_rmse_ci95_high": dev_m["rmse_ci95"][1],
            "dev_mae": dev_m["mae"],
            "dev_r2": dev_m["r2"],
            "dev_pearson_r": dev_m["pearson_r"],
        })
        reg_result_map[f"{rep}__ridge"] = res

    # 3. RBF SVR on key representations
    logger.info("\n=======================================================")
    logger.info("3. RBF SVR REGRESSION BENCHMARK (KEY REPRESENTATIONS)")
    logger.info("=======================================================")

    svr_representations = [
        "whole__mean", "whole__mean_std",
        "early__mean", "middle__mean", "late__mean",
        "concat__mean", "concat__mean_std",
    ]

    for rep in svr_representations:
        t_start = time.time()
        logger.info(f"Training RBF SVR on representation: {rep}...")
        res = train_eval_regression_svr(rep, data)
        dev_m = res["dev_metrics"]
        train_m = res["train_oof_metrics"]
        logger.info(
            f"[{rep}] RBF SVR -> Opt C={res['opt_C']}, gamma={res['opt_gamma']}, eps={res['opt_epsilon']} | "
            f"Train OOF CCC={train_m['ccc']:.4f}, RMSE={train_m['rmse']:.4f} | "
            f"Dev CCC={dev_m['ccc']:.4f} ({dev_m['ccc_ci95'][0]:.4f}, {dev_m['ccc_ci95'][1]:.4f}), "
            f"RMSE={dev_m['rmse']:.4f}, R2={dev_m['r2']:.4f}, r={dev_m['pearson_r']:.4f} ({time.time()-t_start:.1f}s)"
        )
        all_results.append({
            "representation": rep,
            "task": "regression",
            "model": "RBF_SVR",
            "n_train": res["n_train"],
            "n_dev": res["n_dev"],
            "dimension": res["dimension"],
            "opt_hyperparams": f"C={res['opt_C']}, gamma={res['opt_gamma']}, eps={res['opt_epsilon']}",
            "train_oof_f1": np.nan,
            "train_oof_bal_acc": np.nan,
            "train_oof_roc_auc": np.nan,
            "dev_f1": np.nan,
            "dev_f1_ci95_low": np.nan,
            "dev_f1_ci95_high": np.nan,
            "dev_balanced_accuracy": np.nan,
            "dev_accuracy": np.nan,
            "dev_roc_auc": np.nan,
            "dev_roc_auc_ci95_low": np.nan,
            "dev_roc_auc_ci95_high": np.nan,
            "dev_pr_auc": np.nan,
            "dev_precision": np.nan,
            "dev_recall": np.nan,
            "dev_confusion_matrix": "",
            "dev_ccc": dev_m["ccc"],
            "dev_ccc_ci95_low": dev_m["ccc_ci95"][0],
            "dev_ccc_ci95_high": dev_m["ccc_ci95"][1],
            "dev_rmse": dev_m["rmse"],
            "dev_rmse_ci95_low": dev_m["rmse_ci95"][0],
            "dev_rmse_ci95_high": dev_m["rmse_ci95"][1],
            "dev_mae": dev_m["mae"],
            "dev_r2": dev_m["r2"],
            "dev_pearson_r": dev_m["pearson_r"],
        })
        reg_result_map[f"{rep}__rbf_svr"] = res

    # 4. Whole-Interview Baselines on Matched Cohort (N=157 Train / N=54 Dev)
    # To enable 100% fair comparison against Late and Concat models
    logger.info("\n=======================================================")
    logger.info("4. WHOLE-INTERVIEW MATCHED COHORT (N=54) BENCHMARKS")
    logger.info("=======================================================")

    matched_pids_dev = set(pd.read_csv(FEATURES_DIR / "dev" / "complete_cohort_participant_ids.csv")["Participant_ID"])
    matched_results = []

    for agg in ["mean", "median", "trimmed", "mean_std"]:
        rep = f"whole__{agg}"
        # LinearSVC matched
        res_cls = cls_result_map[rep]
        df_dev_pred = pd.read_csv(OUTPUT_DIR / "predictions" / f"dev_cls__{rep}__linear_svc.csv")
        df_matched_cls = df_dev_pred[df_dev_pred["Participant_ID"].isin(matched_pids_dev)].copy()
        
        y_true_m = df_matched_cls["y_true"].values
        y_pred_m = df_matched_cls["y_pred"].values
        y_score_m = df_matched_cls["y_score"].values
        m_cls = compute_classification_metrics(y_true_m, y_pred_m, y_score_m)
        ci_f1_m = bootstrap_ci(lambda yt, yp: float(f1_score(yt, yp, pos_label=1, zero_division=0)), y_true_m, y_pred_m)
        ci_auc_m = bootstrap_ci(lambda yt, ys: float(roc_auc_score(yt, ys)), y_true_m, y_score_m)

        matched_results.append({
            "representation": f"whole_matched__{agg}",
            "task": "classification",
            "model": "LinearSVC",
            "n_train": 157,
            "n_dev": 54,
            "dimension": res_cls["dimension"],
            "opt_hyperparams": res_cls["opt_C"],
            "dev_f1": m_cls["f1"],
            "dev_f1_ci95_low": ci_f1_m[0],
            "dev_f1_ci95_high": ci_f1_m[1],
            "dev_balanced_accuracy": m_cls["balanced_accuracy"],
            "dev_roc_auc": m_cls["roc_auc"],
            "dev_confusion_matrix": m_cls["confusion_matrix"],
            "dev_ccc": np.nan,
            "dev_rmse": np.nan,
        })

        # Ridge matched
        df_reg_pred = pd.read_csv(OUTPUT_DIR / "predictions" / f"dev_reg__{rep}__ridge.csv")
        df_matched_reg = df_reg_pred[df_reg_pred["Participant_ID"].isin(matched_pids_dev)].copy()
        yt_r_m = df_matched_reg["y_true"].values
        yp_r_m = df_matched_reg["y_pred"].values
        m_reg = compute_regression_metrics(yt_r_m, yp_r_m)
        ci_ccc_m = bootstrap_ci(concordance_correlation_coefficient, yt_r_m, yp_r_m)
        ci_rmse_m = bootstrap_ci(lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))), yt_r_m, yp_r_m)

        matched_results.append({
            "representation": f"whole_matched__{agg}",
            "task": "regression",
            "model": "Ridge",
            "n_train": 157,
            "n_dev": 54,
            "dimension": res_cls["dimension"],
            "opt_hyperparams": reg_result_map[f"{rep}__ridge"]["opt_alpha"],
            "dev_f1": np.nan,
            "dev_f1_ci95_low": np.nan,
            "dev_f1_ci95_high": np.nan,
            "dev_balanced_accuracy": np.nan,
            "dev_roc_auc": np.nan,
            "dev_confusion_matrix": "",
            "dev_ccc": m_reg["ccc"],
            "dev_ccc_ci95_low": ci_ccc_m[0],
            "dev_ccc_ci95_high": ci_ccc_m[1],
            "dev_rmse": m_reg["rmse"],
            "dev_rmse_ci95_low": ci_rmse_m[0],
            "dev_rmse_ci95_high": ci_rmse_m[1],
            "dev_mae": m_reg["mae"],
            "dev_r2": m_reg["r2"],
            "dev_pearson_r": m_reg["pearson_r"],
        })

        logger.info(f"[whole_matched__{agg}] Matched N=54 -> LinearSVC F1={m_cls['f1']:.4f}, AUC={m_cls['roc_auc']:.4f} | Ridge CCC={m_reg['ccc']:.4f}, RMSE={m_reg['rmse']:.4f}")

    all_results.extend(matched_results)
    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_csv(OUTPUT_DIR / "reports" / "model_comparison_table.csv", index=False)
    logger.info(f"Saved complete benchmark comparison table to: {OUTPUT_DIR / 'reports' / 'model_comparison_table.csv'}")

    # 5. Paired Bootstrap Difference Tests against Phase 18A MiniLM Champions
    logger.info("\n=======================================================")
    logger.info("5. PAIRED BOOTSTRAP DIFFERENCE TESTS (DISTILBERT VS MINILM)")
    logger.info("=======================================================")

    paired_comparisons = []

    # Helper function to load Phase 18A prediction
    def load_18a_pred(filename: str) -> pd.DataFrame:
        path = PHASE_18A_PRED_DIR / filename
        if path.exists():
            return pd.read_csv(path)
        return None

    # Classification paired tests
    # 1. whole__mean LinearSVC (DistilBERT vs MiniLM)
    p18a_whole_mean_cls = load_18a_pred("dev_cls__whole__mean__linear_svc.csv")
    if p18a_whole_mean_cls is not None:
        d_pred = pd.read_csv(OUTPUT_DIR / "predictions" / "dev_cls__whole__mean__linear_svc.csv")
        merged = pd.merge(d_pred, p18a_whole_mean_cls, on="Participant_ID", suffixes=("_d", "_m"))
        diff_f1 = paired_bootstrap_difference(
            lambda yt, yp: float(f1_score(yt, yp, pos_label=1, zero_division=0)),
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        diff_auc = paired_bootstrap_difference(
            lambda yt, ys: float(roc_auc_score(yt, ys)),
            merged["y_true_d"].values,
            merged["y_score_d"].values,
            merged["y_score_m"].values,
        )
        paired_comparisons.append({
            "comparison": "whole__mean LinearSVC (F1)",
            "task": "classification",
            "metric": "f1",
            "n": len(merged),
            **diff_f1,
        })
        paired_comparisons.append({
            "comparison": "whole__mean LinearSVC (ROC-AUC)",
            "task": "classification",
            "metric": "roc_auc",
            "n": len(merged),
            **diff_auc,
        })

    # 2. middle__mean LinearSVC (DistilBERT vs MiniLM)
    p18a_mid_mean_cls = load_18a_pred("dev_cls__middle__mean__linear_svc.csv")
    if p18a_mid_mean_cls is not None:
        d_pred = pd.read_csv(OUTPUT_DIR / "predictions" / "dev_cls__middle__mean__linear_svc.csv")
        merged = pd.merge(d_pred, p18a_mid_mean_cls, on="Participant_ID", suffixes=("_d", "_m"))
        diff_f1 = paired_bootstrap_difference(
            lambda yt, yp: float(f1_score(yt, yp, pos_label=1, zero_division=0)),
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        diff_auc = paired_bootstrap_difference(
            lambda yt, ys: float(roc_auc_score(yt, ys)),
            merged["y_true_d"].values,
            merged["y_score_d"].values,
            merged["y_score_m"].values,
        )
        paired_comparisons.append({
            "comparison": "middle__mean LinearSVC (F1)",
            "task": "classification",
            "metric": "f1",
            "n": len(merged),
            **diff_f1,
        })
        paired_comparisons.append({
            "comparison": "middle__mean LinearSVC (ROC-AUC)",
            "task": "classification",
            "metric": "roc_auc",
            "n": len(merged),
            **diff_auc,
        })

    # 3. concat__median LinearSVC (DistilBERT vs MiniLM, N=54)
    p18a_concat_med_cls = load_18a_pred("dev_cls__concat__median__linear_svc.csv")
    if p18a_concat_med_cls is not None:
        d_pred = pd.read_csv(OUTPUT_DIR / "predictions" / "dev_cls__concat__median__linear_svc.csv")
        merged = pd.merge(d_pred, p18a_concat_med_cls, on="Participant_ID", suffixes=("_d", "_m"))
        diff_f1 = paired_bootstrap_difference(
            lambda yt, yp: float(f1_score(yt, yp, pos_label=1, zero_division=0)),
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        diff_auc = paired_bootstrap_difference(
            lambda yt, ys: float(roc_auc_score(yt, ys)),
            merged["y_true_d"].values,
            merged["y_score_d"].values,
            merged["y_score_m"].values,
        )
        paired_comparisons.append({
            "comparison": "concat__median LinearSVC (F1)",
            "task": "classification",
            "metric": "f1",
            "n": len(merged),
            **diff_f1,
        })
        paired_comparisons.append({
            "comparison": "concat__median LinearSVC (ROC-AUC)",
            "task": "classification",
            "metric": "roc_auc",
            "n": len(merged),
            **diff_auc,
        })

    # Regression paired tests
    # 4. whole__median Ridge (DistilBERT vs MiniLM, N=56)
    p18a_whole_med_reg = load_18a_pred("dev_reg__whole__median__ridge.csv")
    if p18a_whole_med_reg is not None:
        d_pred = pd.read_csv(OUTPUT_DIR / "predictions" / "dev_reg__whole__median__ridge.csv")
        merged = pd.merge(d_pred, p18a_whole_med_reg, on="Participant_ID", suffixes=("_d", "_m"))
        diff_ccc = paired_bootstrap_difference(
            concordance_correlation_coefficient,
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        diff_rmse = paired_bootstrap_difference(
            lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        paired_comparisons.append({
            "comparison": "whole__median Ridge (CCC)",
            "task": "regression",
            "metric": "ccc",
            "n": len(merged),
            **diff_ccc,
        })
        paired_comparisons.append({
            "comparison": "whole__median Ridge (RMSE)",
            "task": "regression",
            "metric": "rmse",
            "n": len(merged),
            **diff_rmse,
        })

    # 5. middle__median Ridge (DistilBERT vs MiniLM, N=56)
    p18a_mid_med_reg = load_18a_pred("dev_reg__middle__median__ridge.csv")
    if p18a_mid_med_reg is not None:
        d_pred = pd.read_csv(OUTPUT_DIR / "predictions" / "dev_reg__middle__median__ridge.csv")
        merged = pd.merge(d_pred, p18a_mid_med_reg, on="Participant_ID", suffixes=("_d", "_m"))
        diff_ccc = paired_bootstrap_difference(
            concordance_correlation_coefficient,
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        diff_rmse = paired_bootstrap_difference(
            lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        paired_comparisons.append({
            "comparison": "middle__median Ridge (CCC)",
            "task": "regression",
            "metric": "ccc",
            "n": len(merged),
            **diff_ccc,
        })
        paired_comparisons.append({
            "comparison": "middle__median Ridge (RMSE)",
            "task": "regression",
            "metric": "rmse",
            "n": len(merged),
            **diff_rmse,
        })

    # 6. concat__mean RBF SVR (DistilBERT vs MiniLM, N=54)
    p18a_concat_mean_svr = load_18a_pred("dev_reg__concat__mean__rbf_svr.csv")
    if p18a_concat_mean_svr is not None and "concat__mean__rbf_svr" in reg_result_map:
        d_pred = pd.read_csv(OUTPUT_DIR / "predictions" / "dev_reg__concat__mean__rbf_svr.csv")
        merged = pd.merge(d_pred, p18a_concat_mean_svr, on="Participant_ID", suffixes=("_d", "_m"))
        diff_ccc = paired_bootstrap_difference(
            concordance_correlation_coefficient,
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        diff_rmse = paired_bootstrap_difference(
            lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
            merged["y_true_d"].values,
            merged["y_pred_d"].values,
            merged["y_pred_m"].values,
        )
        paired_comparisons.append({
            "comparison": "concat__mean RBF SVR (CCC)",
            "task": "regression",
            "metric": "ccc",
            "n": len(merged),
            **diff_ccc,
        })
        paired_comparisons.append({
            "comparison": "concat__mean RBF SVR (RMSE)",
            "task": "regression",
            "metric": "rmse",
            "n": len(merged),
            **diff_rmse,
        })

    df_paired = pd.DataFrame(paired_comparisons)
    df_paired.to_csv(OUTPUT_DIR / "reports" / "paired_bootstrap_differences.csv", index=False)
    logger.info(f"Saved paired bootstrap differences to: {OUTPUT_DIR / 'reports' / 'paired_bootstrap_differences.csv'}")

    for _, row in df_paired.iterrows():
        logger.info(
            f"Paired Diff [{row['comparison']}]: Delta={row['point_diff']:+.4f} "
            f"(95% CI: [{row['ci95_lower']:+.4f}, {row['ci95_upper']:+.4f}]), "
            f"DistilBERT={row['point_distilbert']:.4f}, MiniLM={row['point_minilm']:.4f}, "
            f"P(D>M)={row['p_distilbert_wins']*100:.1f}%"
        )

    # 6. Generate Diagnostic Visualizations
    logger.info("\n=======================================================")
    logger.info("6. GENERATING DIAGNOSTIC VISUALIZATIONS")
    logger.info("=======================================================")
    generate_diagnostic_plots(df_all_results, paired_comparisons, cls_result_map)

    elapsed_total = time.time() - t0_total
    logger.info(f"\nPhase 18B benchmarking suite completed successfully in {elapsed_total:.2f} seconds.")


if __name__ == "__main__":
    run_benchmark()
