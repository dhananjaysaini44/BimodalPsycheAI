"""
Text Baselines Evaluator for Phase 17A.

Executes:
1. 5-Fold Cross-Validation on official Training set (N=163) under strict data leakage guards.
2. Full Training -> Development Evaluation on official Development set (N=56).
3. Metric calculations, prediction logging, and structured JSON/CSV export.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# Ensure code directory is on sys.path
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from utils.metrics import (
        compute_classification_metrics,
        compute_regression_metrics,
        concordance_correlation_coefficient,
    )
except ImportError:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
        cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
        mean_diff_sq = (mean_true - mean_pred) ** 2
        denominator = var_true + var_pred + mean_diff_sq
        if denominator <= 1e-12:
            return 1.0 if np.allclose(y_true, y_pred) else 0.0
        ccc = (2.0 * cov) / denominator
        return float(np.clip(ccc, -1.0, 1.0))

    def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        ccc_val = concordance_correlation_coefficient(y_true, y_pred)
        rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae_val = float(np.mean(np.abs(y_true - y_pred)))
        var_true = float(np.var(y_true, ddof=0))
        if var_true <= 1e-12:
            r2_val = 0.0
        else:
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2_val = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        class RegMetrics:
            def __init__(self, ccc, rmse, mae, r2):
                self.ccc = ccc
                self.rmse = rmse
                self.mae = mae
                self.r2 = r2
            def to_dict(self):
                return {"ccc": self.ccc, "rmse": self.rmse, "mae": self.mae, "r2": self.r2}
        return RegMetrics(ccc_val, rmse_val, mae_val, r2_val)

    def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Any:
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        auc_val = float(roc_auc_score(y_true, y_score)) if y_score is not None else None
        ap_val = float(average_precision_score(y_true, y_score, pos_label=1)) if y_score is not None else None

        class ClfMetrics:
            def __init__(self, accuracy, balanced_accuracy, f1, precision, recall, roc_auc, average_precision, tn, fp, fn, tp):
                self.accuracy = accuracy
                self.balanced_accuracy = balanced_accuracy
                self.f1 = f1
                self.precision = precision
                self.recall = recall
                self.roc_auc = roc_auc
                self.average_precision = average_precision
                self.tn = tn
                self.fp = fp
                self.fn = fn
                self.tp = tp
            def to_dict(self):
                return {
                    "accuracy": self.accuracy,
                    "balanced_accuracy": self.balanced_accuracy,
                    "f1": self.f1,
                    "precision": self.precision,
                    "recall": self.recall,
                    "roc_auc": self.roc_auc,
                    "average_precision": self.average_precision,
                    "tn": self.tn,
                    "fp": self.fp,
                    "fn": self.fn,
                    "tp": self.tp,
                }
        return ClfMetrics(acc, bal_acc, f1, prec, rec, auc_val, ap_val, tn, fp, fn, tp)

from text_baselines_v1.text_pipeline import (
    FEATURE_EXTRACTOR_CONFIGS,
    MODEL_CONFIGS,
    RANDOM_SEED,
    build_classification_pipeline,
    build_regression_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRANSCRIPTS_PATH = DEFAULT_REPO_ROOT / "ml-training" / "dataset" / "text_v1" / "participant_transcripts.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "ml-training" / "outputs" / "text_baselines_v1"


def evaluate_classification_cv(
    train_df: pd.DataFrame,
    feature_name: str,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs 5-fold Stratified CV strictly on the training set.
    Learned TF-IDF vocabulary and IDF weights are computed solely inside training folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    pids = train_df["Participant_ID"].values
    X_text = train_df["normalized_text"].values
    y = train_df["PHQ_Binary"].values

    oof_preds = np.zeros(len(train_df), dtype=int)
    oof_scores = np.zeros(len(train_df), dtype=float)
    fold_assignments = np.zeros(len(train_df), dtype=int)

    fold_metrics_list = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_text, y)):
        X_tr, y_tr = X_text[train_idx], y[train_idx]
        X_va, y_va = X_text[val_idx], y[val_idx]

        pipeline = build_classification_pipeline(feature_name)
        pipeline.fit(X_tr, y_tr)

        y_pred_fold = pipeline.predict(X_va)
        y_score_fold = pipeline.decision_function(X_va)

        oof_preds[val_idx] = y_pred_fold
        oof_scores[val_idx] = y_score_fold
        fold_assignments[val_idx] = fold_idx + 1

        fold_m = compute_classification_metrics(y_va, y_pred_fold, y_score_fold).to_dict()
        fold_m["fold"] = fold_idx + 1
        fold_metrics_list.append(fold_m)

    total_time = time.time() - t0

    # Overall OOF Metrics
    overall_m = compute_classification_metrics(y, oof_preds, oof_scores).to_dict()

    oof_df = pd.DataFrame({
        "Participant_ID": pids,
        "split": "train",
        "fold": fold_assignments,
        "y_true": y,
        "y_pred": oof_preds,
        "y_score": oof_scores,
    }).sort_values("Participant_ID").reset_index(drop=True)

    cv_summary = {
        "model_id": f"text_classical__class__{feature_name}__linear_svc__cv5",
        "feature_name": feature_name,
        "model_type": "LinearSVC",
        "eval_split": "train_oof",
        "num_participants": len(train_df),
        "n_splits": n_splits,
        "overall_metrics": overall_m,
        "fold_metrics": fold_metrics_list,
        "mean_fold_f1": float(np.mean([m["f1"] for m in fold_metrics_list])),
        "std_fold_f1": float(np.std([m["f1"] for m in fold_metrics_list])),
        "mean_fold_balanced_acc": float(np.mean([m["balanced_accuracy"] for m in fold_metrics_list])),
        "mean_fold_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics_list])),
        "runtime_sec": round(total_time, 3),
    }

    return oof_df, cv_summary


def evaluate_classification_dev(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    feature_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits classification pipeline on the full training set (N=163)
    and evaluates on the held-out development set (N=56).
    """
    X_train = train_df["normalized_text"].values
    y_train = train_df["PHQ_Binary"].values

    X_dev = dev_df["normalized_text"].values
    y_dev = dev_df["PHQ_Binary"].values
    pids_dev = dev_df["Participant_ID"].values

    t0 = time.time()
    pipeline = build_classification_pipeline(feature_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_dev)
    y_score = pipeline.decision_function(X_dev)
    runtime = time.time() - t0

    metrics = compute_classification_metrics(y_dev, y_pred, y_score).to_dict()

    dev_pred_df = pd.DataFrame({
        "Participant_ID": pids_dev,
        "split": "dev",
        "y_true": y_dev,
        "y_pred": y_pred,
        "y_score": y_score,
    }).sort_values("Participant_ID").reset_index(drop=True)

    summary = {
        "model_id": f"text_classical__class__{feature_name}__linear_svc__dev",
        "feature_name": feature_name,
        "model_type": "LinearSVC",
        "eval_split": "dev",
        "train_participants": len(train_df),
        "dev_participants": len(dev_df),
        "metrics": metrics,
        "runtime_sec": round(runtime, 3),
    }

    return dev_pred_df, summary


def evaluate_regression_cv(
    train_df: pd.DataFrame,
    feature_name: str,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs 5-fold CV strictly on the training set for continuous PHQ_Score regression.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    pids = train_df["Participant_ID"].values
    X_text = train_df["normalized_text"].values
    y = train_df["PHQ_Score"].values.astype(float)

    oof_preds = np.zeros(len(train_df), dtype=float)
    fold_assignments = np.zeros(len(train_df), dtype=int)

    fold_metrics_list = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_text, y)):
        X_tr, y_tr = X_text[train_idx], y[train_idx]
        X_va, y_va = X_text[val_idx], y[val_idx]

        pipeline = build_regression_pipeline(feature_name)
        pipeline.fit(X_tr, y_tr)

        y_pred_fold = pipeline.predict(X_va)

        oof_preds[val_idx] = y_pred_fold
        fold_assignments[val_idx] = fold_idx + 1

        fold_m = compute_regression_metrics(y_va, y_pred_fold).to_dict()
        fold_m["fold"] = fold_idx + 1
        fold_metrics_list.append(fold_m)

    total_time = time.time() - t0

    # Overall OOF Metrics
    overall_m = compute_regression_metrics(y, oof_preds).to_dict()

    oof_df = pd.DataFrame({
        "Participant_ID": pids,
        "split": "train",
        "fold": fold_assignments,
        "y_true": y,
        "y_pred": oof_preds,
        "residual": y - oof_preds,
    }).sort_values("Participant_ID").reset_index(drop=True)

    cv_summary = {
        "model_id": f"text_classical__reg__{feature_name}__ridge__cv5",
        "feature_name": feature_name,
        "model_type": "Ridge",
        "eval_split": "train_oof",
        "num_participants": len(train_df),
        "n_splits": n_splits,
        "overall_metrics": overall_m,
        "fold_metrics": fold_metrics_list,
        "mean_fold_ccc": float(np.mean([m["ccc"] for m in fold_metrics_list])),
        "std_fold_ccc": float(np.std([m["ccc"] for m in fold_metrics_list])),
        "mean_fold_rmse": float(np.mean([m["rmse"] for m in fold_metrics_list])),
        "mean_fold_mae": float(np.mean([m["mae"] for m in fold_metrics_list])),
        "runtime_sec": round(total_time, 3),
    }

    return oof_df, cv_summary


def evaluate_regression_dev(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    feature_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits regression pipeline on the full training set (N=163)
    and evaluates on the held-out development set (N=56).
    """
    X_train = train_df["normalized_text"].values
    y_train = train_df["PHQ_Score"].values.astype(float)

    X_dev = dev_df["normalized_text"].values
    y_dev = dev_df["PHQ_Score"].values.astype(float)
    pids_dev = dev_df["Participant_ID"].values

    t0 = time.time()
    pipeline = build_regression_pipeline(feature_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_dev)
    runtime = time.time() - t0

    metrics = compute_regression_metrics(y_dev, y_pred).to_dict()

    dev_pred_df = pd.DataFrame({
        "Participant_ID": pids_dev,
        "split": "dev",
        "y_true": y_dev,
        "y_pred": y_pred,
        "residual": y_dev - y_pred,
    }).sort_values("Participant_ID").reset_index(drop=True)

    summary = {
        "model_id": f"text_classical__reg__{feature_name}__ridge__dev",
        "feature_name": feature_name,
        "model_type": "Ridge",
        "eval_split": "dev",
        "train_participants": len(train_df),
        "dev_participants": len(dev_df),
        "metrics": metrics,
        "runtime_sec": round(runtime, 3),
    }

    return dev_pred_df, summary


def run_all_text_baselines(
    transcripts_path: Path = DEFAULT_TRANSCRIPTS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Orchestrates the complete Phase 17A evaluation across:
    - 3 feature representations: word_tfidf, char_tfidf, combined_tfidf
    - 2 tasks: classification (LinearSVC) and regression (Ridge)
    - 2 evaluation scopes: 5-fold CV (train_oof, N=163) and Development (dev, N=56)
    - Quarantines Test set (N=56) from any evaluation or tuning.
    """
    df = pd.read_csv(transcripts_path)
    train_df = df[df["split"] == "train"].copy().sort_values("Participant_ID").reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].copy().sort_values("Participant_ID").reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().sort_values("Participant_ID").reset_index(drop=True)

    logger.info(f"Loaded splits: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}")
    assert len(train_df) == 163, f"Expected 163 train rows, got {len(train_df)}"
    assert len(dev_df) == 56, f"Expected 56 dev rows, got {len(dev_df)}"
    assert len(test_df) == 56, f"Expected 56 test rows, got {len(test_df)}"
    logger.info("Test set (N=56) is quarantined and will not be evaluated.")

    feature_names = ["word_tfidf", "char_tfidf", "combined_tfidf"]

    # Target subdirectories
    cv_dir = output_dir / "cross_validation"
    dev_dir = output_dir / "development"
    cls_metrics_dir = output_dir / "classification" / "metrics"
    cls_preds_dir = output_dir / "classification" / "predictions"
    reg_metrics_dir = output_dir / "regression" / "metrics"
    reg_preds_dir = output_dir / "regression" / "predictions"
    configs_dir = output_dir / "configs"

    for d in [cv_dir, dev_dir, cls_metrics_dir, cls_preds_dir, reg_metrics_dir, reg_preds_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results_master: Dict[str, Any] = {
        "classification": {"cv": {}, "dev": {}},
        "regression": {"cv": {}, "dev": {}},
    }

    cls_cv_summary_rows = []
    cls_dev_summary_rows = []
    reg_cv_summary_rows = []
    reg_dev_summary_rows = []

    # Save configurations
    config_record = {
        "feature_extractors": FEATURE_EXTRACTOR_CONFIGS,
        "models": MODEL_CONFIGS,
        "splits": {"train_n": 163, "dev_n": 56, "test_n": 56, "test_quarantined": True},
        "cv_folds": 5,
        "random_seed": RANDOM_SEED,
    }
    with open(configs_dir / "experiment_configurations.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    for feat in feature_names:
        logger.info(f"================ Evaluating Feature: {feat} ================")

        # ---------------- 1. Classification CV ----------------
        cls_oof_df, cls_cv_res = evaluate_classification_cv(train_df, feat)
        results_master["classification"]["cv"][feat] = cls_cv_res

        cls_oof_df.to_csv(cv_dir / f"oof_classification__{feat}__linear_svc.csv", index=False)
        cls_oof_df.to_csv(cls_preds_dir / f"{feat}__linear_svc__cv5__predictions.csv", index=False)
        with open(cls_metrics_dir / f"{feat}__linear_svc__cv5__metrics.json", "w", encoding="utf-8") as f:
            json.dump(cls_cv_res, f, indent=2)

        m_oof = cls_cv_res["overall_metrics"]
        cls_cv_summary_rows.append({
            "feature": feat,
            "model": "LinearSVC",
            "eval_split": "train_oof",
            "f1": round(m_oof["f1"], 4),
            "balanced_accuracy": round(m_oof["balanced_accuracy"], 4),
            "accuracy": round(m_oof["accuracy"], 4),
            "precision": round(m_oof["precision"], 4),
            "recall": round(m_oof["recall"], 4),
            "roc_auc": round(m_oof["roc_auc"], 4) if m_oof["roc_auc"] else None,
            "average_precision": round(m_oof["average_precision"], 4) if m_oof["average_precision"] else None,
            "mean_fold_f1": round(cls_cv_res["mean_fold_f1"], 4),
            "std_fold_f1": round(cls_cv_res["std_fold_f1"], 4),
            "runtime_sec": cls_cv_res["runtime_sec"],
        })

        # ---------------- 2. Classification Dev ----------------
        cls_dev_pred_df, cls_dev_res = evaluate_classification_dev(train_df, dev_df, feat)
        results_master["classification"]["dev"][feat] = cls_dev_res

        cls_dev_pred_df.to_csv(dev_dir / f"dev_predictions_classification__{feat}__linear_svc.csv", index=False)
        cls_dev_pred_df.to_csv(cls_preds_dir / f"{feat}__linear_svc__dev__predictions.csv", index=False)
        with open(cls_metrics_dir / f"{feat}__linear_svc__dev__metrics.json", "w", encoding="utf-8") as f:
            json.dump(cls_dev_res, f, indent=2)

        m_dev = cls_dev_res["metrics"]
        cls_dev_summary_rows.append({
            "feature": feat,
            "model": "LinearSVC",
            "eval_split": "dev",
            "f1": round(m_dev["f1"], 4),
            "balanced_accuracy": round(m_dev["balanced_accuracy"], 4),
            "accuracy": round(m_dev["accuracy"], 4),
            "precision": round(m_dev["precision"], 4),
            "recall": round(m_dev["recall"], 4),
            "roc_auc": round(m_dev["roc_auc"], 4) if m_dev["roc_auc"] else None,
            "average_precision": round(m_dev["average_precision"], 4) if m_dev["average_precision"] else None,
            "tn": m_dev["tn"],
            "fp": m_dev["fp"],
            "fn": m_dev["fn"],
            "tp": m_dev["tp"],
            "runtime_sec": cls_dev_res["runtime_sec"],
        })

        # ---------------- 3. Regression CV ----------------
        reg_oof_df, reg_cv_res = evaluate_regression_cv(train_df, feat)
        results_master["regression"]["cv"][feat] = reg_cv_res

        reg_oof_df.to_csv(cv_dir / f"oof_regression__{feat}__ridge.csv", index=False)
        reg_oof_df.to_csv(reg_preds_dir / f"{feat}__ridge__cv5__predictions.csv", index=False)
        with open(reg_metrics_dir / f"{feat}__ridge__cv5__metrics.json", "w", encoding="utf-8") as f:
            json.dump(reg_cv_res, f, indent=2)

        r_oof = reg_cv_res["overall_metrics"]
        reg_cv_summary_rows.append({
            "feature": feat,
            "model": "Ridge",
            "eval_split": "train_oof",
            "ccc": round(r_oof["ccc"], 4),
            "rmse": round(r_oof["rmse"], 4),
            "mae": round(r_oof["mae"], 4),
            "r2": round(r_oof["r2"], 4),
            "mean_fold_ccc": round(reg_cv_res["mean_fold_ccc"], 4),
            "std_fold_ccc": round(reg_cv_res["std_fold_ccc"], 4),
            "runtime_sec": reg_cv_res["runtime_sec"],
        })

        # ---------------- 4. Regression Dev ----------------
        reg_dev_pred_df, reg_dev_res = evaluate_regression_dev(train_df, dev_df, feat)
        results_master["regression"]["dev"][feat] = reg_dev_res

        reg_dev_pred_df.to_csv(dev_dir / f"dev_predictions_regression__{feat}__ridge.csv", index=False)
        reg_dev_pred_df.to_csv(reg_preds_dir / f"{feat}__ridge__dev__predictions.csv", index=False)
        with open(reg_metrics_dir / f"{feat}__ridge__dev__metrics.json", "w", encoding="utf-8") as f:
            json.dump(reg_dev_res, f, indent=2)

        r_dev = reg_dev_res["metrics"]
        reg_dev_summary_rows.append({
            "feature": feat,
            "model": "Ridge",
            "eval_split": "dev",
            "ccc": round(r_dev["ccc"], 4),
            "rmse": round(r_dev["rmse"], 4),
            "mae": round(r_dev["mae"], 4),
            "r2": round(r_dev["r2"], 4),
            "runtime_sec": reg_dev_res["runtime_sec"],
        })

    # Save summary tables
    df_cls_cv = pd.DataFrame(cls_cv_summary_rows)
    df_cls_dev = pd.DataFrame(cls_dev_summary_rows)
    df_reg_cv = pd.DataFrame(reg_cv_summary_rows)
    df_reg_dev = pd.DataFrame(reg_dev_summary_rows)

    df_cls_cv.to_csv(cv_dir / "cv_classification_summary.csv", index=False)
    df_cls_dev.to_csv(dev_dir / "dev_classification_summary.csv", index=False)
    df_reg_cv.to_csv(cv_dir / "cv_regression_summary.csv", index=False)
    df_reg_dev.to_csv(dev_dir / "dev_regression_summary.csv", index=False)

    with open(cv_dir / "cv_classification_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_master["classification"]["cv"], f, indent=2)
    with open(dev_dir / "dev_classification_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_master["classification"]["dev"], f, indent=2)
    with open(cv_dir / "cv_regression_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_master["regression"]["cv"], f, indent=2)
    with open(dev_dir / "dev_regression_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_master["regression"]["dev"], f, indent=2)

    logger.info("All evaluations completed and exported successfully.")
    return results_master


if __name__ == "__main__":
    run_all_text_baselines()
