"""
Shared Classification Metrics Utilities for E-DAIC Benchmark.

Calculates:
- Positive-class F1-score (PHQ_Binary = 1)
- Precision & Recall
- Accuracy & Balanced Accuracy
- ROC-AUC from continuous scores (with edge-case handling)
- Average Precision (PR-AUC) from continuous scores
- Confusion matrix counts (TN, FP, FN, TP)
- FPR, TPR, Precision-Recall curve coordinate points for serialization
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from config import POSITIVE_CLASS_VALUE


@dataclass
class ClassificationMetrics:
    """Encapsulates all evaluation metrics for a binary classification experiment."""
    accuracy: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: Optional[float]
    average_precision: Optional[float]
    tn: int
    fp: int
    fn: int
    tp: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten warnings to semicolon-separated string if needed
        d["warnings_str"] = "; ".join(self.warnings) if self.warnings else "none"
        return d


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    pos_label: int = POSITIVE_CLASS_VALUE,
) -> ClassificationMetrics:
    """
    Computes standard and curve metrics from ground truth labels,
    predicted hard classes, and continuous prediction scores.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    warnings: List[str] = []

    # 1. Discrete label metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

    # Confusion matrix
    # Labels [0, 1] mapped to TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # 2. Continuous score metrics (ROC-AUC & Average Precision)
    auc_val: Optional[float] = None
    ap_val: Optional[float] = None

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=np.float64)
        unique_classes = np.unique(y_true)
        
        if len(unique_classes) < 2:
            warnings.append(f"Only one class present in y_true ({unique_classes}). ROC-AUC and AP are undefined.")
        elif np.all(np.isnan(y_score)):
            warnings.append("Continuous scores are all NaN.")
        else:
            try:
                auc_val = float(roc_auc_score(y_true, y_score))
            except Exception as e:
                warnings.append(f"ROC-AUC computation failed: {e}")

            try:
                ap_val = float(average_precision_score(y_true, y_score, pos_label=pos_label))
            except Exception as e:
                warnings.append(f"Average Precision computation failed: {e}")
    else:
        warnings.append("No continuous scores provided; ROC-AUC and AP not computed.")

    return ClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        f1=f1,
        precision=prec,
        recall=rec,
        roc_auc=auc_val,
        average_precision=ap_val,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        warnings=warnings,
    )


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """Computes and serializes ROC curve coordinates and AUC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {"error": "Single class in y_true", "fpr": [], "tpr": [], "thresholds": [], "auc": None}

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=POSITIVE_CLASS_VALUE)
    auc_val = float(roc_auc_score(y_true, y_score))

    return {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": [float(x) for x in thresholds],
        "auc": auc_val,
    }


def compute_pr_curve_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """Computes and serializes Precision-Recall curve coordinates and Average Precision."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {"error": "Single class in y_true", "precision": [], "recall": [], "thresholds": [], "ap": None}

    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=POSITIVE_CLASS_VALUE)
    ap_val = float(average_precision_score(y_true, y_score, pos_label=POSITIVE_CLASS_VALUE))

    return {
        "precision": [float(x) for x in precision],
        "recall": [float(x) for x in recall],
        "thresholds": [float(x) for x in thresholds],
        "ap": ap_val,
    }


# ==============================================================================
# Regression Metrics (Continuous PHQ_Score)
# ==============================================================================

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Lin's Concordance Correlation Coefficient (CCC):
    CCC = (2 * Cov(y_true, y_pred)) / (Var(y_true) + Var(y_pred) + (mean(y_true) - mean(y_pred))^2)

    Uses population variance/covariance (ddof=0) consistently.
    Returns 0.0 if denominator is 0.0 or if either variance is zero and predictions differ.
    Returns 1.0 if both vectors are identical constants.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
    if len(y_true) == 0:
        return 0.0

    # Check identical vectors
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


@dataclass
class RegressionMetrics:
    """Encapsulates all evaluation metrics for a continuous regression experiment."""
    ccc: float
    rmse: float
    mae: float
    r2: float
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["warnings_str"] = "; ".join(self.warnings) if self.warnings else "none"
        return d


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """
    Computes CCC, RMSE, MAE, and R^2 from ground truth and predictions.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    warnings: List[str] = []

    if len(y_true) != len(y_pred):
        raise ValueError(f"Dimension mismatch: {len(y_true)} true vs {len(y_pred)} pred")

    # CCC
    try:
        ccc_val = concordance_correlation_coefficient(y_true, y_pred)
    except Exception as e:
        warnings.append(f"CCC calculation failed: {e}")
        ccc_val = 0.0

    # RMSE
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAE
    mae_val = float(np.mean(np.abs(y_true - y_pred)))

    # R^2
    var_true = float(np.var(y_true, ddof=0))
    if var_true <= 1e-12:
        warnings.append("Zero variance in y_true; R^2 undefined.")
        r2_val = 0.0
    else:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2_val = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return RegressionMetrics(
        ccc=ccc_val,
        rmse=rmse_val,
        mae=mae_val,
        r2=r2_val,
        warnings=warnings,
    )
