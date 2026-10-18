"""
Regression Evaluation Modules for E-DAIC Benchmark.

Provides:
- 5-fold Cross-Validation on official training participants (n=163) with OOF predictions.
- Development evaluation (fitted on full Train n=163, evaluated on Dev n=56).
- Strict data-leakage protection: StandardScaler fitted inside folds.
- No model persistence.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import RANDOM_SEED, TARGET_REGRESSION
from models.regression_models import get_regressor, REGRESSION_REGISTRY
from utils.metrics import compute_regression_metrics, RegressionMetrics


@dataclass
class FoldRegressionResult:
    """Evaluation result for a single cross-validation fold."""
    fold: int
    train_samples: int
    val_samples: int
    metrics: RegressionMetrics
    y_true_val: np.ndarray
    y_pred_val: np.ndarray
    val_participant_ids: np.ndarray
    runtime_sec: float
    status: str
    error: str = ""


@dataclass
class CVRegressionResult:
    """Aggregate result across all cross-validation folds."""
    experiment_id: str
    representation: str
    pooling_strategy: str
    model_name: str
    feature_count: int
    fold_results: List[FoldRegressionResult]
    oof_predictions_df: pd.DataFrame
    aggregate_oof_metrics: RegressionMetrics
    mean_fold_metrics: Dict[str, float]
    std_fold_metrics: Dict[str, float]
    total_runtime_sec: float
    status: str


@dataclass
class DevRegressionResult:
    """Evaluation result on the official development split."""
    experiment_id: str
    representation: str
    pooling_strategy: str
    model_name: str
    feature_count: int
    train_samples: int
    dev_samples: int
    dev_predictions_df: pd.DataFrame
    metrics: RegressionMetrics
    runtime_sec: float
    status: str


def run_5fold_cv_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    participant_ids: np.ndarray,
    representation: str,
    pooling_strategy: str,
    model_name: str,
    feature_names: Optional[List[str]] = None,
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> CVRegressionResult:
    """
    Executes 5-fold Cross-Validation strictly on official training participants.
    Computes fold-level metrics and out-of-fold aggregate metrics.
    """
    exp_id = f"{representation}__{pooling_strategy}__{model_name}__cv5"
    t_start = time.time()
    feature_count = X_train.shape[1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_results: List[FoldRegressionResult] = []

    oof_preds = np.zeros(len(y_train), dtype=np.float64)
    oof_assigned = np.zeros(len(y_train), dtype=bool)
    oof_folds = np.zeros(len(y_train), dtype=int)

    status = "SUCCESS"

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        f_t0 = time.time()
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        va_pids = participant_ids[val_idx]

        # Verify no participant leakage
        tr_pids_set = set(participant_ids[train_idx])
        va_pids_set = set(va_pids)
        assert len(tr_pids_set.intersection(va_pids_set)) == 0, "Leakage: Participant overlap between train and val folds!"

        # Instantiate fresh pipeline
        pipe = get_regressor(model_name)

        try:
            pipe.fit(X_tr, y_tr)
            y_pred_val = pipe.predict(X_va).astype(np.float64)

            # Record out-of-fold predictions
            oof_preds[val_idx] = y_pred_val
            oof_assigned[val_idx] = True
            oof_folds[val_idx] = fold_idx

            f_metrics = compute_regression_metrics(y_va, y_pred_val)
            f_runtime = time.time() - f_t0

            fold_results.append(FoldRegressionResult(
                fold=fold_idx,
                train_samples=len(train_idx),
                val_samples=len(val_idx),
                metrics=f_metrics,
                y_true_val=y_va,
                y_pred_val=y_pred_val,
                val_participant_ids=va_pids,
                runtime_sec=f_runtime,
                status="SUCCESS",
            ))

        except Exception as e:
            f_runtime = time.time() - f_t0
            err_msg = str(e)
            status = "FAILED"
            dummy_metrics = RegressionMetrics(ccc=0.0, rmse=999.0, mae=999.0, r2=-999.0, warnings=[err_msg])
            fold_results.append(FoldRegressionResult(
                fold=fold_idx,
                train_samples=len(train_idx),
                val_samples=len(val_idx),
                metrics=dummy_metrics,
                y_true_val=y_va,
                y_pred_val=np.zeros_like(y_va),
                val_participant_ids=va_pids,
                runtime_sec=f_runtime,
                status="FAILED",
                error=err_msg,
            ))
        finally:
            # Guarantee no model persistence
            del pipe

    # Check complete OOF coverage
    assert np.all(oof_assigned), "Incomplete out-of-fold prediction coverage!"

    # Compute aggregate OOF metrics
    agg_oof_metrics = compute_regression_metrics(y_train, oof_preds)

    # Compute fold mean and std
    cccs = [f.metrics.ccc for f in fold_results if f.status == "SUCCESS"]
    rmses = [f.metrics.rmse for f in fold_results if f.status == "SUCCESS"]
    maes = [f.metrics.mae for f in fold_results if f.status == "SUCCESS"]
    r2s = [f.metrics.r2 for f in fold_results if f.status == "SUCCESS"]

    mean_metrics = {
        "ccc_mean": float(np.mean(cccs)) if cccs else 0.0,
        "rmse_mean": float(np.mean(rmses)) if rmses else 999.0,
        "mae_mean": float(np.mean(maes)) if maes else 999.0,
        "r2_mean": float(np.mean(r2s)) if r2s else -999.0,
    }
    std_metrics = {
        "ccc_std": float(np.std(cccs, ddof=1)) if len(cccs) > 1 else 0.0,
        "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
    }

    # Build OOF predictions dataframe
    residuals = y_train - oof_preds
    df_oof = pd.DataFrame({
        "Participant_ID": participant_ids,
        "fold": oof_folds,
        "y_true": y_train,
        "y_pred": oof_preds,
        "residual": residuals,
    })

    total_runtime = time.time() - t_start

    return CVRegressionResult(
        experiment_id=exp_id,
        representation=representation,
        pooling_strategy=pooling_strategy,
        model_name=model_name,
        feature_count=feature_count,
        fold_results=fold_results,
        oof_predictions_df=df_oof,
        aggregate_oof_metrics=agg_oof_metrics,
        mean_fold_metrics=mean_metrics,
        std_fold_metrics=std_metrics,
        total_runtime_sec=total_runtime,
        status=status,
    )


def run_development_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    dev_participant_ids: np.ndarray,
    representation: str,
    pooling_strategy: str,
    model_name: str,
    feature_names: Optional[List[str]] = None,
) -> DevRegressionResult:
    """
    Fits regressor on full official training split (n=163) and evaluates on development split (n=56).
    """
    exp_id = f"{representation}__{pooling_strategy}__{model_name}__dev"
    t_start = time.time()
    feature_count = X_train.shape[1]

    pipe = get_regressor(model_name)
    status = "SUCCESS"

    try:
        pipe.fit(X_train, y_train)
        y_pred_dev = pipe.predict(X_dev).astype(np.float64)

        metrics = compute_regression_metrics(y_dev, y_pred_dev)
        residuals = y_dev - y_pred_dev

        df_dev_preds = pd.DataFrame({
            "Participant_ID": dev_participant_ids,
            "y_true": y_dev,
            "y_pred": y_pred_dev,
            "residual": residuals,
        })

    except Exception as e:
        status = "FAILED"
        err_msg = str(e)
        metrics = RegressionMetrics(ccc=0.0, rmse=999.0, mae=999.0, r2=-999.0, warnings=[err_msg])
        df_dev_preds = pd.DataFrame({
            "Participant_ID": dev_participant_ids,
            "y_true": y_dev,
            "y_pred": np.zeros_like(y_dev),
            "residual": y_dev,
        })
    finally:
        # Guarantee no model persistence
        del pipe

    runtime = time.time() - t_start

    return DevRegressionResult(
        experiment_id=exp_id,
        representation=representation,
        pooling_strategy=pooling_strategy,
        model_name=model_name,
        feature_count=feature_count,
        train_samples=len(y_train),
        dev_samples=len(y_dev),
        dev_predictions_df=df_dev_preds,
        metrics=metrics,
        runtime_sec=runtime,
        status=status,
    )
