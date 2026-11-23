"""
Training-Only Cross-Validation Module for E-DAIC Benchmark.

Enforces:
1. Cross-validation is executed strictly on the official training split (163 participants).
2. Development and test sets are never accessed during CV.
3. Preprocessing (StandardScaler) is fitted within each training fold (inside Pipeline).
4. Generates out-of-fold (OOF) predictions for every training participant.
5. Saves fold-level and aggregated metrics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import pandas as pd

from split_loader import SplitData, get_train_cv_splits
from models.classification_models import get_classifier, extract_continuous_scores
from utils.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_roc_curve_data,
    compute_pr_curve_data,
)


@dataclass
class CrossValidationResult:
    """Encapsulates all results from a cross-validation run on the training split."""
    experiment_id: str
    representation: str
    pooling_strategy: str
    model_name: str
    n_splits: int
    fold_metrics: List[ClassificationMetrics]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    oof_metrics: ClassificationMetrics
    oof_predictions_df: pd.DataFrame  # Columns: [Participant_ID, fold, y_true, y_pred, y_score]
    oof_roc_data: Dict[str, Any]
    oof_pr_data: Dict[str, Any]
    total_runtime_sec: float


def run_cross_validation(
    train_data: SplitData,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> CrossValidationResult:
    """
    Runs strictly training-only 5-fold Stratified Cross-Validation for a model.
    """
    if train_data.split_name != "train":
        raise ValueError(
            f"Cross-validation must be run strictly on the official training set. "
            f"Got split_name='{train_data.split_name}'."
        )

    exp_id = f"{train_data.representation}__{train_data.pooling_strategy}__{model_name}__cv{n_splits}"
    folds = get_train_cv_splits(train_data, n_splits=n_splits, stratify=True, random_state=random_state)
    
    fold_metrics: List[ClassificationMetrics] = []
    oof_rows: List[Dict[str, Any]] = []

    t0 = time.time()

    for fold_idx, (tr_indices, val_indices) in enumerate(folds, start=1):
        # 1. Split features and targets for this fold
        X_tr, y_tr = train_data.X[tr_indices], train_data.y_binary[tr_indices]
        X_val, y_val = train_data.X[val_indices], train_data.y_binary[val_indices]
        val_pids = train_data.participant_ids[val_indices]

        # 2. Instantiate clean unfitted pipeline
        pipeline = get_classifier(model_name)

        # 3. Fit pipeline on fold train strictly (scaler fits on X_tr only!)
        pipeline.fit(X_tr, y_tr)

        # 4. Predict on fold validation
        y_pred = pipeline.predict(X_val)
        y_score = extract_continuous_scores(pipeline, X_val)

        # 5. Compute fold metrics
        metrics = compute_classification_metrics(y_val, y_pred, y_score=y_score)
        fold_metrics.append(metrics)

        # 6. Accumulate OOF predictions
        for pid, true_lbl, pred_lbl, score in zip(val_pids, y_val, y_pred, y_score):
            oof_rows.append({
                "Participant_ID": int(pid),
                "fold": fold_idx,
                "y_true": int(true_lbl),
                "y_pred": int(pred_lbl),
                "y_score": float(score),
            })

    total_time = time.time() - t0

    # Assemble OOF DataFrame sorted deterministically by Participant_ID
    oof_df = pd.DataFrame(oof_rows).sort_values("Participant_ID").reset_index(drop=True)

    # Compute overall OOF metrics across all 163 participants
    oof_metrics = compute_classification_metrics(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        y_score=oof_df["y_score"].to_numpy(),
    )

    # Compute mean and std across folds
    metric_keys = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc", "average_precision"]
    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}

    for k in metric_keys:
        vals = [getattr(m, k) for m in fold_metrics if getattr(m, k) is not None]
        if vals:
            mean_metrics[f"{k}_mean"] = float(np.mean(vals))
            std_metrics[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            mean_metrics[f"{k}_mean"] = np.nan
            std_metrics[f"{k}_std"] = np.nan

    # Compute OOF curve coordinate data
    oof_roc_data = compute_roc_curve_data(oof_df["y_true"].to_numpy(), oof_df["y_score"].to_numpy())
    oof_pr_data = compute_pr_curve_data(oof_df["y_true"].to_numpy(), oof_df["y_score"].to_numpy())

    return CrossValidationResult(
        experiment_id=exp_id,
        representation=train_data.representation,
        pooling_strategy=train_data.pooling_strategy,
        model_name=model_name,
        n_splits=n_splits,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        oof_metrics=oof_metrics,
        oof_predictions_df=oof_df,
        oof_roc_data=oof_roc_data,
        oof_pr_data=oof_pr_data,
        total_runtime_sec=total_time,
    )
