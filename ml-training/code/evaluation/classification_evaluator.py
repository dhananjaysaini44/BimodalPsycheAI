"""
Development Evaluation Module for E-DAIC Benchmark.

Evaluates candidate models on the official Development split (56 participants):
1. Model is fit strictly on the complete official training split (163 participants).
2. Evaluated on the held-out development split (56 participants).
3. Generates participant-level predictions, continuous scores, and metrics.
4. Generates ROC and PR curve data for visualization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import pandas as pd

from split_loader import SplitData
from models.classification_models import get_classifier, extract_continuous_scores
from utils.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_roc_curve_data,
    compute_pr_curve_data,
)


@dataclass
class DevelopmentResult:
    """Encapsulates all results from evaluating a model on the development split."""
    experiment_id: str
    representation: str
    pooling_strategy: str
    model_name: str
    metrics: ClassificationMetrics
    predictions_df: pd.DataFrame  # Columns: [Participant_ID, y_true, y_pred, y_score]
    roc_data: Dict[str, Any]
    pr_data: Dict[str, Any]
    runtime_sec: float


def evaluate_on_development(
    train_data: SplitData,
    dev_data: SplitData,
    model_name: str,
) -> DevelopmentResult:
    """
    Fits model on the full official training set (163 participants)
    and evaluates on the official development set (56 participants).
    """
    if train_data.split_name != "train":
        raise ValueError(f"Expected train_data with split_name='train', got '{train_data.split_name}'")
    if dev_data.split_name != "dev":
        raise ValueError(f"Expected dev_data with split_name='dev', got '{dev_data.split_name}'")

    exp_id = f"{train_data.representation}__{train_data.pooling_strategy}__{model_name}__dev"
    t0 = time.time()

    # 1. Instantiate clean unfitted pipeline
    pipeline = get_classifier(model_name)

    # 2. Fit pipeline strictly on full training split
    pipeline.fit(train_data.X, train_data.y_binary)

    # 3. Predict on development split
    y_pred = pipeline.predict(dev_data.X)
    y_score = extract_continuous_scores(pipeline, dev_data.X)

    runtime = time.time() - t0

    # 4. Compute metrics
    metrics = compute_classification_metrics(
        y_true=dev_data.y_binary,
        y_pred=y_pred,
        y_score=y_score,
    )

    # 5. Build predictions DataFrame
    pred_df = pd.DataFrame({
        "Participant_ID": dev_data.participant_ids,
        "y_true": dev_data.y_binary,
        "y_pred": y_pred,
        "y_score": y_score,
    }).sort_values("Participant_ID").reset_index(drop=True)

    # 6. Curve data
    roc_data = compute_roc_curve_data(dev_data.y_binary, y_score)
    pr_data = compute_pr_curve_data(dev_data.y_binary, y_score)

    return DevelopmentResult(
        experiment_id=exp_id,
        representation=train_data.representation,
        pooling_strategy=train_data.pooling_strategy,
        model_name=model_name,
        metrics=metrics,
        predictions_df=pred_df,
        roc_data=roc_data,
        pr_data=pr_data,
        runtime_sec=runtime,
    )
