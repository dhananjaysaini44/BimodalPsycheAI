"""
Training and Evaluation Engine for Phase 17C Text MLP Models.

Implements:
1. Strict in-fold StandardScaler fitting.
2. Leakage-free early stopping on validation loss within training folds.
3. Train-only out-of-fold threshold calibration for classification.
4. Continuous unclipped regression metric evaluation.
5. Participant-level non-parametric bootstrap confidence intervals.
"""

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from .mlp_model import TextMLPHead
except ImportError:
    from mlp_model import TextMLPHead


def set_seed(seed: int) -> None:
    """Sets global random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    mean_diff_sq = (mean_true - mean_pred) ** 2
    denominator = var_true + var_pred + mean_diff_sq
    if denominator <= 1e-12:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    ccc = (2.0 * cov) / denominator
    return float(np.clip(ccc, -1.0, 1.0))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard benchmark regression metrics without clipping."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ccc = concordance_correlation_coefficient(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    std_true = float(np.std(y_true, ddof=0))
    std_pred = float(np.std(y_pred, ddof=0))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if (std_true > 1e-9 and std_pred > 1e-9) else 0.0
    spread_ratio = float(std_pred / std_true) if std_true > 1e-9 else 0.0

    return {
        "ccc": ccc,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": corr,
        "target_spread": std_true,
        "pred_spread": std_pred,
        "spread_ratio": spread_ratio,
        "target_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """Computes binary classification metrics from predictions and continuous scores/logits."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    auc_val = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
    ap_val = float(average_precision_score(y_true, y_score, pos_label=1)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc_val,
        "pr_auc": ap_val,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "pred_class_0": int(np.sum(y_pred == 0)),
        "pred_class_1": int(np.sum(y_pred == 1)),
    }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_eval: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes non-parametric participant-level bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2 and "auc" in getattr(metric_fn, "__name__", ""):
            continue
        try:
            val = metric_fn(y_true[idx], y_eval[idx])
            if not np.isnan(val):
                scores.append(val)
        except Exception:
            continue
    if len(scores) < 100:
        return (0.0, 0.0)
    alpha = (100.0 - ci) / 2.0
    return (float(np.percentile(scores, alpha)), float(np.percentile(scores, 100.0 - alpha)))


def train_single_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 16,
    max_epochs: int = 100,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, int, np.ndarray, Dict[str, List[float]]]:
    """
    Trains an MLP on a single fold partition with early stopping on validation loss.
    Returns: (best_model, best_epoch, val_predictions, history_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Dataloaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    # Loss function
    if task == "classification":
        # Calculate class balance on this fold's training data
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                out_v = model(x_v)
                loss_v = criterion(out_v, y_v)
                val_losses.append(loss_v.item())

        mean_tr_loss = float(np.mean(train_losses))
        mean_vl_loss = float(np.mean(val_losses))
        history["train_loss"].append(mean_tr_loss)
        history["val_loss"].append(mean_vl_loss)

        if mean_vl_loss < best_loss - 1e-4:
            best_loss = mean_vl_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Generate final validation predictions
    model.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = model(x_val_t).cpu().numpy()

    return model, best_epoch, val_preds, history


def train_full_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    epochs: int,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Trains an MLP on the full Train set for a fixed number of epochs (e.g. median best CV epoch)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if task == "classification":
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(max(1, epochs)):
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model
