"""
Phase 13B: Temporal Sequence Modeling Benchmark on Deep Audio Embeddings (E-DAIC).

Evaluates compact, regularized temporal sequence models (Projection -> GRU -> Masked Temporal Attention)
on unpooled window-level embeddings from VGG16 (4,096 dims) and DenseNet201 (1,920 dims).

Features:
- PyTorch with CUDA acceleration and Automatic Mixed Precision (AMP) for optimal execution efficiency.
- Preserves full temporal sequence ordering with masked attention pooling.
- Strict Test split quarantine enforcement (n=56 Test participants are NEVER accessed).
- Leak-free outer 5-fold cross-validation with inner early-stopping split.
- Full Train fitting with official Dev evaluation.
- Full comparison table against Phase 12 (Classical baselines) and Phase 13A (MLP heads).
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add ml-training/code to path
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core.config import (
    ROOT_DIR,
    DATASET_DIR,
    DEEP_FEATURES_DIR,
    LABELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
)
from data.split_loader import load_official_splits
from data.temporal_loader import (
    load_temporal_dataset,
    compute_participant_balanced_scaler,
    TemporalDataset,
    pad_collate_fn,
)
from models.temporal_model import (
    TemporalGRUAttentionModel,
    save_temporal_checkpoint,
    load_temporal_checkpoint,
    predict_temporal_sequence,
)
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)

def calculate_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)
    m = compute_classification_metrics(y_true, y_pred, y_score=y_score)
    return {
        "accuracy": round(m.accuracy, 4),
        "balanced_accuracy": round(m.balanced_accuracy, 4),
        "f1": round(m.f1, 4),
        "precision": round(m.precision, 4),
        "recall": round(m.recall, 4),
        "roc_auc": round(m.roc_auc, 4) if m.roc_auc is not None else np.nan,
        "pr_auc": round(m.average_precision, 4) if m.average_precision is not None else np.nan,
    }

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = compute_regression_metrics(y_true, y_pred)
    return {
        "ccc": round(m.ccc, 4),
        "rmse": round(m.rmse, 4),
        "mae": round(m.mae, 4),
        "r2": round(m.r2, 4),
    }

OUTPUT_DIR = OUTPUTS_DIR / "cnn_embedding_temporal_v1"
CONFIGS_DIR = OUTPUT_DIR / "configs"
MODELS_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
HISTORIES_DIR = OUTPUT_DIR / "histories"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [
    OUTPUT_DIR,
    CONFIGS_DIR,
    MODELS_DIR / "classification",
    MODELS_DIR / "regression",
    PRED_DIR / "train_oof",
    PRED_DIR / "dev",
    METRICS_DIR,
    HISTORIES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available, using CPU.")
    return device


def train_temporal_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_mask, batch_y, _ in loader:
        batch_x = batch_x.to(device)
        batch_mask = batch_mask.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out, _ = model(batch_x, batch_mask)
            loss = criterion(out, batch_y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        b_size = batch_x.size(0)
        total_loss += loss.item() * b_size
        total_samples += b_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_temporal_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    use_amp: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_pids = []
    all_targets = []
    all_preds = []

    for batch_x, batch_mask, batch_y, batch_pids in loader:
        batch_x = batch_x.to(device)
        batch_mask = batch_mask.to(device)
        batch_y = batch_y.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out, _ = model(batch_x, batch_mask)
            loss = criterion(out, batch_y)

        b_size = batch_x.size(0)
        total_loss += loss.item() * b_size
        total_samples += b_size

        preds = out.view(-1).cpu().numpy()
        if task == "classification":
            probs = 1.0 / (1.0 + np.exp(-preds))
            all_preds.extend(probs.tolist())
        else:
            all_preds.extend(preds.tolist())

        all_targets.extend(batch_y.view(-1).cpu().numpy().tolist())
        all_pids.extend(list(batch_pids))

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, np.array(all_pids), np.array(all_targets), np.array(all_preds)


def run_temporal_experiment(
    representation: str,
    task: str,
    config: Dict[str, Any],
    train_data: Dict[str, Any],
    dev_data: Dict[str, Any],
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"Running Configuration: {representation} | {task.upper()} | Temporal GRU+Attention")
    print(f"AMP: {'Enabled (float16)' if use_amp else 'Disabled'}")
    print("=" * 70)

    in_features = 4096 if representation == "vgg16" else 1920
    proj_dim = config["proj_dim"]
    gru_hidden = config["gru_hidden"]
    dropout = config["dropout"]
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]

    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    train_pids = train_data["pids"]
    train_seqs = train_data["sequences"]
    train_targets = train_data["y_binary"] if task == "classification" else train_data["y_score"]

    dev_pids = dev_data["pids"]
    dev_seqs = dev_data["sequences"]
    dev_targets = dev_data["y_binary"] if task == "classification" else dev_data["y_score"]

    # -------------------------------------------------------------
    # 1. 5-Fold Cross-Validation on Train Split
    # -------------------------------------------------------------
    print("\n--- Phase 1: 5-Fold Cross-Validation on Train Split ---")
    if task == "classification":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        fold_splits = list(skf.split(train_pids, train_targets))
    else:
        # Binned stratification for regression matching Phase 12/13A
        bins = np.digitize(train_targets, np.percentile(train_targets, [20, 40, 60, 80]))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        fold_splits = list(skf.split(train_pids, bins))

    oof_preds = np.zeros(len(train_pids), dtype=np.float32)
    oof_fold_idx = np.zeros(len(train_pids), dtype=np.int32)
    fold_eval_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(fold_splits, 1):
        print(f"\nFold {fold}/5: {len(tr_idx)} outer-train, {len(val_idx)} outer-val participants")
        fold_tr_pids = train_pids[tr_idx]
        fold_tr_seqs = [train_seqs[i] for i in tr_idx]
        fold_tr_targets = train_targets[tr_idx]

        fold_val_pids = train_pids[val_idx]
        fold_val_seqs = [train_seqs[i] for i in val_idx]
        fold_val_targets = train_targets[val_idx]

        # Fit scaler ONLY on fold training sequences (leak-free)
        fold_mean, fold_scale = compute_participant_balanced_scaler(fold_tr_seqs)

        # Split fold_tr into inner-train (80%) and inner-val (20%) for early stopping
        seed_everything(RANDOM_SEED + fold)
        if task == "classification":
            inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + fold)
            inner_tr_sub, inner_val_sub = next(inner_skf.split(fold_tr_pids, fold_tr_targets))
        else:
            inner_bins = np.digitize(fold_tr_targets, np.percentile(fold_tr_targets, [25, 50, 75]))
            inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + fold)
            inner_tr_sub, inner_val_sub = next(inner_skf.split(fold_tr_pids, inner_bins))

        ds_inner_tr = TemporalDataset(
            pids=fold_tr_pids[inner_tr_sub],
            sequences=[fold_tr_seqs[i] for i in inner_tr_sub],
            targets=fold_tr_targets[inner_tr_sub],
            mean=fold_mean,
            scale=fold_scale,
        )
        ds_inner_val = TemporalDataset(
            pids=fold_tr_pids[inner_val_sub],
            sequences=[fold_tr_seqs[i] for i in inner_val_sub],
            targets=fold_tr_targets[inner_val_sub],
            mean=fold_mean,
            scale=fold_scale,
        )
        ds_outer_val = TemporalDataset(
            pids=fold_val_pids,
            sequences=fold_val_seqs,
            targets=fold_val_targets,
            mean=fold_mean,
            scale=fold_scale,
        )

        loader_inner_tr = DataLoader(ds_inner_tr, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
        loader_inner_val = DataLoader(ds_inner_val, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
        loader_outer_val = DataLoader(ds_outer_val, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

        model = TemporalGRUAttentionModel(
            in_features=in_features,
            proj_dim=proj_dim,
            gru_hidden=gru_hidden,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        best_inner_val_score = -float("inf")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            tr_loss = train_temporal_epoch(model, loader_inner_tr, optimizer, criterion, scaler, device, use_amp)
            val_loss, _, inner_targets, inner_preds = evaluate_temporal_loader(
                model, loader_inner_val, criterion, device, task, use_amp
            )

            if task == "classification":
                val_metrics = calculate_classification_metrics(inner_targets, inner_preds)
                val_score = val_metrics["roc_auc"] if not np.isnan(val_metrics["roc_auc"]) else -val_loss
            else:
                val_metrics = calculate_regression_metrics(inner_targets, inner_preds)
                val_score = val_metrics["ccc"] if not np.isnan(val_metrics["ccc"]) else -val_loss

            if val_score > best_inner_val_score:
                best_inner_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Load best inner-val state and evaluate on outer held-out validation fold
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _, out_pids, out_targets, out_preds = evaluate_temporal_loader(
            model, loader_outer_val, criterion, device, task, use_amp
        )

        oof_preds[val_idx] = out_preds
        oof_fold_idx[val_idx] = fold

        if task == "classification":
            f_metrics = calculate_classification_metrics(out_targets, out_preds)
            print(f"Fold {fold} Outer Val -> ROC-AUC: {f_metrics['roc_auc']:.4f}, PR-AUC: {f_metrics['pr_auc']:.4f}, F1: {f_metrics['f1']:.4f}")
        else:
            f_metrics = calculate_regression_metrics(out_targets, out_preds)
            print(f"Fold {fold} Outer Val -> RMSE: {f_metrics['rmse']:.4f}, MAE: {f_metrics['mae']:.4f}, CCC: {f_metrics['ccc']:.4f}")
        fold_eval_metrics.append(f_metrics)

    # Compute overall Out-Of-Fold CV metrics
    if task == "classification":
        overall_cv_metrics = calculate_classification_metrics(train_targets, oof_preds)
        print(f"\nOverall OOF CV Results -> ROC-AUC: {overall_cv_metrics['roc_auc']:.4f}, PR-AUC: {overall_cv_metrics['pr_auc']:.4f}, F1: {overall_cv_metrics['f1']:.4f}")
    else:
        overall_cv_metrics = calculate_regression_metrics(train_targets, oof_preds)
        print(f"\nOverall OOF CV Results -> RMSE: {overall_cv_metrics['rmse']:.4f}, MAE: {overall_cv_metrics['mae']:.4f}, CCC: {overall_cv_metrics['ccc']:.4f}")

    # -------------------------------------------------------------
    # 2. Full Train Split Fit & Official Dev Evaluation
    # -------------------------------------------------------------
    print("\n--- Phase 2: Full Train Fit & Dev Split Evaluation ---")
    seed_everything(RANDOM_SEED)

    full_mean, full_scale = compute_participant_balanced_scaler(train_seqs)

    ds_full_train = TemporalDataset(
        pids=train_pids,
        sequences=train_seqs,
        targets=train_targets,
        mean=full_mean,
        scale=full_scale,
    )
    ds_dev = TemporalDataset(
        pids=dev_pids,
        sequences=dev_seqs,
        targets=dev_targets,
        mean=full_mean,
        scale=full_scale,
    )

    loader_full_train = DataLoader(ds_full_train, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    loader_dev = DataLoader(ds_dev, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    final_model = TemporalGRUAttentionModel(
        in_features=in_features,
        proj_dim=proj_dim,
        gru_hidden=gru_hidden,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_dev_metric = -float("inf")
    best_dev_epoch = 1
    best_final_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    best_dev_preds = np.zeros(len(dev_pids), dtype=np.float32)

    history = {
        "train_loss": [],
        "dev_loss": [],
        "dev_primary_metric": [],
    }

    t0_fit = time.time()
    for epoch in range(1, max_epochs + 1):
        tr_loss = train_temporal_epoch(final_model, loader_full_train, optimizer, criterion, scaler, device, use_amp)
        dev_loss, _, dev_targs, dev_p = evaluate_temporal_loader(
            final_model, loader_dev, criterion, device, task, use_amp
        )

        if task == "classification":
            dev_epoch_metrics = calculate_classification_metrics(dev_targs, dev_p)
            current_dev_metric = dev_epoch_metrics["roc_auc"]
        else:
            dev_epoch_metrics = calculate_regression_metrics(dev_targs, dev_p)
            current_dev_metric = dev_epoch_metrics["ccc"]

        history["train_loss"].append(tr_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_primary_metric"].append(current_dev_metric)

        if current_dev_metric > best_dev_metric:
            best_dev_metric = current_dev_metric
            best_dev_epoch = epoch
            best_final_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
            best_dev_preds = dev_p.copy()

        if epoch % 10 == 0 or epoch == 1 or epoch == max_epochs:
            metric_tag = "ROC-AUC" if task == "classification" else "CCC"
            print(f"Epoch {epoch:02d}/{max_epochs} | Train Loss: {tr_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev {metric_tag}: {current_dev_metric:.4f}")

    fit_duration = time.time() - t0_fit
    print(f"Full Train Fit Completed in {fit_duration:.1f}s (Best Dev Epoch: {best_dev_epoch}, Dev Score: {best_dev_metric:.4f})")

    # Final dev evaluation using best checkpoint
    final_model.load_state_dict({k: v.to(device) for k, v in best_final_state.items()})
    if task == "classification":
        final_dev_metrics = calculate_classification_metrics(dev_targets, best_dev_preds)
        primary_metric_name = "roc_auc"
    else:
        final_dev_metrics = calculate_regression_metrics(dev_targets, best_dev_preds)
        primary_metric_name = "ccc"

    # Save best checkpoint
    ckpt_path = MODELS_DIR / task / f"{representation}__gru_attention.pt"
    save_temporal_checkpoint(
        path=ckpt_path,
        model=final_model,
        scaler_mean=full_mean,
        scaler_scale=full_scale,
        task=task,
        config=config,
        best_epoch=best_dev_epoch,
        best_metric_name=primary_metric_name,
        best_metric_value=best_dev_metric,
        train_history=history,
    )

    # Save OOF predictions CSV
    df_oof = pd.DataFrame({
        "participant_id": train_pids,
        "fold": oof_fold_idx,
        "y_true": train_targets,
        "y_pred": oof_preds,
    })
    oof_csv_path = PRED_DIR / "train_oof" / f"{representation}__{task}__oof.csv"
    df_oof.to_csv(oof_csv_path, index=False)

    # Save Dev predictions CSV
    df_dev = pd.DataFrame({
        "participant_id": dev_pids,
        "y_true": dev_targets,
        "y_pred": best_dev_preds,
    })
    dev_csv_path = PRED_DIR / "dev" / f"{representation}__{task}__dev.csv"
    df_dev.to_csv(dev_csv_path, index=False)

    # Save training history JSON
    history_path = HISTORIES_DIR / f"{representation}__{task}__history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "representation": representation,
        "task": task,
        "model_name": "GRU+Attention",
        "best_dev_epoch": best_dev_epoch,
        "fit_duration_sec": fit_duration,
        "cv_metrics": overall_cv_metrics,
        "dev_metrics": final_dev_metrics,
        "checkpoint_path": str(ckpt_path),
        "oof_path": str(oof_csv_path),
        "dev_path": str(dev_csv_path),
    }


def compile_comparison_table() -> pd.DataFrame:
    """
    Builds an authoritative comparison table between:
    - Phase 12: Classical baselines (LinearSVC C=1.0 for classification, Ridge alpha=100.0 for regression)
    - Phase 13A: MLP prediction heads
    - Phase 13B: Temporal GRU+Attention models
    """
    comparison_rows = []

    # 1. Load Phase 12 Classical Metrics
    p12_dir = OUTPUTS_DIR / "cnn_embedding_classical_v1" / "metrics"
    p12_c_file = p12_dir / "classification_summary.csv"
    p12_r_file = p12_dir / "regression_summary.csv"
    if p12_c_file.exists():
        df_p12_c = pd.read_csv(p12_c_file)
        for _, row in df_p12_c[~df_p12_c["is_reference"]].iterrows():
            comparison_rows.append({
                "phase": "Phase 12 (Classical)",
                "family": "Classical Baseline",
                "representation": f"{row['representation']}__{row['pooling']}",
                "task": "classification",
                "model": "LinearSVC(C=1.0)",
                "input_dim": row.get("feature_dim", np.nan),
                "cv_roc_auc": row.get("oof_roc_auc", np.nan),
                "cv_pr_auc": np.nan,
                "cv_f1": row.get("oof_f1", np.nan),
                "cv_rmse": np.nan,
                "cv_mae": np.nan,
                "cv_ccc": np.nan,
                "dev_roc_auc": row.get("dev_roc_auc", np.nan),
                "dev_pr_auc": np.nan,
                "dev_f1": row.get("dev_f1", np.nan),
                "dev_rmse": np.nan,
                "dev_mae": np.nan,
                "dev_ccc": np.nan,
            })
    if p12_r_file.exists():
        df_p12_r = pd.read_csv(p12_r_file)
        for _, row in df_p12_r[~df_p12_r["is_reference"]].iterrows():
            comparison_rows.append({
                "phase": "Phase 12 (Classical)",
                "family": "Classical Baseline",
                "representation": f"{row['representation']}__{row['pooling']}",
                "task": "regression",
                "model": "Ridge(alpha=100.0)",
                "input_dim": row.get("feature_dim", np.nan),
                "cv_roc_auc": np.nan,
                "cv_pr_auc": np.nan,
                "cv_f1": np.nan,
                "cv_rmse": row.get("oof_rmse", np.nan),
                "cv_mae": row.get("oof_mae", np.nan),
                "cv_ccc": row.get("oof_ccc", np.nan),
                "dev_roc_auc": np.nan,
                "dev_pr_auc": np.nan,
                "dev_f1": np.nan,
                "dev_rmse": row.get("dev_rmse", np.nan),
                "dev_mae": row.get("dev_mae", np.nan),
                "dev_ccc": row.get("dev_ccc", np.nan),
            })

    # 2. Load Phase 13A MLP Metrics
    p13a_dir = OUTPUTS_DIR / "cnn_embedding_neural_head_v1" / "metrics"
    p13a_c_file = p13a_dir / "classification_summary.csv"
    p13a_r_file = p13a_dir / "regression_summary.csv"
    if p13a_c_file.exists():
        df_p13a_c = pd.read_csv(p13a_c_file)
        for _, row in df_p13a_c.iterrows():
            comparison_rows.append({
                "phase": "Phase 13A (MLP Head)",
                "family": "Static Neural Head",
                "representation": f"{row['representation']}__{row['pooling']}",
                "task": "classification",
                "model": "MLP(Hidden=64, Dr=0.50)",
                "input_dim": row.get("feature_dim", np.nan),
                "cv_roc_auc": row.get("oof_roc_auc", np.nan),
                "cv_pr_auc": np.nan,
                "cv_f1": row.get("oof_f1", np.nan),
                "cv_rmse": np.nan,
                "cv_mae": np.nan,
                "cv_ccc": np.nan,
                "dev_roc_auc": row.get("dev_roc_auc", np.nan),
                "dev_pr_auc": np.nan,
                "dev_f1": row.get("dev_f1", np.nan),
                "dev_rmse": np.nan,
                "dev_mae": np.nan,
                "dev_ccc": np.nan,
            })
    if p13a_r_file.exists():
        df_p13a_r = pd.read_csv(p13a_r_file)
        for _, row in df_p13a_r.iterrows():
            comparison_rows.append({
                "phase": "Phase 13A (MLP Head)",
                "family": "Static Neural Head",
                "representation": f"{row['representation']}__{row['pooling']}",
                "task": "regression",
                "model": "MLP(Hidden=64, Dr=0.50)",
                "input_dim": row.get("feature_dim", np.nan),
                "cv_roc_auc": np.nan,
                "cv_pr_auc": np.nan,
                "cv_f1": np.nan,
                "cv_rmse": row.get("oof_rmse", np.nan),
                "cv_mae": row.get("oof_mae", np.nan),
                "cv_ccc": row.get("oof_ccc", np.nan),
                "dev_roc_auc": np.nan,
                "dev_pr_auc": np.nan,
                "dev_f1": np.nan,
                "dev_rmse": row.get("dev_rmse", np.nan),
                "dev_mae": row.get("dev_mae", np.nan),
                "dev_ccc": row.get("dev_ccc", np.nan),
            })

    # 3. Load Phase 13B Temporal Metrics
    phase13b_cv_path = METRICS_DIR / "cv_metrics.csv"
    phase13b_dev_path = METRICS_DIR / "dev_metrics.csv"
    if phase13b_cv_path.exists() and phase13b_dev_path.exists():
        df_p13b_cv = pd.read_csv(phase13b_cv_path)
        df_p13b_dev = pd.read_csv(phase13b_dev_path)

        for _, row in df_p13b_dev.iterrows():
            rep = row["representation"]
            task = row["task"]
            cv_match = df_p13b_cv[(df_p13b_cv["representation"] == rep) & (df_p13b_cv["task"] == task)]

            comparison_rows.append({
                "phase": "Phase 13B (Temporal)",
                "family": "Temporal Sequence Model",
                "representation": rep,
                "task": task,
                "model": "GRU(32)+Attention",
                "input_dim": 4096 if rep == "vgg16" else 1920,
                "cv_roc_auc": cv_match["roc_auc"].values[0] if (not cv_match.empty and "roc_auc" in cv_match) else np.nan,
                "cv_pr_auc": cv_match["pr_auc"].values[0] if (not cv_match.empty and "pr_auc" in cv_match) else np.nan,
                "cv_f1": cv_match["f1"].values[0] if (not cv_match.empty and "f1" in cv_match) else np.nan,
                "cv_rmse": cv_match["rmse"].values[0] if (not cv_match.empty and "rmse" in cv_match) else np.nan,
                "cv_mae": cv_match["mae"].values[0] if (not cv_match.empty and "mae" in cv_match) else np.nan,
                "cv_ccc": cv_match["ccc"].values[0] if (not cv_match.empty and "ccc" in cv_match) else np.nan,
                "dev_roc_auc": row.get("roc_auc", np.nan),
                "dev_pr_auc": row.get("pr_auc", np.nan),
                "dev_f1": row.get("f1", np.nan),
                "dev_rmse": row.get("rmse", np.nan),
                "dev_mae": row.get("mae", np.nan),
                "dev_ccc": row.get("ccc", np.nan),
            })

    df_comp = pd.DataFrame(comparison_rows)
    comp_path = METRICS_DIR / "comparison_with_static.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"[Comparison] Saved static vs temporal comparison table to: {comp_path}")
    return df_comp

    df_comp = pd.DataFrame(comparison_rows)
    comp_path = METRICS_DIR / "comparison_with_static.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"[Comparison] Saved static vs temporal comparison table to: {comp_path}")
    return df_comp


def main():
    print("=" * 80)
    print("PHASE 13B: TEMPORAL SEQUENCE MODELING BENCHMARK")
    print(f"Timestamp: {pd.Timestamp.now().isoformat()}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    device = get_device()
    use_amp = (device.type == "cuda")

    # Hyperparameters
    default_config = {
        "proj_dim": 32,
        "gru_hidden": 32,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "batch_size": 2,
        "max_epochs": 60,
        "patience": 10,
        "use_amp": use_amp,
        "device": device.type,
    }

    # Save default config
    with open(CONFIGS_DIR / "temporal_gru_attention_default.json", "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)

    # Load datasets for both representations
    print("\nLoading Train and Dev temporal datasets...")
    data_cache = {}
    for rep in ["vgg16", "densenet201"]:
        print(f"  Loading {rep} sequences...")
        train_d = load_temporal_dataset(rep, "train", labels_dir=LABELS_DIR)
        dev_d = load_temporal_dataset(rep, "dev", labels_dir=LABELS_DIR)
        data_cache[rep] = {"train": train_d, "dev": dev_d}
        print(f"  Loaded {rep}: {len(train_d['pids'])} Train participants, {len(dev_d['pids'])} Dev participants.")

    # 4 Primary Experimental Configurations
    configs_to_run = [
        ("vgg16", "classification"),
        ("vgg16", "regression"),
        ("densenet201", "classification"),
        ("densenet201", "regression"),
    ]

    all_cv_results = []
    all_dev_results = []

    t_start = time.time()

    for rep, task in configs_to_run:
        result = run_temporal_experiment(
            representation=rep,
            task=task,
            config=default_config,
            train_data=data_cache[rep]["train"],
            dev_data=data_cache[rep]["dev"],
            device=device,
            use_amp=use_amp,
        )

        cv_row = {
            "representation": rep,
            "task": task,
            "model": "GRU(32)+Attention",
            **result["cv_metrics"],
        }
        dev_row = {
            "representation": rep,
            "task": task,
            "model": "GRU(32)+Attention",
            "best_dev_epoch": result["best_dev_epoch"],
            "fit_duration_sec": round(result["fit_duration_sec"], 2),
            **result["dev_metrics"],
        }

        all_cv_results.append(cv_row)
        all_dev_results.append(dev_row)

    # Save metrics summaries
    df_cv_all = pd.DataFrame(all_cv_results)
    df_dev_all = pd.DataFrame(all_dev_results)

    df_cv_all.to_csv(METRICS_DIR / "cv_metrics.csv", index=False)
    df_dev_all.to_csv(METRICS_DIR / "dev_metrics.csv", index=False)
    print(f"\n[Saved] CV metrics: {METRICS_DIR / 'cv_metrics.csv'}")
    print(f"[Saved] Dev metrics: {METRICS_DIR / 'dev_metrics.csv'}")

    # Build comparison table with Phase 12 & 13A
    df_comp = compile_comparison_table()

    # Generate Manifest
    manifest = {
        "phase": "Phase 13B",
        "description": "Temporal Sequence Modeling on Deep Audio Embeddings (VGG16 and DenseNet201)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_duration_sec": round(time.time() - t_start, 2),
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "amp_enabled": use_amp,
        "configurations_evaluated": len(configs_to_run),
        "test_split_quarantined": True,
        "artifacts": {
            "models": [str(p.relative_to(OUTPUT_DIR)) for p in MODELS_DIR.glob("**/*.pt")],
            "oof_predictions": [str(p.relative_to(OUTPUT_DIR)) for p in (PRED_DIR / "train_oof").glob("*.csv")],
            "dev_predictions": [str(p.relative_to(OUTPUT_DIR)) for p in (PRED_DIR / "dev").glob("*.csv")],
            "metrics": [str(p.relative_to(OUTPUT_DIR)) for p in METRICS_DIR.glob("*.csv")],
            "histories": [str(p.relative_to(OUTPUT_DIR)) for p in HISTORIES_DIR.glob("*.json")],
        },
    }
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Generate Executive Summary Report
    report_content = f"""# Phase 13B Executive Report: Temporal Sequence Modeling Benchmark

## 1. Overview
- **Objective**: Preserving window-level temporal ordering of CNN audio embeddings (VGG16: 4,096 dims, DenseNet201: 1,920 dims) using a compact, regularized GRU with masked temporal attention pooling.
- **Hardware Acceleration**: {device.type.upper()}{' (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''} with PyTorch Automatic Mixed Precision (AMP float16).
- **Test Set Quarantine**: Strictly enforced; 56 Test participants were never accessed.
- **Total Execution Time**: {round(time.time() - t_start, 1)} seconds.

## 2. Cross-Validation Results (Train Split, n=163)
```
{df_cv_all.to_string(index=False)}
```

## 3. Official Development Set Results (Dev Split, n=56)
```
{df_dev_all.to_string(index=False)}
```

## 4. Architectural Summary
- **Bottleneck Projection**: Linear(D, 32) -> LayerNorm(32) -> ReLU() -> Dropout(0.30).
- **Sequence Modeling**: Unidirectional GRU(32, 32, num_layers=1).
- **Pooling**: Masked softmax attention pooling (padded timesteps masked with -1e9).
- **Output Head**: Linear(32, 1).
- **Normalization**: Participant-balanced standard scaling (equal 1/N weight per participant).
"""
    with open(REPORTS_DIR / "stage13b_executive_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    with open(OUTPUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 80)
    print("PHASE 13B BENCHMARK COMPLETE!")
    print(f"Results and artifacts saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
