"""
Phase 18C Experiment Runner: Trainable Attention Pooling on Frozen DistilBERT Embeddings.

Features:
- Participant-level 5-fold Stratified CV (Classification) and K-Fold CV (Regression) on Train (N=163).
- Train-only hyperparameter tuning over compact declared grid.
- Early stopping (patience 10, max 100 epochs) evaluated strictly on internal validation fold.
- Threshold calibration (tau*) evaluated strictly on Train OOF logits across 300 candidates.
- Multi-seed execution across seeds 42, 123, 456.
- Leak-free Development evaluation (N=56 full cohort and N=54 matched complete cohort).
- Paired participant-level bootstrap difference tests (B=2,000) against Phase 18B fixed-pooling champions.
- Diagnostic visualization generation.
"""

import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from text_attention_v1.models import AttentionClassifier, AttentionRegressor
from text_attention_v1.dataset import (
    load_participant_chunk_sequences,
    load_targets,
    ParticipantSequenceDataset,
    collate_variable_sequences,
)
from text_attention_v1.plots import (
    plot_classification_benchmark,
    plot_regression_benchmark,
    plot_cv_fold_variability,
    plot_seed_stability,
    plot_paired_bootstrap_differences,
    plot_training_loss_curves,
    plot_attention_weight_profiles,
)

# Root directory configuration
REPO_ROOT = Path(__file__).resolve().parents[3]
ML_TRAINING = REPO_ROOT / "ml-training"
DATASET_DIR = ML_TRAINING / "dataset" / "text_distilbert_v1"
LABELS_DIR = ML_TRAINING / "edaic-woz" / "labels"
OUTPUT_DIR = ML_TRAINING / "outputs" / "text_attention_v1"
PRED_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"
CV_DIR = OUTPUT_DIR / "cv_results"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# Configure logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "phase_18c_experiments.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Phase18C")


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Lin's Concordance Correlation Coefficient using population moments (ddof=0)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    true_var = np.var(y_true, ddof=0)
    pred_var = np.var(y_pred, ddof=0)
    covariance = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    denom = true_var + pred_var + (true_mean - pred_mean) ** 2
    if denom == 0:
        return 0.0
    return float((2.0 * covariance) / denom)


def bootstrap_ci(
    metric_fn: Any,
    y_true: np.ndarray,
    y_eval: np.ndarray,
    n_bootstraps: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval over paired participant resamples."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_stats = []
    alpha_low = (100.0 - ci) / 2.0
    alpha_high = 100.0 - alpha_low

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_sample = y_true[idx]
        ye_sample = y_eval[idx]
        try:
            stat = metric_fn(yt_sample, ye_sample)
            if not np.isnan(stat) and not np.isinf(stat):
                boot_stats.append(stat)
        except Exception:
            continue

    if len(boot_stats) < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(boot_stats, alpha_low)), float(np.percentile(boot_stats, alpha_high)))


def calibrate_threshold(y_true: np.ndarray, y_scores: np.ndarray, n_candidates: int = 300) -> Tuple[float, float]:
    """
    Finds threshold tau* that maximizes F1 score on Train OOF decision scores (logits).
    """
    s_min, s_max = np.min(y_scores), np.max(y_scores)
    if s_min == s_max:
        return 0.0, float(f1_score(y_true, (y_scores >= s_min).astype(int), pos_label=1, zero_division=0))

    thresholds = np.linspace(s_min, s_max, n_candidates)
    best_f1 = -1.0
    best_tau = 0.0

    for tau in thresholds:
        preds = (y_scores >= tau).astype(int)
        score = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_tau = float(tau)

    return best_tau, best_f1


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_epochs: int = 100,
    patience: int = 10,
) -> Tuple[nn.Module, List[float], List[float], int]:
    """
    Trains a fold model with internal early stopping on validation loss.
    Returns: best_model, train_loss_history, val_loss_history, best_epoch
    """
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        batch_train_losses = []
        for bx, bmask, by, _ in train_loader:
            bx, bmask, by = bx.to(device), bmask.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx, bmask)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

        epoch_train_loss = float(np.mean(batch_train_losses))
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for bx, bmask, by, _ in val_loader:
                bx, bmask, by = bx.to(device), bmask.to(device), by.to(device)
                out, _ = model(bx, bmask)
                loss = criterion(out, by)
                batch_val_losses.append(loss.item())

        epoch_val_loss = float(np.mean(batch_val_losses))
        val_losses.append(epoch_val_loss)

        # Early stopping check
        if epoch_val_loss < best_loss - 1e-4:
            best_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, np.ndarray]]:
    """
    Evaluates model on loader, returning true labels, continuous predictions/logits,
    participant IDs, and attention weight arrays for each participant.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_pids = []
    attn_dict = {}

    with torch.no_grad():
        for bx, bmask, by, bpids in loader:
            bx, bmask = bx.to(device), bmask.to(device)
            out, alpha = model(bx, bmask)

            all_preds.extend(out.cpu().numpy().tolist())
            all_targets.extend(by.numpy().tolist())
            all_pids.extend(bpids)

            # Store attention weights
            alpha_np = alpha.cpu().numpy()
            mask_np = bmask.cpu().numpy()
            for i, pid in enumerate(bpids):
                valid_len = int(np.sum(mask_np[i]))
                attn_dict[pid] = alpha_np[i, :valid_len]

    return np.array(all_targets), np.array(all_preds), all_pids, attn_dict


def run_cv_grid_search(
    task: str,
    arm: str,
    chunk_dict: Dict[int, np.ndarray],
    df_train: pd.DataFrame,
    param_grid: List[Dict[str, Any]],
    device: str,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs grid search strictly on Train 5-fold CV to select optimal hyperparameters.
    """
    logger.info(f"Starting Train CV Grid Search for {arm} on {task} (Seed {seed}, {len(param_grid)} configs)...")
    pids = df_train["Participant_ID"].to_numpy()
    targets = df_train["target"].to_numpy()
    labels_dict = dict(zip(pids, targets))

    if task == "classification":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(skf.split(pids, targets.astype(int)))
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(kf.split(pids, targets))

    grid_records = []

    for cfg_idx, cfg in enumerate(param_grid):
        fold_scores = []
        oof_preds = np.zeros(len(pids), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_pids = pids[train_idx].tolist()
            val_pids = pids[val_idx].tolist()

            train_ds = ParticipantSequenceDataset(train_pids, chunk_dict, labels_dict)
            val_ds = ParticipantSequenceDataset(val_pids, chunk_dict, labels_dict)

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_variable_sequences)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_variable_sequences)

            set_seed(seed + fold)
            if task == "classification":
                model = AttentionClassifier(
                    embed_dim=768, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]
                ).to(device)
                # Compute pos_weight from training fold only
                y_tr = targets[train_idx]
                n_pos = np.sum(y_tr == 1)
                n_neg = len(y_tr) - n_pos
                pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                model = AttentionRegressor(
                    embed_dim=768, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]
                ).to(device)
                criterion = nn.MSELoss()

            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

            best_model, _, _, _ = train_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_epochs=100,
                patience=10,
            )

            yt_val, yp_val, val_pids_ret, _ = evaluate_loader(best_model, val_loader, device)
            oof_preds[val_idx] = yp_val

            if task == "classification":
                # Fold threshold search
                tau_fold, fold_f1 = calibrate_threshold(yt_val, yp_val)
                fold_scores.append(fold_f1)
            else:
                fold_ccc = compute_ccc(yt_val, yp_val)
                fold_scores.append(fold_ccc)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        if task == "classification":
            tau_oof, oof_f1 = calibrate_threshold(targets, oof_preds)
            oof_eval_score = oof_f1
        else:
            oof_ccc = compute_ccc(targets, oof_preds)
            oof_eval_score = oof_ccc

        rec = {
            "task": task,
            "arm": arm,
            "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "mean_cv_score": mean_score,
            "std_cv_score": std_score,
            "oof_eval_score": oof_eval_score,
        }
        grid_records.append(rec)
        logger.info(f"  Cfg {cfg_idx+1:02d}/{len(param_grid):02d} | h={cfg['hidden_dim']}, drop={cfg['dropout']}, lr={cfg['lr']}, wd={cfg['weight_decay']} -> Mean CV={mean_score:.4f} (±{std_score:.4f}), OOF={oof_eval_score:.4f}")

    df_grid = pd.DataFrame(grid_records)
    # Select best config by mean CV score
    best_row = df_grid.sort_values("mean_cv_score", ascending=False).iloc[0]
    best_cfg = {
        "hidden_dim": int(best_row["hidden_dim"]),
        "dropout": float(best_row["dropout"]),
        "lr": float(best_row["lr"]),
        "weight_decay": float(best_row["weight_decay"]),
    }
    logger.info(f"Optimal Train-CV Configuration for {arm} ({task}): {best_cfg} (Mean CV={best_row['mean_cv_score']:.4f})")
    return best_cfg, df_grid


def execute_phase18c() -> None:
    """Master execution pipeline for Phase 18C."""
    logger.info("================================================================================")
    logger.info("PHASE 18C: TRAINABLE ATTENTION POOLING OVER FROZEN DISTILBERT EMBEDDINGS")
    logger.info("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load data
    logger.info("Loading participant chunk sequences and targets...")
    whole_chunks, _ = load_participant_chunk_sequences(DATASET_DIR, segment_mode="whole")
    middle_chunks, _ = load_participant_chunk_sequences(DATASET_DIR, segment_mode="middle")

    df_train_cls, df_dev_cls = load_targets(LABELS_DIR, task="classification")
    df_train_reg, df_dev_reg = load_targets(LABELS_DIR, task="regression")

    matched_pids_dev = set(pd.read_csv(DATASET_DIR / "features" / "dev" / "complete_cohort_participant_ids.csv")["Participant_ID"])
    logger.info(f"Loaded: Train N={len(df_train_cls)}, Dev N={len(df_dev_cls)} (Matched N={len(matched_pids_dev)})")

    # 2. Predeclared Hyperparameter Grid
    param_grid = []
    for h in [32, 64]:
        for d in [0.2, 0.4]:
            for lr in [0.001, 0.0003]:
                for wd in [0.0001, 0.001]:
                    param_grid.append({
                        "hidden_dim": h,
                        "dropout": d,
                        "lr": lr,
                        "weight_decay": wd,
                    })
    logger.info(f"Predeclared hyperparameter grid contains {len(param_grid)} configurations.")

    seeds = [42, 123, 456]
    experiments = [
        {"arm": "whole_attention", "task": "classification", "chunks": whole_chunks, "df_tr": df_train_cls, "df_dev": df_dev_cls},
        {"arm": "middle_attention", "task": "classification", "chunks": middle_chunks, "df_tr": df_train_cls, "df_dev": df_dev_cls},
        {"arm": "whole_attention", "task": "regression", "chunks": whole_chunks, "df_tr": df_train_reg, "df_dev": df_dev_reg},
        {"arm": "middle_attention", "task": "regression", "chunks": middle_chunks, "df_tr": df_train_reg, "df_dev": df_dev_reg},
    ]

    all_results = []
    fold_records = []
    seed_records = []
    history_curves = {}
    saved_dev_predictions = {}
    example_attention_weights = []

    for exp in experiments:
        arm = exp["arm"]
        task = exp["task"]
        chunk_dict = exp["chunks"]
        df_tr = exp["df_tr"]
        df_dv = exp["df_dev"]

        logger.info("\n--------------------------------------------------------------------------------")
        logger.info(f"EXPERIMENT: {arm.upper()} | TASK: {task.upper()}")
        logger.info("--------------------------------------------------------------------------------")

        # Step 1: Run Grid Search on Train CV using Seed 42
        best_cfg, df_grid = run_cv_grid_search(task, arm, chunk_dict, df_tr, param_grid, device, seed=42)
        df_grid.to_csv(CV_DIR / f"grid_search_{arm}_{task}.csv", index=False)

        curve_key = f"{arm}_{task}"
        history_curves[curve_key] = {}

        # Multi-seed evaluations
        seed_oof_metrics = []
        seed_dev_metrics = []

        for s_idx, seed in enumerate(seeds):
            set_seed(seed)
            pids_tr = df_tr["Participant_ID"].to_numpy()
            y_tr = df_tr["target"].to_numpy()
            labels_tr_dict = dict(zip(pids_tr, y_tr))

            pids_dev = df_dv["Participant_ID"].to_numpy()
            y_dev = df_dv["target"].to_numpy()
            labels_dev_dict = dict(zip(pids_dev, y_dev))

            if task == "classification":
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                splits = list(skf.split(pids_tr, y_tr.astype(int)))
            else:
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                splits = list(kf.split(pids_tr, y_tr))

            oof_preds = np.zeros(len(pids_tr), dtype=np.float32)
            fold_models = []

            # Create Dev loader
            dev_ds = ParticipantSequenceDataset(pids_dev.tolist(), chunk_dict, labels_dev_dict)
            dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate_variable_sequences)

            for fold, (tr_idx, val_idx) in enumerate(splits):
                fold_train_pids = pids_tr[tr_idx].tolist()
                fold_val_pids = pids_tr[val_idx].tolist()

                train_ds = ParticipantSequenceDataset(fold_train_pids, chunk_dict, labels_tr_dict)
                val_ds = ParticipantSequenceDataset(fold_val_pids, chunk_dict, labels_tr_dict)

                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_variable_sequences)
                val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_variable_sequences)

                set_seed(seed + fold * 10)
                if task == "classification":
                    model = AttentionClassifier(
                        embed_dim=768, hidden_dim=best_cfg["hidden_dim"], dropout=best_cfg["dropout"]
                    ).to(device)
                    y_tr_fold = y_tr[tr_idx]
                    n_pos = np.sum(y_tr_fold == 1)
                    n_neg = len(y_tr_fold) - n_pos
                    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    model = AttentionRegressor(
                        embed_dim=768, hidden_dim=best_cfg["hidden_dim"], dropout=best_cfg["dropout"]
                    ).to(device)
                    criterion = nn.MSELoss()

                optimizer = torch.optim.AdamW(model.parameters(), lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"])

                best_model, tr_loss, va_loss, best_ep = train_fold(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    max_epochs=100,
                    patience=10,
                )

                yt_val, yp_val, _, _ = evaluate_loader(best_model, val_loader, device)
                oof_preds[val_idx] = yp_val
                fold_models.append(best_model)

                # Record loss curves for seed 42
                if seed == 42:
                    history_curves[curve_key][f"Fold_{fold+1}"] = {
                        "train_loss": tr_loss,
                        "val_loss": va_loss,
                        "best_epoch": best_ep,
                    }

                # Record fold performance
                if task == "classification":
                    tau_f, f_f1 = calibrate_threshold(yt_val, yp_val)
                    f_ba = balanced_accuracy_score(yt_val, (yp_val >= tau_f).astype(int))
                    f_auc = roc_auc_score(yt_val, yp_val)
                    fold_records.append({
                        "task": task, "arm": arm, "seed": seed, "fold": fold + 1,
                        "val_f1": f_f1, "val_bal_acc": f_ba, "val_auc": f_auc, "best_epoch": best_ep
                    })
                else:
                    f_ccc = compute_ccc(yt_val, yp_val)
                    f_rmse = float(np.sqrt(mean_squared_error(yt_val, yp_val)))
                    fold_records.append({
                        "task": task, "arm": arm, "seed": seed, "fold": fold + 1,
                        "val_ccc": f_ccc, "val_rmse": f_rmse, "best_epoch": best_ep
                    })

            # Save Train OOF predictions
            if task == "classification":
                tau_star, oof_f1 = calibrate_threshold(y_tr, oof_preds)
                oof_preds_binary = (oof_preds >= tau_star).astype(int)
                oof_ba = balanced_accuracy_score(y_tr, oof_preds_binary)
                oof_auc = roc_auc_score(y_tr, oof_preds)
                df_oof = pd.DataFrame({
                    "Participant_ID": pids_tr,
                    "split": "train",
                    "task": task,
                    "arm": arm,
                    "seed": seed,
                    "y_true": y_tr.astype(int),
                    "decision_score": oof_preds,
                    "y_pred": oof_preds_binary,
                    "calibrated_tau": tau_star,
                })
            else:
                oof_ccc = compute_ccc(y_tr, oof_preds)
                oof_rmse = float(np.sqrt(mean_squared_error(y_tr, oof_preds)))
                df_oof = pd.DataFrame({
                    "Participant_ID": pids_tr,
                    "split": "train",
                    "task": task,
                    "arm": arm,
                    "seed": seed,
                    "y_true": y_tr,
                    "y_pred": oof_preds,
                })

            df_oof.to_csv(PRED_DIR / f"oof_{task[:3]}__{arm}__seed{seed}.csv", index=False)

            # Development Ensemble Evaluation across 5 folds
            dev_fold_preds = []
            dev_attn_all = []
            for fm in fold_models:
                _, d_preds, _, d_attn = evaluate_loader(fm, dev_loader, device)
                dev_fold_preds.append(d_preds)
                dev_attn_all.append(d_attn)

            # Average predictions across folds
            dev_ensemble_pred = np.mean(dev_fold_preds, axis=0)

            # Save sample attention weights from Seed 42, Fold 0 for visualization
            if seed == 42 and len(example_attention_weights) < 4:
                sample_pids_to_plot = [300, 301, 302, 303]
                for spid in sample_pids_to_plot:
                    if spid in dev_attn_all[0] and len(example_attention_weights) < 4:
                        ex_row = df_dv[df_dv["Participant_ID"] == spid].iloc[0]
                        example_attention_weights.append({
                            "participant_id": spid,
                            "phq_binary": int(df_dev_cls[df_dev_cls["Participant_ID"] == spid]["target"].iloc[0]),
                            "phq_score": float(df_dev_reg[df_dev_reg["Participant_ID"] == spid]["target"].iloc[0]),
                            "attention_weights": dev_attn_all[0][spid].tolist(),
                        })

            if task == "classification":
                dev_binary = (dev_ensemble_pred >= tau_star).astype(int)
                df_dev_out = pd.DataFrame({
                    "Participant_ID": pids_dev,
                    "split": "dev",
                    "task": task,
                    "arm": arm,
                    "seed": seed,
                    "y_true": y_dev.astype(int),
                    "decision_score": dev_ensemble_pred,
                    "y_pred": dev_binary,
                    "calibrated_tau": tau_star,
                })
            else:
                df_dev_out = pd.DataFrame({
                    "Participant_ID": pids_dev,
                    "split": "dev",
                    "task": task,
                    "arm": arm,
                    "seed": seed,
                    "y_true": y_dev,
                    "y_pred": dev_ensemble_pred,
                })

            df_dev_out.to_csv(PRED_DIR / f"dev_{task[:3]}__{arm}__seed{seed}.csv", index=False)
            if seed == 42:
                saved_dev_predictions[f"{arm}_{task}"] = df_dev_out

            # Compute Development Metrics on Full Cohort (N=56)
            if task == "classification":
                dev_f1 = f1_score(y_dev, dev_binary, pos_label=1, zero_division=0)
                dev_ba = balanced_accuracy_score(y_dev, dev_binary)
                dev_auc = roc_auc_score(y_dev, dev_ensemble_pred)
                dev_pr_auc = average_precision_score(y_dev, dev_ensemble_pred)
                dev_prec = precision_score(y_dev, dev_binary, pos_label=1, zero_division=0)
                dev_rec = recall_score(y_dev, dev_binary, pos_label=1, zero_division=0)
                dev_acc = accuracy_score(y_dev, dev_binary)
                tn, fp, fn, tp = confusion_matrix(y_dev, dev_binary).ravel()
                cm_str = f"{tn}/{fp}/{fn}/{tp}"

                ci_f1 = bootstrap_ci(lambda yt, yp: float(f1_score(yt, yp, pos_label=1, zero_division=0)), y_dev, dev_binary)
                ci_auc = bootstrap_ci(lambda yt, ys: float(roc_auc_score(yt, ys)), y_dev, dev_ensemble_pred)

                seed_records.append({
                    "task": task, "arm": arm, "seed": seed, "dev_f1": dev_f1, "dev_ba": dev_ba, "dev_auc": dev_auc
                })
                logger.info(f"  [Seed {seed}] Dev Full (N=56) -> F1={dev_f1:.4f} (95% CI [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]), BalAcc={dev_ba:.4f}, AUC={dev_auc:.4f}, CM={cm_str}")

                # Matched Cohort (N=54)
                df_matched = df_dev_out[df_dev_out["Participant_ID"].isin(matched_pids_dev)]
                yt_m = df_matched["y_true"].to_numpy()
                yp_m = df_matched["y_pred"].to_numpy()
                ys_m = df_matched["decision_score"].to_numpy()
                m_f1 = f1_score(yt_m, yp_m, pos_label=1, zero_division=0)
                m_ba = balanced_accuracy_score(yt_m, yp_m)
                m_auc = roc_auc_score(yt_m, ys_m)
                tn_m, fp_m, fn_m, tp_m = confusion_matrix(yt_m, yp_m).ravel()

                all_results.append({
                    "model_label": f"{arm} (Seed {seed})",
                    "representation": arm,
                    "task": task,
                    "seed": seed,
                    "cohort": "full",
                    "n_dev": 56,
                    "opt_hyperparams": f"h={best_cfg['hidden_dim']},drop={best_cfg['dropout']},lr={best_cfg['lr']},wd={best_cfg['weight_decay']}",
                    "train_oof_primary": oof_f1,
                    "calibrated_tau": tau_star,
                    "dev_f1": dev_f1,
                    "dev_f1_ci_low": ci_f1[0],
                    "dev_f1_ci_high": ci_f1[1],
                    "dev_balanced_accuracy": dev_ba,
                    "dev_roc_auc": dev_auc,
                    "dev_roc_auc_ci_low": ci_auc[0],
                    "dev_roc_auc_ci_high": ci_auc[1],
                    "dev_pr_auc": dev_pr_auc,
                    "dev_precision": dev_prec,
                    "dev_recall": dev_rec,
                    "dev_accuracy": dev_acc,
                    "dev_confusion_matrix": cm_str,
                    "dev_ccc": np.nan, "dev_rmse": np.nan, "dev_mae": np.nan, "dev_r2": np.nan, "dev_pearson_r": np.nan,
                })
                all_results.append({
                    "model_label": f"{arm}_matched (Seed {seed})",
                    "representation": f"{arm}_matched",
                    "task": task,
                    "seed": seed,
                    "cohort": "matched",
                    "n_dev": 54,
                    "opt_hyperparams": f"h={best_cfg['hidden_dim']},drop={best_cfg['dropout']},lr={best_cfg['lr']},wd={best_cfg['weight_decay']}",
                    "train_oof_primary": oof_f1,
                    "calibrated_tau": tau_star,
                    "dev_f1": m_f1,
                    "dev_f1_ci_low": np.nan, "dev_f1_ci_high": np.nan,
                    "dev_balanced_accuracy": m_ba,
                    "dev_roc_auc": m_auc,
                    "dev_roc_auc_ci_low": np.nan, "dev_roc_auc_ci_high": np.nan,
                    "dev_pr_auc": average_precision_score(yt_m, ys_m),
                    "dev_precision": precision_score(yt_m, yp_m, pos_label=1, zero_division=0),
                    "dev_recall": recall_score(yt_m, yp_m, pos_label=1, zero_division=0),
                    "dev_accuracy": accuracy_score(yt_m, yp_m),
                    "dev_confusion_matrix": f"{tn_m}/{fp_m}/{fn_m}/{tp_m}",
                    "dev_ccc": np.nan, "dev_rmse": np.nan, "dev_mae": np.nan, "dev_r2": np.nan, "dev_pearson_r": np.nan,
                })
            else:
                dev_ccc = compute_ccc(y_dev, dev_ensemble_pred)
                dev_rmse = float(np.sqrt(mean_squared_error(y_dev, dev_ensemble_pred)))
                dev_mae = float(mean_absolute_error(y_dev, dev_ensemble_pred))
                dev_r2 = float(r2_score(y_dev, dev_ensemble_pred))
                dev_r = float(pearsonr(y_dev, dev_ensemble_pred)[0])

                ci_ccc = bootstrap_ci(compute_ccc, y_dev, dev_ensemble_pred)
                ci_rmse = bootstrap_ci(lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))), y_dev, dev_ensemble_pred)

                seed_records.append({
                    "task": task, "arm": arm, "seed": seed, "dev_ccc": dev_ccc, "dev_rmse": dev_rmse, "dev_r2": dev_r2
                })
                logger.info(f"  [Seed {seed}] Dev Full (N=56) -> CCC={dev_ccc:.4f} (95% CI [{ci_ccc[0]:.4f}, {ci_ccc[1]:.4f}]), RMSE={dev_rmse:.4f}, R2={dev_r2:+.4f}, r={dev_r:.4f}")

                # Matched Cohort (N=54)
                df_matched = df_dev_out[df_dev_out["Participant_ID"].isin(matched_pids_dev)]
                yt_m = df_matched["y_true"].to_numpy()
                yp_m = df_matched["y_pred"].to_numpy()
                m_ccc = compute_ccc(yt_m, yp_m)
                m_rmse = float(np.sqrt(mean_squared_error(yt_m, yp_m)))
                m_mae = float(mean_absolute_error(yt_m, yp_m))
                m_r2 = float(r2_score(yt_m, yp_m))
                m_r = float(pearsonr(yt_m, yp_m)[0])

                all_results.append({
                    "model_label": f"{arm} (Seed {seed})",
                    "representation": arm,
                    "task": task,
                    "seed": seed,
                    "cohort": "full",
                    "n_dev": 56,
                    "opt_hyperparams": f"h={best_cfg['hidden_dim']},drop={best_cfg['dropout']},lr={best_cfg['lr']},wd={best_cfg['weight_decay']}",
                    "train_oof_primary": oof_ccc,
                    "calibrated_tau": np.nan,
                    "dev_f1": np.nan, "dev_f1_ci_low": np.nan, "dev_f1_ci_high": np.nan,
                    "dev_balanced_accuracy": np.nan, "dev_roc_auc": np.nan, "dev_roc_auc_ci_low": np.nan, "dev_roc_auc_ci_high": np.nan,
                    "dev_pr_auc": np.nan, "dev_precision": np.nan, "dev_recall": np.nan, "dev_accuracy": np.nan, "dev_confusion_matrix": "",
                    "dev_ccc": dev_ccc,
                    "dev_ccc_ci_low": ci_ccc[0],
                    "dev_ccc_ci_high": ci_ccc[1],
                    "dev_rmse": dev_rmse,
                    "dev_rmse_ci_low": ci_rmse[0],
                    "dev_rmse_ci_high": ci_rmse[1],
                    "dev_mae": dev_mae,
                    "dev_r2": dev_r2,
                    "dev_pearson_r": dev_r,
                })
                all_results.append({
                    "model_label": f"{arm}_matched (Seed {seed})",
                    "representation": f"{arm}_matched",
                    "task": task,
                    "seed": seed,
                    "cohort": "matched",
                    "n_dev": 54,
                    "opt_hyperparams": f"h={best_cfg['hidden_dim']},drop={best_cfg['dropout']},lr={best_cfg['lr']},wd={best_cfg['weight_decay']}",
                    "train_oof_primary": oof_ccc,
                    "calibrated_tau": np.nan,
                    "dev_f1": np.nan, "dev_f1_ci_low": np.nan, "dev_f1_ci_high": np.nan,
                    "dev_balanced_accuracy": np.nan, "dev_roc_auc": np.nan, "dev_roc_auc_ci_low": np.nan, "dev_roc_auc_ci_high": np.nan,
                    "dev_pr_auc": np.nan, "dev_precision": np.nan, "dev_recall": np.nan, "dev_accuracy": np.nan, "dev_confusion_matrix": "",
                    "dev_ccc": m_ccc,
                    "dev_ccc_ci_low": np.nan, "dev_ccc_ci_high": np.nan,
                    "dev_rmse": m_rmse,
                    "dev_rmse_ci_low": np.nan, "dev_rmse_ci_high": np.nan,
                    "dev_mae": m_mae,
                    "dev_r2": m_r2,
                    "dev_pearson_r": m_r,
                })

    # Add historical Phase 18B baselines to results table for comprehensive benchmarking
    historical_baselines = [
        {"model_label": "whole__median LinearSVC (18B)", "representation": "whole__median", "task": "classification", "seed": 42, "cohort": "full", "n_dev": 56, "opt_hyperparams": "C=0.001, None", "train_oof_primary": 0.5227, "calibrated_tau": 0.0531, "dev_f1": 0.4516, "dev_f1_ci_low": 0.2069, "dev_f1_ci_high": 0.6667, "dev_balanced_accuracy": 0.6553, "dev_roc_auc": 0.6591, "dev_roc_auc_ci_low": 0.477, "dev_roc_auc_ci_high": 0.814, "dev_pr_auc": 0.3831, "dev_precision": 0.3684, "dev_recall": 0.5833, "dev_accuracy": 0.6964, "dev_confusion_matrix": "32/12/5/7", "dev_ccc": np.nan, "dev_rmse": np.nan, "dev_mae": np.nan, "dev_r2": np.nan, "dev_pearson_r": np.nan},
        {"model_label": "middle__mean LinearSVC (18B)", "representation": "middle__mean", "task": "classification", "seed": 42, "cohort": "full", "n_dev": 56, "opt_hyperparams": "C=0.001, balanced", "train_oof_primary": 0.5397, "calibrated_tau": 0.3756, "dev_f1": 0.3846, "dev_f1_ci_low": 0.1111, "dev_f1_ci_high": 0.6087, "dev_balanced_accuracy": 0.6061, "dev_roc_auc": 0.5606, "dev_roc_auc_ci_low": 0.377, "dev_roc_auc_ci_high": 0.727, "dev_pr_auc": 0.3015, "dev_precision": 0.3571, "dev_recall": 0.4167, "dev_accuracy": 0.7143, "dev_confusion_matrix": "35/9/7/5", "dev_ccc": np.nan, "dev_rmse": np.nan, "dev_mae": np.nan, "dev_r2": np.nan, "dev_pearson_r": np.nan},
        {"model_label": "whole__mean_std Ridge (18B)", "representation": "whole__mean_std", "task": "regression", "seed": 42, "cohort": "full", "n_dev": 56, "opt_hyperparams": "alpha=0.0001", "train_oof_primary": 0.4055, "calibrated_tau": np.nan, "dev_f1": np.nan, "dev_f1_ci_low": np.nan, "dev_f1_ci_high": np.nan, "dev_balanced_accuracy": np.nan, "dev_roc_auc": np.nan, "dev_roc_auc_ci_low": np.nan, "dev_roc_auc_ci_high": np.nan, "dev_pr_auc": np.nan, "dev_precision": np.nan, "dev_recall": np.nan, "dev_accuracy": np.nan, "dev_confusion_matrix": "", "dev_ccc": 0.4048, "dev_ccc_ci_low": 0.1704, "dev_ccc_ci_high": 0.5841, "dev_rmse": 5.3423, "dev_rmse_ci_low": 4.380, "dev_rmse_ci_high": 6.168, "dev_mae": 4.3012, "dev_r2": 0.1099, "dev_pearson_r": 0.4307},
        {"model_label": "middle__mean Ridge (18B)", "representation": "middle__mean", "task": "regression", "seed": 42, "cohort": "full", "n_dev": 56, "opt_hyperparams": "alpha=100.0", "train_oof_primary": 0.3702, "calibrated_tau": np.nan, "dev_f1": np.nan, "dev_f1_ci_low": np.nan, "dev_f1_ci_high": np.nan, "dev_balanced_accuracy": np.nan, "dev_roc_auc": np.nan, "dev_roc_auc_ci_low": np.nan, "dev_roc_auc_ci_high": np.nan, "dev_pr_auc": np.nan, "dev_precision": np.nan, "dev_recall": np.nan, "dev_accuracy": np.nan, "dev_confusion_matrix": "", "dev_ccc": 0.4022, "dev_ccc_ci_low": 0.1834, "dev_ccc_ci_high": 0.5818, "dev_rmse": 5.1890, "dev_rmse_ci_low": 4.159, "dev_rmse_ci_high": 6.079, "dev_mae": 4.2251, "dev_r2": 0.1602, "dev_pearson_r": 0.4685},
    ]
    all_results.extend(historical_baselines)

    df_master = pd.DataFrame(all_results)
    df_master.to_csv(REPORTS_DIR / "model_comparison_table.csv", index=False)
    logger.info(f"Saved master model comparison table to: {REPORTS_DIR / 'model_comparison_table.csv'}")

    # Step 3: Paired Participant-Level Bootstrap Difference Tests (B=2,000)
    logger.info("\nRunning Paired Participant-Level Bootstrap Difference Tests (B=2,000, Seed 42)...")
    distilbert_preds_dir = ML_TRAINING / "outputs" / "text_distilbert_v1" / "predictions"
    paired_comparisons = [
        {
            "comparison": "Whole Attention vs Whole Median (Cls F1)",
            "task": "classification", "metric": "f1",
            "df_attn": saved_dev_predictions["whole_attention_classification"],
            "base_file": distilbert_preds_dir / "dev_cls__whole__median__linear_svc.csv",
        },
        {
            "comparison": "Whole Attention vs Whole Median (Cls ROC-AUC)",
            "task": "classification", "metric": "roc_auc",
            "df_attn": saved_dev_predictions["whole_attention_classification"],
            "base_file": distilbert_preds_dir / "dev_cls__whole__median__linear_svc.csv",
        },
        {
            "comparison": "Middle Attention vs Middle Mean (Cls F1)",
            "task": "classification", "metric": "f1",
            "df_attn": saved_dev_predictions["middle_attention_classification"],
            "base_file": distilbert_preds_dir / "dev_cls__middle__mean__linear_svc.csv",
        },
        {
            "comparison": "Middle Attention vs Middle Mean (Cls ROC-AUC)",
            "task": "classification", "metric": "roc_auc",
            "df_attn": saved_dev_predictions["middle_attention_classification"],
            "base_file": distilbert_preds_dir / "dev_cls__middle__mean__linear_svc.csv",
        },
        {
            "comparison": "Whole Attention vs Whole Mean+Std (Reg CCC)",
            "task": "regression", "metric": "ccc",
            "df_attn": saved_dev_predictions["whole_attention_regression"],
            "base_file": distilbert_preds_dir / "dev_reg__whole__mean_std__ridge.csv",
        },
        {
            "comparison": "Whole Attention vs Whole Mean+Std (Reg RMSE)",
            "task": "regression", "metric": "rmse",
            "df_attn": saved_dev_predictions["whole_attention_regression"],
            "base_file": distilbert_preds_dir / "dev_reg__whole__mean_std__ridge.csv",
        },
        {
            "comparison": "Middle Attention vs Middle Mean (Reg CCC)",
            "task": "regression", "metric": "ccc",
            "df_attn": saved_dev_predictions["middle_attention_regression"],
            "base_file": distilbert_preds_dir / "dev_reg__middle__mean__ridge.csv",
        },
        {
            "comparison": "Middle Attention vs Middle Mean (Reg RMSE)",
            "task": "regression", "metric": "rmse",
            "df_attn": saved_dev_predictions["middle_attention_regression"],
            "base_file": distilbert_preds_dir / "dev_reg__middle__mean__ridge.csv",
        },
    ]

    boot_records = []
    rng = np.random.RandomState(42)
    B = 2000

    for pc in paired_comparisons:
        comp_name = pc["comparison"]
        task = pc["task"]
        metric = pc["metric"]
        df_attn = pc["df_attn"].sort_values("Participant_ID").reset_index(drop=True)
        df_base = pd.read_csv(pc["base_file"]).sort_values("Participant_ID").reset_index(drop=True)

        assert (df_attn["Participant_ID"].values == df_base["Participant_ID"].values).all(), "Participant mismatch in paired test"
        yt = df_attn["y_true"].to_numpy()
        n_part = len(yt)

        if task == "classification":
            y_pred_attn = df_attn["y_pred"].to_numpy()
            y_score_attn = df_attn["decision_score"].to_numpy()
            y_pred_base = df_base["y_pred"].to_numpy()
            y_score_base = df_base["y_score"].to_numpy()

            if metric == "f1":
                pt_attn = f1_score(yt, y_pred_attn, pos_label=1, zero_division=0)
                pt_base = f1_score(yt, y_pred_base, pos_label=1, zero_division=0)
                diff_fn = lambda yt_s, idx: (
                    f1_score(yt_s, y_pred_attn[idx], pos_label=1, zero_division=0) -
                    f1_score(yt_s, y_pred_base[idx], pos_label=1, zero_division=0)
                )
            else:
                pt_attn = roc_auc_score(yt, y_score_attn)
                pt_base = roc_auc_score(yt, y_score_base)
                diff_fn = lambda yt_s, idx: (
                    roc_auc_score(yt_s, y_score_attn[idx]) -
                    roc_auc_score(yt_s, y_score_base[idx])
                )
        else:
            y_pred_attn = df_attn["y_pred"].to_numpy()
            y_pred_base = df_base["y_pred"].to_numpy()

            if metric == "ccc":
                pt_attn = compute_ccc(yt, y_pred_attn)
                pt_base = compute_ccc(yt, y_pred_base)
                diff_fn = lambda yt_s, idx: (
                    compute_ccc(yt_s, y_pred_attn[idx]) -
                    compute_ccc(yt_s, y_pred_base[idx])
                )
            else:
                pt_attn = float(np.sqrt(mean_squared_error(yt, y_pred_attn)))
                pt_base = float(np.sqrt(mean_squared_error(yt, y_pred_base)))
                diff_fn = lambda yt_s, idx: (
                    float(np.sqrt(mean_squared_error(yt_s, y_pred_attn[idx]))) -
                    float(np.sqrt(mean_squared_error(yt_s, y_pred_base[idx])))
                )

        pt_diff = pt_attn - pt_base

        diff_samples = []
        for _ in range(B):
            b_idx = rng.randint(0, n_part, size=n_part)
            yt_s = yt[b_idx]
            try:
                diff_val = diff_fn(yt_s, b_idx)
                if not np.isnan(diff_val) and not np.isinf(diff_val):
                    diff_samples.append(diff_val)
            except Exception:
                continue

        ci_low = float(np.percentile(diff_samples, 2.5))
        ci_high = float(np.percentile(diff_samples, 97.5))
        p_attn_superior = float(np.mean([d < 0 if metric == "rmse" else d > 0 for d in diff_samples]))

        boot_records.append({
            "comparison": comp_name,
            "task": task,
            "metric": metric,
            "n": n_part,
            "point_attn": pt_attn,
            "point_base": pt_base,
            "point_diff": pt_diff,
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "zero_crossed": (ci_low <= 0.0 <= ci_high),
            "p_attention_superior": p_attn_superior,
        })
        logger.info(f"  {comp_name:<45} | Attn={pt_attn:.4f}, Base={pt_base:.4f} | Diff={pt_diff:+.4f} (95% CI [{ci_low:+.4f}, {ci_high:+.4f}]) | Superiority={p_attn_superior:.1%}")

    df_boot = pd.DataFrame(boot_records)
    df_boot.to_csv(REPORTS_DIR / "paired_bootstrap_differences.csv", index=False)
    logger.info(f"Saved paired bootstrap differences to: {REPORTS_DIR / 'paired_bootstrap_differences.csv'}")

    # Step 4: Generate Diagnostic Visualizations
    logger.info("\nGenerating diagnostic plots in plots/...")
    plot_classification_benchmark(df_master, PLOTS_DIR / "classification_benchmark_attention.png")
    plot_regression_benchmark(df_master, PLOTS_DIR / "regression_benchmark_attention.png")
    plot_cv_fold_variability(fold_records, PLOTS_DIR / "cv_fold_variability.png")
    plot_seed_stability(seed_records, PLOTS_DIR / "seed_stability_attention.png")
    plot_paired_bootstrap_differences(df_boot, PLOTS_DIR / "paired_bootstrap_differences_attention.png")
    plot_training_loss_curves(history_curves, PLOTS_DIR / "training_loss_curves.png")
    plot_attention_weight_profiles(example_attention_weights, PLOTS_DIR / "attention_weight_profiles.png")
    logger.info("All 7 diagnostic visualizations generated successfully!")

    logger.info("\n================================================================================")
    logger.info("PHASE 18C EXPERIMENTS COMPLETE AND VERIFIED")
    logger.info("================================================================================")


if __name__ == "__main__":
    execute_phase18c()
