"""
Phase 15 - Script 01: Build Artifact Inventory, Model Registry, and Data Coverage Report.

Performs a forensic scan of all datasets, predictions, models, configs, histories, and metrics
across all benchmark phases in the BimodalPsycheAI repository.

Outputs:
- outputs/model_visualizations_v1/artifact_inventory.csv
- outputs/model_visualizations_v1/model_registry.csv
- outputs/model_visualizations_v1/data_coverage_report.csv
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Project paths
PROJECT_ROOT = Path("E:/Projects/BimodalPsycheAI")
ML_TRAINING = PROJECT_ROOT / "ml-training"
OUTPUTS_DIR = ML_TRAINING / "outputs"
OUTPUT_DIR = OUTPUTS_DIR / "model_visualizations_v1"
DATASET_DIR = ML_TRAINING / "dataset"
EDAIC_WOZ = ML_TRAINING / "edaic-woz"


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Lin's Concordance Correlation Coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return np.nan
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true, ddof=0)
    var_pred = np.var(y_pred, ddof=0)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-12:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return float(np.clip((2.0 * cov) / denom, -1.0, 1.0))


def scan_artifacts() -> pd.DataFrame:
    """Inventories every artifact in the repository across datasets and outputs."""
    print("--- Scanning Repository Artifacts ---")
    records = []

    scan_dirs = [
        ("outputs", OUTPUTS_DIR),
        ("dataset", DATASET_DIR),
        ("labels", EDAIC_WOZ / "labels"),
    ]

    for category, base_path in scan_dirs:
        if not base_path.exists():
            continue
        for p in base_path.rglob("*"):
            if p.is_file():
                # Relative path
                try:
                    rel_p = str(p.relative_to(PROJECT_ROOT))
                except ValueError:
                    rel_p = str(p)

                stat = p.stat()
                ext = p.suffix.lower()
                
                # Determine phase/source
                source_phase = "unknown"
                if "classical_final_test" in rel_p:
                    source_phase = "Final Test (Test Benchmark)"
                elif "classical_controlled" in rel_p:
                    source_phase = "Phase Controlled Classical"
                elif "cnn_embedding_temporal" in rel_p:
                    source_phase = "Phase 13B (Temporal GRU+Attn)"
                elif "cnn_embedding_neural_head" in rel_p:
                    source_phase = "Phase 13A (MLP Neural Heads)"
                elif "cnn_embedding_classical" in rel_p:
                    source_phase = "Phase 12 (CNN Classical Baselines)"
                elif "deep_features_v1" in rel_p:
                    source_phase = "Stage 0 (Deep CNN Features)"
                elif "classification" in rel_p:
                    source_phase = "Classical Acoustic Benchmark (Classification)"
                elif "regression" in rel_p:
                    source_phase = "Classical Acoustic Benchmark (Regression)"
                elif "dataset" in rel_p:
                    source_phase = "Acoustic Pooled Datasets"
                elif "labels" in rel_p:
                    source_phase = "Official E-DAIC Labels & Splits"

                # Artifact type
                art_type = "other"
                if ext == ".csv":
                    if "prediction" in p.name.lower() or "_pred" in p.name.lower() or "_oof" in p.name.lower():
                        art_type = "predictions"
                    elif "summary" in p.name.lower() or "metric" in p.name.lower() or "result" in p.name.lower():
                        art_type = "metrics_summary"
                    elif "history" in p.name.lower():
                        art_type = "training_history"
                    elif "split" in p.name.lower() or "label" in p.name.lower():
                        art_type = "labels_split"
                    else:
                        art_type = "feature_dataset"
                elif ext == ".json":
                    if "config" in p.name.lower():
                        art_type = "model_config"
                    elif "history" in p.name.lower():
                        art_type = "training_history"
                    elif "manifest" in p.name.lower():
                        art_type = "manifest"
                    else:
                        art_type = "metadata_json"
                elif ext == ".pt":
                    art_type = "model_checkpoint"
                elif ext == ".png":
                    art_type = "figure_png"
                elif ext == ".log":
                    art_type = "execution_log"
                elif ext == ".md":
                    art_type = "documentation_report"

                records.append({
                    "artifact_name": p.name,
                    "relative_path": rel_p.replace("\\", "/"),
                    "file_extension": ext,
                    "artifact_type": art_type,
                    "source_phase": source_phase,
                    "file_size_bytes": stat.st_size,
                    "file_size_kb": round(stat.st_size / 1024, 2),
                    "last_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                })

    df = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "artifact_inventory.csv"
    df.to_csv(out_csv, index=False)
    print(f"Artifact inventory complete: {len(df)} files cataloged -> {out_csv}")
    return df


def scan_datasets() -> pd.DataFrame:
    """Performs deep data quality and coverage audit on all feature datasets."""
    print("--- Auditing Feature Datasets & Splits ---")
    records = []

    # Load official split IDs
    train_split_csv = EDAIC_WOZ / "labels" / "train_split.csv"
    dev_split_csv = EDAIC_WOZ / "labels" / "dev_split.csv"
    test_split_csv = EDAIC_WOZ / "labels" / "test_split.csv"

    train_pids = set(pd.read_csv(train_split_csv)["Participant_ID"]) if train_split_csv.exists() else set()
    dev_pids = set(pd.read_csv(dev_split_csv)["Participant_ID"]) if dev_split_csv.exists() else set()
    test_pids = set(pd.read_csv(test_split_csv)["Participant_ID"]) if test_split_csv.exists() else set()

    dataset_files = []
    # 1. Classical acoustic datasets
    for f in DATASET_DIR.glob("*/*.csv"):
        if f.parent.name not in ["splits", "deep_features_v1", "metadata"]:
            dataset_files.append((f"acoustic__{f.parent.name}__{f.stem}", f, f.parent.name, f.stem))

    # 2. Deep CNN features
    cnn_pooled = DATASET_DIR / "deep_features_v1" / "pooled"
    if cnn_pooled.exists():
        for f in cnn_pooled.glob("*.csv"):
            parts = f.stem.split("__")
            rep = parts[0]
            pool = parts[1] if len(parts) > 1 else "unknown"
            dataset_files.append((f"cnn__{f.stem}", f, rep, pool))

    for ds_id, path, rep, pool in dataset_files:
        df = pd.read_csv(path)
        pid_col = None
        for col in ["Participant_ID", "participant_id", "ID", "id"]:
            if col in df.columns:
                pid_col = col
                break

        if pid_col is not None:
            pids = df[pid_col].dropna().astype(int).tolist()
            unique_pids = set(pids)
            n_duplicates = len(pids) - len(unique_pids)
            train_cov = len(unique_pids.intersection(train_pids))
            dev_cov = len(unique_pids.intersection(dev_pids))
            test_cov = len(unique_pids.intersection(test_pids))
        else:
            pids = []
            unique_pids = set()
            n_duplicates = 0
            train_cov, dev_cov, test_cov = 0, 0, 0

        # Numeric features
        num_cols = df.select_dtypes(include=[np.number]).columns
        feat_cols = [c for c in num_cols if c != pid_col and "phq" not in c.lower()]

        missing_count = int(df.isna().sum().sum())
        inf_count = int(np.isinf(df[feat_cols].to_numpy()).sum()) if len(feat_cols) > 0 else 0

        # Constant features (std == 0)
        stds = df[feat_cols].std(numeric_only=True)
        constant_cols = int((stds == 0).sum())

        records.append({
            "dataset_id": ds_id,
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "representation": rep,
            "pooling_strategy": pool,
            "rows": len(df),
            "columns": len(df.columns),
            "feature_dim": len(feat_cols),
            "unique_participants": len(unique_pids),
            "duplicate_participants": n_duplicates,
            "train_participants": train_cov,
            "dev_participants": dev_cov,
            "test_participants": test_cov,
            "missing_values": missing_count,
            "infinite_values": inf_count,
            "constant_columns": constant_cols,
            "file_size_kb": round(path.stat().st_size / 1024, 2),
        })

    df_cov = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "data_coverage_report.csv"
    df_cov.to_csv(out_csv, index=False)
    print(f"Data coverage audit complete: {len(df_cov)} datasets analyzed -> {out_csv}")
    return df_cov


def build_model_registry() -> pd.DataFrame:
    """Builds the authoritative model registry across all evaluation suites."""
    print("--- Building Master Model Registry ---")
    registry_records = []

    # =========================================================================
    # 1. Broad Acoustic Classification Benchmark (480 configurations)
    # =========================================================================
    class_pred_dir = OUTPUTS_DIR / "classification" / "predictions"
    if class_pred_dir.exists():
        for p_file in sorted(class_pred_dir.glob("*__predictions.csv")):
            name = p_file.stem.replace("__predictions", "")
            parts = name.split("__")
            # Expected format: {rep}__{pool}__{model}__{split}
            if len(parts) >= 4:
                rep = parts[0]
                pool = parts[1]
                model_name = parts[2]
                split = parts[3]
            else:
                continue

            config_path = OUTPUTS_DIR / "classification" / "configs" / f"{name}.json"
            model_id = f"classical_acoustic__class__{rep}__{pool}__{model_name}__{split}"

            df_p = pd.read_csv(p_file)
            y_true = df_p["y_true"].to_numpy(dtype=int)
            y_pred = df_p["y_pred"].to_numpy(dtype=int)
            y_score = df_p["y_score"].to_numpy(dtype=float) if "y_score" in df_p.columns else None

            # Calculate metrics
            acc = float(accuracy_score(y_true, y_pred))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            auc = np.nan
            ap = np.nan
            has_scores = False
            if y_score is not None and len(np.unique(y_true)) > 1 and not np.all(np.isnan(y_score)):
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                    ap = float(average_precision_score(y_true, y_score))
                    has_scores = True
                except Exception:
                    pass

            registry_records.append({
                "model_id": model_id,
                "benchmark_suite": "Broad Classical Acoustic",
                "family": "Classical Acoustic",
                "architecture": model_name,
                "task": "classification",
                "target": "PHQ_Binary",
                "representation": rep,
                "pooling": pool,
                "evaluation_split": "train_oof" if split == "cv5" else split,
                "participant_count": len(df_p),
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                "average_precision": round(ap, 4) if not np.isnan(ap) else None,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "ccc": None, "rmse": None, "mae": None, "r2": None,
                "has_continuous_scores": has_scores,
                "has_training_history": False,
                "has_checkpoint": False,
                "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                "history_artifact": None,
                "checkpoint_artifact": None,
                "warnings": "none" if not np.isnan(auc) else "continuous scores constant or undefined",
            })

    # =========================================================================
    # 2. Broad Acoustic Regression Benchmark (480 configurations)
    # =========================================================================
    reg_pred_dir = OUTPUTS_DIR / "regression" / "predictions"
    if reg_pred_dir.exists():
        for p_file in sorted(reg_pred_dir.glob("*__predictions.csv")):
            name = p_file.stem.replace("__predictions", "")
            parts = name.split("__")
            if len(parts) >= 4:
                rep = parts[0]
                pool = parts[1]
                model_name = parts[2]
                split = parts[3]
            else:
                continue

            config_path = OUTPUTS_DIR / "regression" / "configs" / f"{name}__config.json"
            model_id = f"classical_acoustic__reg__{rep}__{pool}__{model_name}__{split}"

            df_p = pd.read_csv(p_file)
            y_true = df_p["y_true"].to_numpy(dtype=float)
            y_pred = df_p["y_pred"].to_numpy(dtype=float)

            ccc = compute_ccc(y_true, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))

            registry_records.append({
                "model_id": model_id,
                "benchmark_suite": "Broad Classical Acoustic",
                "family": "Classical Acoustic",
                "architecture": model_name,
                "task": "regression",
                "target": "PHQ_Score",
                "representation": rep,
                "pooling": pool,
                "evaluation_split": "train_oof" if split == "cv5" else split,
                "participant_count": len(df_p),
                "accuracy": None, "balanced_accuracy": None, "f1": None, "precision": None, "recall": None,
                "roc_auc": None, "average_precision": None, "tn": None, "fp": None, "fn": None, "tp": None,
                "ccc": round(ccc, 4) if not np.isnan(ccc) else None,
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "has_continuous_scores": True,
                "has_training_history": False,
                "has_checkpoint": False,
                "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                "history_artifact": None,
                "checkpoint_artifact": None,
                "warnings": "none" if r2 >= 0 else "negative r2 indicates variance inflation",
            })

    # =========================================================================
    # 3. Controlled Classical Benchmark (20 predictions)
    # =========================================================================
    ctrl_pred_dir = OUTPUTS_DIR / "classical_controlled" / "predictions"
    if ctrl_pred_dir.exists():
        for p_file in sorted(ctrl_pred_dir.glob("*.csv")):
            name = p_file.stem.replace("__predictions", "")
            parts = name.split("__")
            if parts[0] == "reference_baseline":
                rep = parts[1]
                pool = parts[2]
                model_name = parts[3]
                split_raw = parts[4].replace("_oof", "")
                config_path = OUTPUTS_DIR / "classical_controlled" / "configs" / f"reference_baseline__{rep}__{pool}__{model_name}__config.json"
            else:
                rep = parts[0]
                pool = parts[1]
                model_name = parts[2]
                split_raw = parts[3].replace("_oof", "")
                config_path = OUTPUTS_DIR / "classical_controlled" / "configs" / f"{rep}__{pool}__{model_name}__config.json"

            eval_split = "train_oof" if "cv5" in split_raw else split_raw
            task = "classification" if "svc" in model_name or "dummy_class" in model_name else "regression"
            if "xgboost" in model_name:
                task = "regression"

            model_id = f"controlled_classical__{task}__{rep}__{pool}__{model_name}__{eval_split}"

            df_p = pd.read_csv(p_file)
            y_true = df_p["y_true"].to_numpy()
            y_pred = df_p["y_pred"].to_numpy()

            if task == "classification":
                y_true = y_true.astype(int)
                y_pred = y_pred.astype(int)
                y_score = df_p["y_score"].to_numpy(dtype=float) if "y_score" in df_p.columns else None

                acc = float(accuracy_score(y_true, y_pred))
                bal_acc = float(balanced_accuracy_score(y_true, y_pred))
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                prec = float(precision_score(y_true, y_pred, zero_division=0))
                rec = float(recall_score(y_true, y_pred, zero_division=0))
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                auc = float(roc_auc_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan
                ap = float(average_precision_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Controlled Classical Benchmark",
                    "family": "Controlled Classical",
                    "architecture": model_name,
                    "task": "classification",
                    "target": "PHQ_Binary",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": eval_split,
                    "participant_count": len(df_p),
                    "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4), "f1": round(f1, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                    "average_precision": round(ap, 4) if not np.isnan(ap) else None,
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "ccc": None, "rmse": None, "mae": None, "r2": None,
                    "has_continuous_scores": y_score is not None,
                    "has_training_history": False,
                    "has_checkpoint": False,
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": None, "checkpoint_artifact": None, "warnings": "none",
                })
            else:
                y_true = y_true.astype(float)
                y_pred = y_pred.astype(float)
                ccc = compute_ccc(y_true, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Controlled Classical Benchmark",
                    "family": "Controlled Classical",
                    "architecture": model_name,
                    "task": "regression",
                    "target": "PHQ_Score",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": eval_split,
                    "participant_count": len(df_p),
                    "accuracy": None, "balanced_accuracy": None, "f1": None, "precision": None, "recall": None,
                    "roc_auc": None, "average_precision": None, "tn": None, "fp": None, "fn": None, "tp": None,
                    "ccc": round(ccc, 4) if not np.isnan(ccc) else None,
                    "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
                    "has_continuous_scores": True,
                    "has_training_history": False, "has_checkpoint": False,
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": None, "checkpoint_artifact": None,
                    "warnings": "none" if r2 >= 0 else "negative r2 indicates variance inflation",
                })

    # =========================================================================
    # 4. Official Final Test Benchmark (9 predictions)
    # =========================================================================
    test_pred_dir = OUTPUTS_DIR / "classical_final_test_v1" / "predictions"
    if test_pred_dir.exists():
        for p_file in sorted(test_pred_dir.rglob("*.csv")):
            task = p_file.parent.name  # classification or regression
            name = p_file.stem.replace("__test_predictions", "")
            parts = name.split("__")
            rep = parts[0]
            pool = parts[1]
            model_name = parts[2]
            config_path = OUTPUTS_DIR / "classical_final_test_v1" / "configs" / f"{rep}__{pool}__{model_name}__config.json"
            model_id = f"final_test__{task}__{rep}__{pool}__{model_name}__test"

            df_p = pd.read_csv(p_file)
            y_true = df_p["y_true"].to_numpy()
            y_pred = df_p["y_pred"].to_numpy()

            if task == "classification":
                y_true = y_true.astype(int)
                y_pred = y_pred.astype(int)
                y_score = df_p["y_score"].to_numpy(dtype=float) if "y_score" in df_p.columns else None

                acc = float(accuracy_score(y_true, y_pred))
                bal_acc = float(balanced_accuracy_score(y_true, y_pred))
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                prec = float(precision_score(y_true, y_pred, zero_division=0))
                rec = float(recall_score(y_true, y_pred, zero_division=0))
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                auc = float(roc_auc_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan
                ap = float(average_precision_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Official Final Test Benchmark",
                    "family": "Official Test",
                    "architecture": model_name,
                    "task": "classification",
                    "target": "PHQ_Binary",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": "test",
                    "participant_count": len(df_p),
                    "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4), "f1": round(f1, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                    "average_precision": round(ap, 4) if not np.isnan(ap) else None,
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "ccc": None, "rmse": None, "mae": None, "r2": None,
                    "has_continuous_scores": y_score is not None,
                    "has_training_history": False, "has_checkpoint": False,
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": None, "checkpoint_artifact": None,
                    "warnings": "official final test evaluation (preserved history)",
                })
            else:
                y_true = y_true.astype(float)
                y_pred = y_pred.astype(float)
                ccc = compute_ccc(y_true, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Official Final Test Benchmark",
                    "family": "Official Test",
                    "architecture": model_name,
                    "task": "regression",
                    "target": "PHQ_Score",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": "test",
                    "participant_count": len(df_p),
                    "accuracy": None, "balanced_accuracy": None, "f1": None, "precision": None, "recall": None,
                    "roc_auc": None, "average_precision": None, "tn": None, "fp": None, "fn": None, "tp": None,
                    "ccc": round(ccc, 4) if not np.isnan(ccc) else None,
                    "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
                    "has_continuous_scores": True,
                    "has_training_history": False, "has_checkpoint": False,
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": None, "checkpoint_artifact": None,
                    "warnings": "official final test evaluation (preserved history)",
                })

    # =========================================================================
    # 5. CNN Embedding Classical Baselines (Phase 12, 20 predictions)
    # =========================================================================
    cnn_pred_dir = OUTPUTS_DIR / "cnn_embedding_classical_v1" / "predictions"
    if cnn_pred_dir.exists():
        for p_file in sorted(cnn_pred_dir.rglob("*.csv")):
            split_folder = p_file.parent.name  # dev or train_oof
            stem = p_file.stem.replace("__dev_pred", "").replace("__oof_pred", "")
            parts = stem.split("__")
            if "reference" in stem:
                rep = parts[1]
                pool = parts[2]
                model_name = parts[3]
                task = "classification" if "class" in model_name else "regression"
            else:
                rep = parts[0]
                pool = parts[1]
                model_name = parts[2]
                task = "classification" if "linear_svc" in model_name else "regression"

            config_path = OUTPUTS_DIR / "cnn_embedding_classical_v1" / "configs" / f"{stem}__config.json"
            model_id = f"cnn_classical__{task}__{rep}__{pool}__{model_name}__{split_folder}"

            df_p = pd.read_csv(p_file)
            y_true = df_p["y_true"].to_numpy()
            y_pred = df_p["y_pred"].to_numpy()

            if task == "classification":
                y_true = y_true.astype(int)
                y_pred = y_pred.astype(int)
                y_score = df_p["y_score"].to_numpy(dtype=float) if "y_score" in df_p.columns else (
                    df_p["decision_function"].to_numpy(dtype=float) if "decision_function" in df_p.columns else None
                )

                acc = float(accuracy_score(y_true, y_pred))
                bal_acc = float(balanced_accuracy_score(y_true, y_pred))
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                prec = float(precision_score(y_true, y_pred, zero_division=0))
                rec = float(recall_score(y_true, y_pred, zero_division=0))
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                auc = float(roc_auc_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan
                ap = float(average_precision_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Phase 12 (CNN Classical Baselines)",
                    "family": "Frozen CNN Classical",
                    "architecture": model_name,
                    "task": "classification",
                    "target": "PHQ_Binary",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": split_folder,
                    "participant_count": len(df_p),
                    "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4), "f1": round(f1, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                    "average_precision": round(ap, 4) if not np.isnan(ap) else None,
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "ccc": None, "rmse": None, "mae": None, "r2": None,
                    "has_continuous_scores": y_score is not None,
                    "has_training_history": False, "has_checkpoint": False,
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": None, "checkpoint_artifact": None, "warnings": "none",
                })
            else:
                y_true = y_true.astype(float)
                y_pred = y_pred.astype(float)
                ccc = compute_ccc(y_true, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Phase 12 (CNN Classical Baselines)",
                    "family": "Frozen CNN Classical",
                    "architecture": model_name,
                    "task": "regression",
                    "target": "PHQ_Score",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": split_folder,
                    "participant_count": len(df_p),
                    "accuracy": None, "balanced_accuracy": None, "f1": None, "precision": None, "recall": None,
                    "roc_auc": None, "average_precision": None, "tn": None, "fp": None, "fn": None, "tp": None,
                    "ccc": round(ccc, 4) if not np.isnan(ccc) else None,
                    "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
                    "has_continuous_scores": True,
                    "has_training_history": False, "has_checkpoint": False,
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": None, "checkpoint_artifact": None,
                    "warnings": "none" if r2 >= 0 else "negative r2 indicates variance inflation",
                })

    # =========================================================================
    # 6. CNN Embedding Neural Prediction Heads (Phase 13A, 16 predictions)
    # =========================================================================
    mlp_pred_dir = OUTPUTS_DIR / "cnn_embedding_neural_head_v1" / "predictions"
    if mlp_pred_dir.exists():
        for p_file in sorted(mlp_pred_dir.rglob("*.csv")):
            split_folder = p_file.parent.name
            stem = p_file.stem.replace("_dev_pred", "").replace("_oof_pred", "")
            parts = stem.split("__")
            rep = parts[0]
            pool = parts[1]
            model_tag = parts[2]
            task = "classification" if "class" in model_tag else "regression"
            config_path = OUTPUTS_DIR / "cnn_embedding_neural_head_v1" / "configs" / f"{stem}__config.json"
            hist_path = OUTPUTS_DIR / "cnn_embedding_neural_head_v1" / "histories" / f"{stem}__history.csv"
            ckpt_path = OUTPUTS_DIR / "cnn_embedding_neural_head_v1" / "models" / task / f"{stem}.pt"

            model_id = f"cnn_neural_head__{task}__{rep}__{pool}__mlp_head__{split_folder}"

            df_p = pd.read_csv(p_file)
            y_true = df_p["y_true"].to_numpy()
            y_pred = df_p["y_pred"].to_numpy()

            if task == "classification":
                y_true = y_true.astype(int)
                y_pred = y_pred.astype(int)
                y_score = df_p["y_score"].to_numpy(dtype=float) if "y_score" in df_p.columns else (
                    df_p["probability"].to_numpy(dtype=float) if "probability" in df_p.columns else None
                )

                acc = float(accuracy_score(y_true, y_pred))
                bal_acc = float(balanced_accuracy_score(y_true, y_pred))
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                prec = float(precision_score(y_true, y_pred, zero_division=0))
                rec = float(recall_score(y_true, y_pred, zero_division=0))
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                auc = float(roc_auc_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan
                ap = float(average_precision_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Phase 13A (Neural Prediction Heads)",
                    "family": "Neural Prediction Head",
                    "architecture": "MLP(Hidden=32, Dr=0.50)",
                    "task": "classification",
                    "target": "PHQ_Binary",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": split_folder,
                    "participant_count": len(df_p),
                    "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4), "f1": round(f1, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                    "average_precision": round(ap, 4) if not np.isnan(ap) else None,
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "ccc": None, "rmse": None, "mae": None, "r2": None,
                    "has_continuous_scores": y_score is not None,
                    "has_training_history": hist_path.exists(),
                    "has_checkpoint": ckpt_path.exists(),
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": str(hist_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if hist_path.exists() else None,
                    "checkpoint_artifact": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if ckpt_path.exists() else None,
                    "warnings": "none",
                })
            else:
                y_true = y_true.astype(float)
                y_pred = y_pred.astype(float)
                ccc = compute_ccc(y_true, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Phase 13A (Neural Prediction Heads)",
                    "family": "Neural Prediction Head",
                    "architecture": "MLP(Hidden=32, Dr=0.50)",
                    "task": "regression",
                    "target": "PHQ_Score",
                    "representation": rep,
                    "pooling": pool,
                    "evaluation_split": split_folder,
                    "participant_count": len(df_p),
                    "accuracy": None, "balanced_accuracy": None, "f1": None, "precision": None, "recall": None,
                    "roc_auc": None, "average_precision": None, "tn": None, "fp": None, "fn": None, "tp": None,
                    "ccc": round(ccc, 4) if not np.isnan(ccc) else None,
                    "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
                    "has_continuous_scores": True,
                    "has_training_history": hist_path.exists(),
                    "has_checkpoint": ckpt_path.exists(),
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": str(hist_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if hist_path.exists() else None,
                    "checkpoint_artifact": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if ckpt_path.exists() else None,
                    "warnings": "none" if r2 >= 0 else "negative r2 indicates variance inflation",
                })

    # =========================================================================
    # 7. CNN Embedding Temporal Sequence Models (Phase 13B, 8 predictions)
    # =========================================================================
    temp_pred_dir = OUTPUTS_DIR / "cnn_embedding_temporal_v1" / "predictions"
    if temp_pred_dir.exists():
        for p_file in sorted(temp_pred_dir.rglob("*.csv")):
            split_folder = p_file.parent.name
            stem = p_file.stem.replace("__dev", "").replace("__oof", "")
            parts = stem.split("__")
            rep = parts[0]
            task = parts[1]

            config_path = OUTPUTS_DIR / "cnn_embedding_temporal_v1" / "configs" / "temporal_gru_attention_default.json"
            hist_path = OUTPUTS_DIR / "cnn_embedding_temporal_v1" / "histories" / f"{rep}__{task}__history.json"
            ckpt_path = OUTPUTS_DIR / "cnn_embedding_temporal_v1" / "models" / task / f"{rep}__gru_attention.pt"

            model_id = f"cnn_temporal__{task}__{rep}__sequential__gru_attention__{split_folder}"

            df_p = pd.read_csv(p_file)
            y_true = df_p["y_true"].to_numpy()
            y_pred = df_p["y_pred"].to_numpy()

            if task == "classification":
                # y_pred in temporal classification predictions is continuous probability
                y_true = y_true.astype(int)
                y_score = y_pred.astype(float)
                y_pred_hard = (y_score >= 0.5).astype(int)

                acc = float(accuracy_score(y_true, y_pred_hard))
                bal_acc = float(balanced_accuracy_score(y_true, y_pred_hard))
                f1 = float(f1_score(y_true, y_pred_hard, zero_division=0))
                prec = float(precision_score(y_true, y_pred_hard, zero_division=0))
                rec = float(recall_score(y_true, y_pred_hard, zero_division=0))
                cm = confusion_matrix(y_true, y_pred_hard, labels=[0, 1])
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
                ap = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Phase 13B (Temporal Sequence Models)",
                    "family": "Temporal Sequence Model",
                    "architecture": "GRU(32)+Attention",
                    "task": "classification",
                    "target": "PHQ_Binary",
                    "representation": rep,
                    "pooling": "sequential",
                    "evaluation_split": split_folder,
                    "participant_count": len(df_p),
                    "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4), "f1": round(f1, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                    "average_precision": round(ap, 4) if not np.isnan(ap) else None,
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "ccc": None, "rmse": None, "mae": None, "r2": None,
                    "has_continuous_scores": True,
                    "has_training_history": hist_path.exists(),
                    "has_checkpoint": ckpt_path.exists(),
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": str(hist_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if hist_path.exists() else None,
                    "checkpoint_artifact": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if ckpt_path.exists() else None,
                    "warnings": "none",
                })
            else:
                y_true = y_true.astype(float)
                y_pred = y_pred.astype(float)
                ccc = compute_ccc(y_true, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                registry_records.append({
                    "model_id": model_id,
                    "benchmark_suite": "Phase 13B (Temporal Sequence Models)",
                    "family": "Temporal Sequence Model",
                    "architecture": "GRU(32)+Attention",
                    "task": "regression",
                    "target": "PHQ_Score",
                    "representation": rep,
                    "pooling": "sequential",
                    "evaluation_split": split_folder,
                    "participant_count": len(df_p),
                    "accuracy": None, "balanced_accuracy": None, "f1": None, "precision": None, "recall": None,
                    "roc_auc": None, "average_precision": None, "tn": None, "fp": None, "fn": None, "tp": None,
                    "ccc": round(ccc, 4) if not np.isnan(ccc) else None,
                    "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
                    "has_continuous_scores": True,
                    "has_training_history": hist_path.exists(),
                    "has_checkpoint": ckpt_path.exists(),
                    "prediction_artifact": str(p_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "config_artifact": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if config_path.exists() else None,
                    "history_artifact": str(hist_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if hist_path.exists() else None,
                    "checkpoint_artifact": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if ckpt_path.exists() else None,
                    "warnings": "none" if r2 >= 0 else "negative r2 indicates variance inflation",
                })

    df_reg = pd.DataFrame(registry_records)
    out_csv = OUTPUT_DIR / "model_registry.csv"
    df_reg.to_csv(out_csv, index=False)
    print(f"Master model registry complete: {len(df_reg)} model entries registered -> {out_csv}")
    print(f"  Classification models: {len(df_reg[df_reg['task'] == 'classification'])}")
    print(f"  Regression models:     {len(df_reg[df_reg['task'] == 'regression'])}")
    print("  By Benchmark Suite:")
    for suite, count in df_reg["benchmark_suite"].value_counts().items():
        print(f"    - {suite}: {count}")
    return df_reg


def main():
    print("=" * 80)
    print("PHASE 15: AUDIT, ARTIFACT INVENTORY & MASTER MODEL REGISTRY")
    print("=" * 80)
    t0 = time.time()

    scan_artifacts()
    scan_datasets()
    build_model_registry()

    print(f"\nAudit completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
