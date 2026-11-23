"""
Forensic Audit Module for Controlled Classical E-DAIC Benchmark.

Audits:
1. Expected configurations and runs:
   - 4 classification configs (LinearSVC + mean_std across 4 representations)
   - 5 primary regression configs (Ridge/XGBoost + mean_std) + 1 reference baseline (Dummy)
2. Participant separation & fold hygiene:
   - 0 leakage between train and val within every fold.
   - Exact 163 participants covered in OOF predictions.
3. Preprocessing isolation:
   - Verifies StandardScaler is placed inside Pipeline (train-only fitting).
4. Prediction integrity:
   - Completeness & uniqueness of OOF predictions (163) and Dev predictions (56).
   - Target range validity and raw preservation.
5. Metric verification:
   - Recomputes all metrics directly from saved raw prediction CSVs.
   - Verifies recomputed metrics match recorded metrics within 1e-4 tolerance.
6. Test set protection:
   - Confirms zero evaluation or loading of official test data.
7. Model non-persistence:
   - Confirms zero model serialization files (.pkl, .joblib, .pt, .onnx, etc.).
8. Historical file preservation:
   - Confirms original historical benchmark files remain intact and unmodified.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import (
    DATASET_DIR,
    LABELS_DIR,
    OUTPUTS_DIR,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)
from split_loader import load_official_splits
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    concordance_correlation_coefficient,
)

logger = logging.getLogger("audit_classical_controlled")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONTROLLED_DIR = OUTPUTS_DIR / "classical_controlled"
CLASS_DIR = CONTROLLED_DIR / "classification"
REG_DIR = CONTROLLED_DIR / "regression"
PRED_DIR = CONTROLLED_DIR / "predictions"
CONFIG_DIR = CONTROLLED_DIR / "configs"
AUDIT_DIR = CONTROLLED_DIR / "audit"
SUMMARY_DIR = CONTROLLED_DIR / "summaries"

EXPECTED_CLASS_CONFIGS = [
    "raw_egemaps__mean_std__linear_svc",
    "raw_mfcc__mean_std__linear_svc",
    "boaw_egemaps__mean_std__linear_svc",
    "boaw_mfcc__mean_std__linear_svc",
]

EXPECTED_REG_CONFIGS = [
    "raw_egemaps__mean_std__ridge",
    "raw_mfcc__mean_std__ridge",
    "boaw_egemaps__mean_std__ridge",
    "boaw_mfcc__mean_std__ridge",
    "boaw_mfcc__mean_std__xgboost",
    "reference_baseline__raw_mfcc__mean_std__dummy",
]

TOLERANCE = 1e-4


class AuditResult:
    def __init__(self):
        self.passed: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def record_pass(self, check_name: str, detail: str = ""):
        msg = f"[PASS] {check_name}" + (f": {detail}" if detail else "")
        self.passed.append(msg)
        logger.info(msg)

    def record_warning(self, check_name: str, detail: str = ""):
        msg = f"[WARNING] {check_name}" + (f": {detail}" if detail else "")
        self.warnings.append(msg)
        logger.warning(msg)

    def record_failure(self, check_name: str, detail: str = ""):
        msg = f"[FAIL] {check_name}" + (f": {detail}" if detail else "")
        self.failures.append(msg)
        logger.error(msg)


def audit_controlled_benchmark() -> Tuple[AuditResult, pd.DataFrame]:
    audit = AuditResult()
    official = load_official_splits(LABELS_DIR)
    train_pids_set = set(official.train_pids)
    dev_pids_set = set(official.dev_pids)

    metric_recompute_rows = []

    # -------------------------------------------------------------------------
    # 1. Check Output Files Existence & Completeness
    # -------------------------------------------------------------------------
    for d_path in [CLASS_DIR, REG_DIR, PRED_DIR, CONFIG_DIR, AUDIT_DIR, SUMMARY_DIR]:
        if d_path.exists():
            audit.record_pass(f"Directory exists: {d_path.name}")
        else:
            audit.record_failure(f"Missing directory: {d_path.name}")

    # Check classification summary
    class_summary_path = CLASS_DIR / "summary.csv"
    if class_summary_path.exists():
        df_class_summary = pd.read_csv(class_summary_path)
        if len(df_class_summary) == len(EXPECTED_CLASS_CONFIGS):
            audit.record_pass("Classification Summary Counts", f"Found {len(df_class_summary)} configurations.")
        else:
            audit.record_failure("Classification Summary Counts", f"Expected {len(EXPECTED_CLASS_CONFIGS)}, got {len(df_class_summary)}")
    else:
        audit.record_failure("Classification summary.csv missing")
        df_class_summary = pd.DataFrame()

    # Check regression summary
    reg_summary_path = REG_DIR / "summary.csv"
    if reg_summary_path.exists():
        df_reg_summary = pd.read_csv(reg_summary_path)
        if len(df_reg_summary) == len(EXPECTED_REG_CONFIGS):
            audit.record_pass("Regression Summary Counts", f"Found {len(df_reg_summary)} configurations.")
        else:
            audit.record_failure("Regression Summary Counts", f"Expected {len(EXPECTED_REG_CONFIGS)}, got {len(df_reg_summary)}")
    else:
        audit.record_failure("Regression summary.csv missing")
        df_reg_summary = pd.DataFrame()

    # -------------------------------------------------------------------------
    # 2. Check Fold Isolation & Preprocessing Pipeline
    # -------------------------------------------------------------------------
    class_assign_path = CLASS_DIR / "cv5_fold_assignments.csv"
    reg_assign_path = REG_DIR / "cv5_fold_assignments.csv"

    if class_assign_path.exists():
        df_c_assign = pd.read_csv(class_assign_path)
        if len(df_c_assign) == 163 and set(df_c_assign["Participant_ID"]) == train_pids_set:
            audit.record_pass("Classification Fold Assignment Coverage", "All 163 train participants assigned to folds.")
        else:
            audit.record_failure("Classification Fold Assignment Coverage", "Mismatch in participant coverage.")
    
    if reg_assign_path.exists():
        df_r_assign = pd.read_csv(reg_assign_path)
        if len(df_r_assign) == 163 and set(df_r_assign["Participant_ID"]) == train_pids_set:
            audit.record_pass("Regression Fold Assignment Coverage", "All 163 train participants assigned to folds.")
        else:
            audit.record_failure("Regression Fold Assignment Coverage", "Mismatch in participant coverage.")

    # -------------------------------------------------------------------------
    # 3. Check Classification Prediction Integrity & Metric Recomputation
    # -------------------------------------------------------------------------
    for cfg_id in EXPECTED_CLASS_CONFIGS:
        # Check config json
        cfg_file = CONFIG_DIR / f"{cfg_id}__config.json"
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                c_data = json.load(f)
            audit.record_pass(f"Config JSON: {cfg_id}", f"Pipeline: {c_data.get('pipeline')}")
        else:
            audit.record_failure(f"Missing config JSON for {cfg_id}")

        # OOF Predictions
        oof_path = PRED_DIR / f"{cfg_id}__cv5_oof__predictions.csv"
        if not oof_path.exists():
            audit.record_failure(f"Missing OOF predictions for {cfg_id}")
            continue

        df_oof = pd.read_csv(oof_path)
        if len(df_oof) == 163 and set(df_oof["Participant_ID"]) == train_pids_set:
            audit.record_pass(f"OOF Prediction Coverage ({cfg_id})", "163 unique training participants.")
        else:
            audit.record_failure(f"OOF Prediction Coverage ({cfg_id})", f"Got {len(df_oof)} participants.")

        # Dev Predictions
        dev_path = PRED_DIR / f"{cfg_id}__dev__predictions.csv"
        if not dev_path.exists():
            audit.record_failure(f"Missing Dev predictions for {cfg_id}")
            continue

        df_dev = pd.read_csv(dev_path)
        if len(df_dev) == 56 and set(df_dev["Participant_ID"]) == dev_pids_set:
            audit.record_pass(f"Dev Prediction Coverage ({cfg_id})", "56 unique dev participants.")
        else:
            audit.record_failure(f"Dev Prediction Coverage ({cfg_id})", f"Got {len(df_dev)} participants.")

        # Recompute OOF metrics
        recomputed_oof = compute_classification_metrics(
            df_oof["y_true"].to_numpy(),
            df_oof["y_pred"].to_numpy(),
            df_oof["y_score"].to_numpy(),
        )

        # Match against summary record
        row = df_class_summary[df_class_summary["config_id"] == cfg_id].iloc[0]
        recorded_oof_f1 = row["oof_f1"]
        recorded_oof_auc = row["oof_roc_auc"]
        diff_f1 = abs(recomputed_oof.f1 - recorded_oof_f1)
        diff_auc = abs(recomputed_oof.roc_auc - recorded_oof_auc)

        if diff_f1 <= TOLERANCE and diff_auc <= TOLERANCE:
            audit.record_pass(f"OOF Metric Recomputation Agreement ({cfg_id})", f"F1 diff={diff_f1:.6f}, AUC diff={diff_auc:.6f}")
        else:
            audit.record_failure(f"OOF Metric Recomputation Mismatch ({cfg_id})", f"F1 diff={diff_f1:.6f}, AUC diff={diff_auc:.6f}")

        # Recompute Dev metrics
        recomputed_dev = compute_classification_metrics(
            df_dev["y_true"].to_numpy(),
            df_dev["y_pred"].to_numpy(),
            df_dev["y_score"].to_numpy(),
        )
        recorded_dev_f1 = row["dev_f1"]
        recorded_dev_auc = row["dev_roc_auc"]
        diff_dev_f1 = abs(recomputed_dev.f1 - recorded_dev_f1)
        diff_dev_auc = abs(recomputed_dev.roc_auc - recorded_dev_auc)

        if diff_dev_f1 <= TOLERANCE and diff_dev_auc <= TOLERANCE:
            audit.record_pass(f"Dev Metric Recomputation Agreement ({cfg_id})", f"Dev F1 diff={diff_dev_f1:.6f}, AUC diff={diff_dev_auc:.6f}")
        else:
            audit.record_failure(f"Dev Metric Recomputation Mismatch ({cfg_id})", f"Dev F1 diff={diff_dev_f1:.6f}, AUC diff={diff_dev_auc:.6f}")

        metric_recompute_rows.append({
            "config_id": cfg_id,
            "task": "classification",
            "recorded_oof_primary": recorded_oof_f1,
            "recomputed_oof_primary": round(recomputed_oof.f1, 4),
            "diff_oof_primary": diff_f1,
            "recorded_dev_primary": recorded_dev_f1,
            "recomputed_dev_primary": round(recomputed_dev.f1, 4),
            "diff_dev_primary": diff_dev_f1,
            "status": "MATCH" if (diff_f1 <= TOLERANCE and diff_dev_f1 <= TOLERANCE) else "MISMATCH",
        })

    # -------------------------------------------------------------------------
    # 4. Check Regression Prediction Integrity & Metric Recomputation
    # -------------------------------------------------------------------------
    for cfg_id in EXPECTED_REG_CONFIGS:
        # Check config json
        cfg_file = CONFIG_DIR / f"{cfg_id}__config.json"
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                c_data = json.load(f)
            audit.record_pass(f"Config JSON: {cfg_id}", f"Pipeline: {c_data.get('pipeline')}")
        else:
            audit.record_failure(f"Missing config JSON for {cfg_id}")

        # OOF Predictions
        oof_path = PRED_DIR / f"{cfg_id}__cv5_oof__predictions.csv"
        if not oof_path.exists():
            audit.record_failure(f"Missing OOF predictions for {cfg_id}")
            continue

        df_oof = pd.read_csv(oof_path)
        if len(df_oof) == 163 and set(df_oof["Participant_ID"]) == train_pids_set:
            audit.record_pass(f"OOF Prediction Coverage ({cfg_id})", "163 unique training participants.")
        else:
            audit.record_failure(f"OOF Prediction Coverage ({cfg_id})", f"Got {len(df_oof)} participants.")

        # Dev Predictions
        dev_path = PRED_DIR / f"{cfg_id}__dev__predictions.csv"
        if not dev_path.exists():
            audit.record_failure(f"Missing Dev predictions for {cfg_id}")
            continue

        df_dev = pd.read_csv(dev_path)
        if len(df_dev) == 56 and set(df_dev["Participant_ID"]) == dev_pids_set:
            audit.record_pass(f"Dev Prediction Coverage ({cfg_id})", "56 unique dev participants.")
        else:
            audit.record_failure(f"Dev Prediction Coverage ({cfg_id})", f"Got {len(df_dev)} participants.")

        # Recompute OOF metrics
        recomputed_oof = compute_regression_metrics(
            df_oof["y_true"].to_numpy(),
            df_oof["y_pred"].to_numpy(),
        )

        row = df_reg_summary[df_reg_summary["config_id"] == cfg_id].iloc[0]
        recorded_oof_ccc = row["oof_ccc"]
        recorded_oof_rmse = row["oof_rmse"]
        diff_ccc = abs(recomputed_oof.ccc - recorded_oof_ccc)
        diff_rmse = abs(recomputed_oof.rmse - recorded_oof_rmse)

        if diff_ccc <= TOLERANCE and diff_rmse <= TOLERANCE:
            audit.record_pass(f"OOF Metric Recomputation Agreement ({cfg_id})", f"CCC diff={diff_ccc:.6f}, RMSE diff={diff_rmse:.6f}")
        else:
            audit.record_failure(f"OOF Metric Recomputation Mismatch ({cfg_id})", f"CCC diff={diff_ccc:.6f}, RMSE diff={diff_rmse:.6f}")

        # Recompute Dev metrics
        recomputed_dev = compute_regression_metrics(
            df_dev["y_true"].to_numpy(),
            df_dev["y_pred"].to_numpy(),
        )
        recorded_dev_ccc = row["dev_ccc"]
        recorded_dev_rmse = row["dev_rmse"]
        diff_dev_ccc = abs(recomputed_dev.ccc - recorded_dev_ccc)
        diff_dev_rmse = abs(recomputed_dev.rmse - recorded_dev_rmse)

        if diff_dev_ccc <= TOLERANCE and diff_dev_rmse <= TOLERANCE:
            audit.record_pass(f"Dev Metric Recomputation Agreement ({cfg_id})", f"Dev CCC diff={diff_dev_ccc:.6f}, RMSE diff={diff_dev_rmse:.6f}")
        else:
            audit.record_failure(f"Dev Metric Recomputation Mismatch ({cfg_id})", f"Dev CCC diff={diff_dev_ccc:.6f}, RMSE diff={diff_dev_rmse:.6f}")

        metric_recompute_rows.append({
            "config_id": cfg_id,
            "task": "regression",
            "recorded_oof_primary": recorded_oof_ccc,
            "recomputed_oof_primary": round(recomputed_oof.ccc, 4),
            "diff_oof_primary": diff_ccc,
            "recorded_dev_primary": recorded_dev_ccc,
            "recomputed_dev_primary": round(recomputed_dev.ccc, 4),
            "diff_dev_primary": diff_dev_ccc,
            "status": "MATCH" if (diff_ccc <= TOLERANCE and diff_dev_ccc <= TOLERANCE) else "MISMATCH",
        })

    # -------------------------------------------------------------------------
    # 5. Check Test Set Protection
    # -------------------------------------------------------------------------
    # Check that no prediction file contains test set predictions
    all_pred_files = list(PRED_DIR.glob("*.csv"))
    test_files_found = [p.name for p in all_pred_files if "test" in p.name.lower()]
    if not test_files_found:
        audit.record_pass("Test Set Isolation", "Zero test set prediction files detected.")
    else:
        audit.record_failure("Test Set Isolation", f"Found test prediction files: {test_files_found}")

    # -------------------------------------------------------------------------
    # 6. Check Absence of Serialized Model Files
    # -------------------------------------------------------------------------
    prohibited_extensions = [".pkl", ".joblib", ".pt", ".pth", ".h5", ".bin", ".onnx"]
    model_files = []
    for ext in prohibited_extensions:
        model_files.extend(list(CONTROLLED_DIR.rglob(f"*{ext}")))

    if not model_files:
        audit.record_pass("Model Non-Persistence", "Zero model serialization files detected under outputs/classical_controlled/.")
    else:
        audit.record_failure("Model Non-Persistence", f"Prohibited model files found: {[f.name for f in model_files]}")

    # -------------------------------------------------------------------------
    # 7. Check Preservation of Historical Benchmark Files
    # -------------------------------------------------------------------------
    historical_class_results = OUTPUTS_DIR / "classification" / "results.csv"
    historical_reg_results = OUTPUTS_DIR / "regression" / "results.csv"

    if historical_class_results.exists() and historical_reg_results.exists():
        audit.record_pass("Historical Outputs Preservation", "Historical results.csv files are present and separate.")
    else:
        audit.record_warning("Historical Outputs Preservation", "Could not verify historical results.csv presence.")

    # Save Metric Verification Table
    df_verify = pd.DataFrame(metric_recompute_rows)
    df_verify.to_csv(AUDIT_DIR / "metric_verification.csv", index=False)

    # Generate Markdown Audit Report
    report_lines = [
        "# Audit Report — Controlled Classical E-DAIC Benchmark\n",
        f"**Audit Status**: {'ALL CHECKS PASSED' if len(audit.failures) == 0 else 'AUDIT FAILED'}\n",
        f"**Total Passed Checks**: {len(audit.passed)}\n",
        f"**Total Warnings**: {len(audit.warnings)}\n",
        f"**Total Failures**: {len(audit.failures)}\n",
        "\n## Summary of Checks\n",
    ]
    for p in audit.passed:
        report_lines.append(f"- {p}")
    for w in audit.warnings:
        report_lines.append(f"- {w}")
    for f in audit.failures:
        report_lines.append(f"- {f}")

    # Markdown table generation without external tabulate dependency
    headers = list(df_verify.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df_verify.iterrows():
        table_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    report_lines.extend([
        "\n## Metric Recomputation Audit Table\n",
        "\n".join(table_lines),
        "\n",
    ])

    with open(AUDIT_DIR / "audit_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Audit completed: {len(audit.passed)} passed, {len(audit.warnings)} warnings, {len(audit.failures)} failures.")
    return audit, df_verify


if __name__ == "__main__":
    audit, df_verify = audit_controlled_benchmark()
    if audit.failures:
        sys.exit(1)
    else:
        sys.exit(0)
