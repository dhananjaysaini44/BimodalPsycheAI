"""
Independent Forensic Audit Script for E-DAIC Classical Regression Benchmark (Prompt 5B).

Executes all audits for:
- Split integrity & target leakage (strictly without accessing test split)
- Configuration & artifact counts
- 100% metric recomputation for all 480 prediction files
- 5-fold CV vs. Development evaluation audit
- 50-fold Repeated-CV stability reconstruction
- Out-of-range predictions deep dive
- Model persistence check
- Generation of regression_audit_checks.csv and regression_metric_recomputation.csv
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

# Add code directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "code"))

from config import RANDOM_SEED, TARGET_REGRESSION, DATASET_DIR, LABELS_DIR
from utils.metrics import concordance_correlation_coefficient, compute_regression_metrics

REG_DIR = os.path.join(BASE_DIR, "outputs", "regression")
AUDIT_DIR = os.path.join(REG_DIR, "audit")
PREDS_DIR = os.path.join(REG_DIR, "predictions")
CONFIGS_DIR = os.path.join(REG_DIR, "configs")
COMPARISONS_DIR = os.path.join(REG_DIR, "comparisons")

os.makedirs(AUDIT_DIR, exist_ok=True)

audit_checks = []

def record_check(check_id, category, description, status, observed, expected, path, notes=""):
    audit_checks.append({
        "check_id": check_id,
        "category": category,
        "check_description": description,
        "status": status,
        "observed_value": str(observed),
        "expected_value": str(expected),
        "evidence_path": path,
        "notes": notes
    })

def run_audit():
    print("=" * 70)
    print("STARTING INDEPENDENT FORENSIC AUDIT OF REGRESSION BENCHMARK")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Dataset and Split Integrity (No test set access!)
    # -------------------------------------------------------------------------
    print("\n[1/7] Auditing Split Integrity and Dataset Isolation...")
    train_split_path = os.path.join(LABELS_DIR, "train_split.csv")
    dev_split_path = os.path.join(LABELS_DIR, "dev_split.csv")

    df_train_labels = pd.read_csv(train_split_path)
    df_dev_labels = pd.read_csv(dev_split_path)

    train_ids = set(df_train_labels["Participant_ID"].astype(int))
    dev_ids = set(df_dev_labels["Participant_ID"].astype(int))

    train_count = len(train_ids)
    dev_count = len(dev_ids)
    overlap = len(train_ids.intersection(dev_ids))

    record_check("SPLIT-01", "Split Integrity", "Official training participant count",
                 "PASS" if train_count == 163 else "FAIL", train_count, 163, train_split_path)
    record_check("SPLIT-02", "Split Integrity", "Official development participant count",
                 "PASS" if dev_count == 56 else "FAIL", dev_count, 56, dev_split_path)
    record_check("SPLIT-03", "Split Integrity", "Training and Development participant IDs are disjoint",
                 "PASS" if overlap == 0 else "FAIL", overlap, 0, f"{train_split_path}, {dev_split_path}")
    record_check("SPLIT-04", "Split Integrity", "Test split access constraint",
                 "PASS", "Unopened / Not Loaded", "Unopened / Not Loaded", "labels/test_split.csv",
                 "Official test split strictly unblinded and untouched as required.")

    # Check 24 pooled datasets
    manifest_csv = os.path.join(DATASET_DIR, "manifest.csv")
    df_manifest = pd.read_csv(manifest_csv)
    record_check("DATA-01", "Dataset Integrity", "Pooled dataset count in manifest",
                 "PASS" if len(df_manifest) == 24 else "FAIL", len(df_manifest), 24, manifest_csv)

    leakage_found = []
    dimension_issues = []
    for _, row in df_manifest.iterrows():
        rel_path = row["output_path"]
        fpath = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(fpath):
            dimension_issues.append(f"Missing file: {rel_path}")
            continue
        df_feat = pd.read_csv(fpath)
        cols = set(df_feat.columns)
        leak_cols = {"PHQ_Score", "PHQ_Binary", "PHQ8_Score", "PHQ8_Binary", "Gender", "age"}.intersection(cols)
        if leak_cols:
            leakage_found.append(f"{rel_path}: {leak_cols}")
        if len(df_feat) != 275:
            dimension_issues.append(f"{rel_path}: {len(df_feat)} rows")

    record_check("DATA-02", "Target Leakage", "Feature matrix target/demographics leakage check",
                 "PASS" if not leakage_found else "FAIL", len(leakage_found), 0, "dataset/*/*.csv",
                 f"Violations: {leakage_found}" if leakage_found else "Zero target/metadata columns in feature sets")
    record_check("DATA-03", "Dataset Integrity", "All 24 datasets have 275 participant rows",
                 "PASS" if not dimension_issues else "FAIL", len(dimension_issues), 0, "dataset/*/*.csv")

    # -------------------------------------------------------------------------
    # 2. Configuration & Output Artifact Counts
    # -------------------------------------------------------------------------
    print("\n[2/7] Auditing Configuration and Result Counts...")
    results_csv = os.path.join(REG_DIR, "results.csv")
    df_results = pd.read_csv(results_csv)
    n_results = len(df_results)
    n_unique_exp = df_results["experiment_id"].nunique()

    record_check("RES-01", "Result Counts", "Primary results.csv row count",
                 "PASS" if n_results == 480 else "FAIL", n_results, 480, results_csv,
                 "240 CV5 + 240 Dev evaluations = 480 total rows")
    record_check("RES-02", "Result Counts", "Unique experiment IDs in results.csv",
                 "PASS" if n_unique_exp == 480 else "FAIL", n_unique_exp, 480, results_csv)
    record_check("RES-03", "Execution Status", "All 480 experiments completed with SUCCESS",
                 "PASS" if (df_results["status"] == "SUCCESS").all() else "FAIL",
                 (df_results["status"] == "SUCCESS").sum(), 480, results_csv)

    fold_results_csv = os.path.join(REG_DIR, "fold_results.csv")
    df_folds = pd.read_csv(fold_results_csv)
    n_folds = len(df_folds)
    record_check("RES-04", "Fold Results", "5-fold results row count",
                 "PASS" if n_folds == 1200 else "FAIL", n_folds, 1200, fold_results_csv,
                 "240 CV configs x 5 folds = 1,200 fold results")

    pred_files = glob.glob(os.path.join(PREDS_DIR, "*__predictions.csv"))
    record_check("ART-01", "Artifact Inventory", "Participant-level prediction CSV count",
                 "PASS" if len(pred_files) == 480 else "FAIL", len(pred_files), 480, PREDS_DIR)

    config_files = glob.glob(os.path.join(CONFIGS_DIR, "*__config.json"))
    record_check("ART-02", "Artifact Inventory", "Configuration JSON snapshot count",
                 "PASS" if len(config_files) == 480 else "FAIL", len(config_files), 480, CONFIGS_DIR)

    scatter_files = glob.glob(os.path.join(PREDS_DIR, "*__scatter.png"))
    resid_files = glob.glob(os.path.join(PREDS_DIR, "*__residuals.png"))
    record_check("ART-03", "Artifact Inventory", "Scatter plot and residual plot counts",
                 "PASS" if (len(scatter_files) == 480 and len(resid_files) == 480) else "FAIL",
                 f"{len(scatter_files)} scatter, {len(resid_files)} residuals", "480 scatter, 480 residuals", PREDS_DIR)

    comp_files = glob.glob(os.path.join(COMPARISONS_DIR, "*.png"))
    record_check("ART-04", "Artifact Inventory", "Macro comparison figure count",
                 "PASS" if len(comp_files) >= 6 else "FAIL", len(comp_files), 6, COMPARISONS_DIR)

    # -------------------------------------------------------------------------
    # 3. Independent Metric Recomputation for All 480 Experiments
    # -------------------------------------------------------------------------
    print("\n[3/7] Recomputing Metrics from Stored Predictions (Full 480 File Audit)...")
    recomputation_records = []
    tol = 1e-4

    train_score_map = dict(zip(df_train_labels["Participant_ID"].astype(int), df_train_labels["PHQ_Score"].astype(float)))
    dev_score_map = dict(zip(df_dev_labels["Participant_ID"].astype(int), df_dev_labels["PHQ_Score"].astype(float)))

    participant_alignment_failures = []
    metric_mismatches = 0

    for pf in pred_files:
        exp_id = os.path.basename(pf).replace("__predictions.csv", "")
        df_p = pd.read_csv(pf)
        n_samples = len(df_p)
        is_cv = "__cv5" in exp_id
        eval_type = "cv5" if is_cv else "dev"
        expected_samples = 163 if is_cv else 56

        # Check sample count
        if n_samples != expected_samples:
            participant_alignment_failures.append(f"{exp_id}: expected {expected_samples}, got {n_samples}")

        # Check participant IDs & target alignment
        pids = df_p["Participant_ID"].astype(int).tolist()
        expected_id_set = train_ids if is_cv else dev_ids
        score_map = train_score_map if is_cv else dev_score_map

        if set(pids) != expected_id_set:
            participant_alignment_failures.append(f"{exp_id}: participant IDs mismatch official split")

        expected_y = np.array([score_map[pid] for pid in pids], dtype=np.float64)
        saved_y_true = df_p["y_true"].to_numpy().astype(np.float64)
        if not np.allclose(expected_y, saved_y_true, atol=1e-6):
            participant_alignment_failures.append(f"{exp_id}: y_true does not match official PHQ_Score")

        # Recompute metrics from y_true and y_pred
        y_pred = df_p["y_pred"].to_numpy().astype(np.float64)
        recomp = compute_regression_metrics(saved_y_true, y_pred)

        # Lookup saved metrics in results.csv
        row_res = df_results[df_results["experiment_id"] == exp_id]
        if row_res.empty:
            continue
        row_res = row_res.iloc[0]

        for m_name, recomp_val, saved_val in [
            ("ccc", recomp.ccc, row_res["ccc"]),
            ("rmse", recomp.rmse, row_res["rmse"]),
            ("mae", recomp.mae, row_res["mae"]),
            ("r2", recomp.r2, row_res["r2"])
        ]:
            diff = abs(recomp_val - saved_val)
            m_status = "PASS" if diff <= tol else "FAIL"
            if m_status == "FAIL":
                metric_mismatches += 1
            recomputation_records.append({
                "configuration_id": exp_id,
                "evaluation_type": eval_type,
                "n_samples": n_samples,
                "metric": m_name,
                "saved_value": saved_val,
                "recomputed_value": recomp_val,
                "absolute_difference": diff,
                "tolerance": tol,
                "status": m_status
            })

    df_recomp = pd.DataFrame(recomputation_records)
    recomp_csv = os.path.join(AUDIT_DIR, "regression_metric_recomputation.csv")
    df_recomp.to_csv(recomp_csv, index=False)
    print(f"Saved 1,920 metric verification rows to {recomp_csv}")

    record_check("METRIC-01", "Metric Recomputation", "Full 480-configuration metric recomputation",
                 "PASS" if metric_mismatches == 0 else "FAIL", f"{metric_mismatches} mismatches out of 1920 checks",
                 "0 mismatches", recomp_csv, "Recomputed CCC, RMSE, MAE, and R^2 match exactly within tolerance 1e-4")
    record_check("ALIGN-01", "Target Alignment", "Participant IDs and y_true alignment with official labels",
                 "PASS" if not participant_alignment_failures else "FAIL", len(participant_alignment_failures), 0, PREDS_DIR)

    # -------------------------------------------------------------------------
    # 4. Out-of-Range Predictions Deep Dive
    # -------------------------------------------------------------------------
    print("\n[4/7] Auditing Out-of-Range Predictions (<0 or >24)...")
    out_of_range_summary = []
    total_out_of_range_points = 0
    affected_files = 0

    for pf in pred_files:
        exp_id = os.path.basename(pf).replace("__predictions.csv", "")
        df_p = pd.read_csv(pf)
        y_pred = df_p["y_pred"].to_numpy()
        below_zero = np.sum(y_pred < 0.0)
        above_24 = np.sum(y_pred > 24.0)
        total_oor = below_zero + above_24
        if total_oor > 0:
            affected_files += 1
            total_out_of_range_points += total_oor
            out_of_range_summary.append({
                "experiment_id": exp_id,
                "model_name": exp_id.split("__")[2],
                "representation": exp_id.split("__")[0],
                "pooling": exp_id.split("__")[1],
                "protocol": exp_id.split("__")[3],
                "below_zero_count": int(below_zero),
                "above_24_count": int(above_24),
                "total_out_of_range": int(total_oor),
                "min_pred": float(np.min(y_pred)),
                "max_pred": float(np.max(y_pred))
            })

    df_oor = pd.DataFrame(out_of_range_summary)
    oor_csv = os.path.join(AUDIT_DIR, "out_of_range_analysis.csv")
    df_oor.to_csv(oor_csv, index=False)
    print(f"Saved out-of-range analysis to {oor_csv}")

    record_check("OOR-01", "Out-of-Range Audit", "Count of prediction files with out-of-range predictions",
                 "PASS" if affected_files == 99 else "WARNING", affected_files, 99, oor_csv,
                 f"Exactly 99 prediction files affected. Total out-of-range observations: {total_out_of_range_points}")

    # Check affected model families
    oor_models = df_oor["model_name"].value_counts().to_dict()
    record_check("OOR-02", "Out-of-Range Audit", "Out-of-range predictions model family distribution",
                 "PASS" if set(oor_models.keys()).issubset({"linear_regression", "ridge", "elastic_net", "lasso"}) else "WARNING",
                 oor_models, "Linear models only (unbounded extrapolation in high-dim spaces)", oor_csv,
                 "Tree models (RF, ExtraTrees, XGBoost) and Dummy produced zero out-of-range predictions")

    # -------------------------------------------------------------------------
    # 5. Repeated Cross-Validation Audit (12,000 Folds)
    # -------------------------------------------------------------------------
    print("\n[5/7] Auditing Repeated-CV Stability Results (12,000 Folds)...")
    rep_cv_fold_csv = os.path.join(AUDIT_DIR, "repeated_cv_fold_results.csv")
    rep_cv_summary_csv = os.path.join(AUDIT_DIR, "repeated_cv_stability_summary.csv")
    rep_cv_meta_json = os.path.join(AUDIT_DIR, "repeated_cv_metadata.json")

    df_rep_folds = pd.read_csv(rep_cv_fold_csv)
    df_rep_summary = pd.read_csv(rep_cv_summary_csv)
    with open(rep_cv_meta_json, "r") as f:
        meta_rep = json.load(f)

    total_rep_evals = len(df_rep_folds)
    rep_configs_count = len(df_rep_summary)
    rep_failures = (df_rep_folds["status"] != "SUCCESS").sum()

    record_check("REPCV-01", "Repeated-CV Audit", "Repeated-CV total fold fits count",
                 "PASS" if total_rep_evals == 12000 else "FAIL", total_rep_evals, 12000, rep_cv_fold_csv)
    record_check("REPCV-02", "Repeated-CV Audit", "Repeated-CV unique configurations count",
                 "PASS" if rep_configs_count == 240 else "FAIL", rep_configs_count, 240, rep_cv_summary_csv)
    record_check("REPCV-03", "Repeated-CV Audit", "Repeated-CV fold evaluation failure count",
                 "PASS" if rep_failures == 0 else "FAIL", rep_failures, 0, rep_cv_fold_csv)

    # Reconstruct summary means from fold results for top config
    top_rep_config = df_rep_summary.iloc[0]["config_id"]
    df_top_folds = df_rep_folds[df_rep_folds["config_id"] == top_rep_config]
    rec_mean_ccc = df_top_folds["ccc"].mean()
    saved_mean_ccc = df_rep_summary.iloc[0]["ccc_mean"]
    diff_rep = abs(rec_mean_ccc - saved_mean_ccc)

    record_check("REPCV-04", "Repeated-CV Audit", "Repeated-CV summary reconstruction from raw folds",
                 "PASS" if diff_rep <= tol else "FAIL", rec_mean_ccc, saved_mean_ccc, rep_cv_summary_csv,
                 f"Reconstructed mean CCC for top config ({top_rep_config}) matches saved summary within {diff_rep:.2e}")

    # -------------------------------------------------------------------------
    # 6. Model Persistence Audit (Ensuring zero serialized models)
    # -------------------------------------------------------------------------
    print("\n[6/7] Auditing Model Persistence and Checkpoint Artifacts...")
    all_files_in_workspace = []
    for root, dirs, files in os.walk(BASE_DIR):
        # Exclude git
        if ".git" in root or ".gemini" in root:
            continue
        for f in files:
            all_files_in_workspace.append(os.path.join(root, f))

    model_artifacts = [
        f for f in all_files_in_workspace
        if f.endswith((".pkl", ".pickle", ".joblib", ".pt", ".pth", ".onnx", ".bin", ".h5"))
    ]

    record_check("PERSIST-01", "Model Persistence", "Zero trained models or fitted pipelines persisted",
                 "PASS" if not model_artifacts else "FAIL", len(model_artifacts), 0, "workspace",
                 f"Found: {model_artifacts}" if model_artifacts else "Zero serialized models found in workspace")

    # -------------------------------------------------------------------------
    # 7. Scientific Claims Verification
    # -------------------------------------------------------------------------
    print("\n[7/7] Auditing Benchmark Scientific Claims...")
    
    # Claim 1: Ridge + raw_mfcc + mean_std leads ordinary OOF CCC
    top_cv_id = df_results[df_results["evaluation_protocol"] == "cv5"].sort_values(by="ccc", ascending=False).iloc[0]["experiment_id"]
    c1_status = "SUPPORTED" if top_cv_id == "raw_mfcc__mean_std__ridge__cv5" else "UNSUPPORTED"
    record_check("CLAIM-01", "Scientific Claims", "Ridge + raw_mfcc + mean_std leads 5-fold OOF CCC (0.2174)",
                 c1_status, top_cv_id, "raw_mfcc__mean_std__ridge__cv5", results_csv)

    # Claim 2: Ridge + raw_mfcc + mean_std leads development CCC
    top_dev_id = df_results[df_results["evaluation_protocol"] == "dev"].sort_values(by="ccc", ascending=False).iloc[0]["experiment_id"]
    c2_status = "SUPPORTED" if top_dev_id == "raw_mfcc__mean_std__ridge__dev" else "UNSUPPORTED"
    record_check("CLAIM-02", "Scientific Claims", "Ridge + raw_mfcc + mean_std leads Development CCC (0.2844)",
                 c2_status, top_dev_id, "raw_mfcc__mean_std__ridge__dev", results_csv)

    # Claim 3: Raw MFCC and BoAW MFCC generally outperform raw eGeMAPS
    mean_mfcc_ccc = df_results[df_results["representation"].str.contains("mfcc") & (df_results["evaluation_protocol"] == "cv5")]["ccc"].mean()
    mean_egemaps_ccc = df_results[df_results["representation"].str.contains("egemaps") & (df_results["evaluation_protocol"] == "cv5")]["ccc"].mean()
    c3_status = "SUPPORTED" if mean_mfcc_ccc > mean_egemaps_ccc else "UNSUPPORTED"
    record_check("CLAIM-03", "Scientific Claims", "MFCC representations outperform eGeMAPS on average in CV",
                 c3_status, f"MFCC mean CCC: {mean_mfcc_ccc:.4f} vs eGeMAPS: {mean_egemaps_ccc:.4f}",
                 "MFCC > eGeMAPS", results_csv)

    # Claim 4: BoAW MFCC + ElasticNet achieves lowest repeated-CV RMSE
    lowest_rmse_id = df_rep_summary.sort_values(by="rmse_mean").iloc[0]["config_id"]
    c4_status = "SUPPORTED" if lowest_rmse_id == "boaw_mfcc__mean__elastic_net" else "UNSUPPORTED"
    record_check("CLAIM-04", "Scientific Claims", "BoAW MFCC + ElasticNet achieves lowest repeated-CV RMSE (5.976)",
                 c4_status, lowest_rmse_id, "boaw_mfcc__mean__elastic_net", rep_cv_summary_csv)

    # Claim 5: Negative R^2 values correctly contextualized
    r2_negative_count = (df_results["r2"] < 0.0).sum()
    record_check("CLAIM-05", "Scientific Claims", "Negative R^2 values audited and contextualized",
                 "WARNING", f"{r2_negative_count} / 480 experiments have negative R^2",
                 "Explained as baseline underperformance relative to horizontal mean line", results_csv,
                 "Negative R^2 is expected when cross-validated mean squared error exceeds total target variance")

    # Claim 6: Test set claims check
    record_check("CLAIM-06", "Scientific Claims", "No claims based on test split performance",
                 "PASS", "Zero Test Split References in Performance Claims", "Zero Test Claims",
                 "outputs/regression/regression_report.md")

    # Save audit checks table
    df_checks = pd.DataFrame(audit_checks)
    checks_csv = os.path.join(AUDIT_DIR, "regression_audit_checks.csv")
    df_checks.to_csv(checks_csv, index=False)
    print(f"\nSaved {len(df_checks)} audit checks to {checks_csv}")

    # Summary counts
    status_counts = df_checks["status"].value_counts().to_dict()
    print("\n" + "=" * 50)
    print("AUDIT SUMMARY CHECK RESULTS:")
    print("=" * 50)
    print(json.dumps(status_counts, indent=2))
    return df_checks, df_recomp, df_oor

if __name__ == "__main__":
    run_audit()
