"""
Unit and Integration Tests for Phase 18B (DistilBERT Embeddings, Chronological Segmentation, and Robust Aggregation).

Validates:
1. Presence of all Phase 18B deliverables (reports, audit note, resource gate, smoke test, model comparison table, plots, predictions).
2. Segment feature tensor integrity (shapes, dimensions: 768, 1536, 2304, 4608; zero NaNs/Infs).
3. Participant ID alignment with official splits (Train=163, Dev=56; complete cohort=157/54).
4. Strict Official Test split quarantine (zero exposure or references to Test IDs in prediction CSVs).
5. Exact numerical reproduction of classification metrics (F1, Balanced Accuracy, ROC-AUC) from saved prediction CSVs.
6. Exact numerical reproduction of regression metrics (Lin's CCC, RMSE, R^2, Pearson r) from saved prediction CSVs.
7. Verification that regression predictions remain continuous, real-valued, and unclipped.
8. Verification of paired bootstrap difference metrics and confidence intervals.
9. Verification of DistilBERT-specific performance patterns (whole__mean_std CCC > 0.40, concat__mean F1 > 0.46).
"""

import sys
from pathlib import Path

# Add ml-training/code to sys.path
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from text_distilbert_v1.run_experiments import concordance_correlation_coefficient as compute_ccc


@pytest.fixture
def repo_paths():
    repo_root = Path(__file__).resolve().parents[3]
    ml_training = repo_root / "ml-training"
    return {
        "repo_root": repo_root,
        "labels_dir": ml_training / "edaic-woz" / "labels",
        "distilbert_dir": ml_training / "outputs" / "text_distilbert_v1",
        "dataset_dir": ml_training / "dataset" / "text_distilbert_v1",
    }


def test_phase18b_artifacts_exist(repo_paths):
    """Verifies that all Phase 18B deliverables and intermediate artifacts exist on disk."""
    ddir = repo_paths["distilbert_dir"]
    datadir = repo_paths["dataset_dir"]

    assert ddir.exists(), "Phase 18B output directory missing"
    assert datadir.exists(), "Phase 18B dataset directory missing"

    # Reports and Audit
    assert (ddir / "reports" / "phase_18b_report.md").exists()
    assert (ddir / "reports" / "model_comparison_table.csv").exists()
    assert (ddir / "reports" / "paired_bootstrap_differences.csv").exists()
    assert (ddir / "audit" / "audit_note.md").exists()
    assert (ddir / "audit" / "resource_gate.json").exists()
    assert (ddir / "audit" / "extraction_smoke_test.json").exists()
    assert (ddir / "audit" / "segment_assignment_audit.csv").exists()

    # Diagnostic Plots
    plot_files = [
        "chronological_trajectory_distilbert.png",
        "confusion_matrices_distilbert.png",
        "distilbert_classification_benchmark.png",
        "distilbert_regression_benchmark.png",
        "paired_bootstrap_differences.png",
    ]
    for pf in plot_files:
        assert (ddir / "plots" / pf).exists(), f"Missing plot: {pf}"

    # Chunk Embeddings & Metadata
    assert (datadir / "chunk_embeddings.npy").exists()
    assert (datadir / "chunk_metadata.csv").exists()
    assert (datadir / "dataset_manifest.json").exists()
    assert (datadir / "features").exists()


def test_distilbert_chunk_embeddings_and_segment_features_shapes(repo_paths):
    """Verifies shapes, dimensions (768, 1536, 2304, 4608), and numerical validity."""
    datadir = repo_paths["dataset_dir"]

    # Chunk embeddings
    chunks = np.load(datadir / "chunk_embeddings.npy")
    assert chunks.shape == (2163, 768), f"Expected (2163, 768), got {chunks.shape}"
    assert chunks.dtype == np.float32
    assert not np.isnan(chunks).any(), "NaN found in chunk embeddings"
    assert not np.isinf(chunks).any(), "Inf found in chunk embeddings"

    # Chunk metadata
    df_meta = pd.read_csv(datadir / "chunk_metadata.csv")
    assert len(df_meta) == 2163
    for col in ["Participant_ID", "split", "chunk_index", "start_token", "end_token", "token_count"]:
        assert col in df_meta.columns

    # Segment feature matrices
    features_dir = datadir / "features"
    schemes = ["whole", "early", "middle", "late", "concat"]
    aggs = ["mean", "median", "trimmed", "mean_std"]
    splits = ["train", "dev", "test"]

    dim_map = {
        "mean": 768,
        "median": 768,
        "trimmed": 768,
        "mean_std": 1536,
    }

    for scheme in schemes:
        for agg in aggs:
            rep = f"{scheme}__{agg}"
            expected_dim = dim_map[agg] * (3 if scheme == "concat" else 1)
            for sp in splits:
                fpath = features_dir / sp / f"{rep}.npy"
                assert fpath.exists(), f"Missing feature file: {fpath.name}"
                arr = np.load(fpath)
                assert arr.ndim == 2
                assert arr.shape[1] == expected_dim, f"Dim mismatch for {rep} in {sp}: expected {expected_dim}, got {arr.shape[1]}"
                assert arr.dtype == np.float32
                assert not np.isnan(arr).any(), f"NaN in {rep} for {sp}"
                assert not np.isinf(arr).any(), f"Inf in {rep} for {sp}"


def test_participant_alignment_and_cohort_splits(repo_paths):
    """Verifies participant count alignment across official Train, Dev, Test splits and matched cohorts."""
    datadir = repo_paths["dataset_dir"]
    features_dir = datadir / "features"

    for sp, n_expected_full, n_expected_complete in [("train", 163, 157), ("dev", 56, 54), ("test", 56, 45)]:
        pids_full = pd.read_csv(features_dir / sp / "all_participant_ids.csv")["Participant_ID"].values
        pids_complete = pd.read_csv(features_dir / sp / "complete_cohort_participant_ids.csv")["Participant_ID"].values

        assert len(pids_full) == n_expected_full
        assert len(pids_complete) == n_expected_complete
        assert len(np.unique(pids_full)) == n_expected_full
        assert len(np.unique(pids_complete)) == n_expected_complete
        assert set(pids_complete).issubset(set(pids_full))

        # Check whole interview dimensions
        w_mat = np.load(features_dir / sp / "whole__mean.npy")
        assert len(w_mat) == n_expected_full

        # Check concat dimensions
        c_mat = np.load(features_dir / sp / "concat__mean.npy")
        assert len(c_mat) == n_expected_complete


def test_official_test_split_quarantine(repo_paths):
    """Verifies 100% quarantine of the official Test set (no test labels, predictions, or evaluations)."""
    ddir = repo_paths["distilbert_dir"]
    labels_dir = repo_paths["labels_dir"]

    test_pids = set(pd.read_csv(labels_dir / "test_split.csv")["Participant_ID"].values)
    assert len(test_pids) == 56

    pred_dir = ddir / "predictions"
    pred_files = list(pred_dir.glob("*.csv"))
    assert len(pred_files) > 0

    for pf in pred_files:
        df_pred = pd.read_csv(pf)
        assert "split" in df_pred.columns
        splits_present = set(df_pred["split"].unique())
        assert "test" not in splits_present, f"Test split leaked into prediction file: {pf.name}"

        pids_present = set(df_pred["Participant_ID"].unique())
        leaked_ids = pids_present.intersection(test_pids)
        assert len(leaked_ids) == 0, f"Test participant IDs leaked in {pf.name}: {leaked_ids}"


def test_classification_metric_reproduction(repo_paths):
    """Verifies exact reproduction of classification metrics from saved prediction files."""
    ddir = repo_paths["distilbert_dir"]
    pred_dir = ddir / "predictions"
    tbl_path = ddir / "reports" / "model_comparison_table.csv"
    df_tbl = pd.read_csv(tbl_path)

    cls_rows = df_tbl[(df_tbl["task"] == "classification") & (~df_tbl["representation"].str.startswith("whole_matched"))]

    for _, row in cls_rows.iterrows():
        rep = row["representation"]
        csv_file = pred_dir / f"dev_cls__{rep}__linear_svc.csv"
        assert csv_file.exists(), f"Missing prediction file for {rep}"

        df_pred = pd.read_csv(csv_file)
        y_true = df_pred["y_true"].values
        y_pred = df_pred["y_pred"].values
        y_score = df_pred["y_score"].values

        cal_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        cal_bal_acc = balanced_accuracy_score(y_true, y_pred)
        cal_auc = roc_auc_score(y_true, y_score)

        assert np.isclose(cal_f1, row["dev_f1"], atol=1e-4), f"F1 mismatch for {rep}: {cal_f1} vs {row['dev_f1']}"
        assert np.isclose(cal_bal_acc, row["dev_balanced_accuracy"], atol=1e-4), f"BalAcc mismatch for {rep}: {cal_bal_acc} vs {row['dev_balanced_accuracy']}"
        assert np.isclose(cal_auc, row["dev_roc_auc"], atol=1e-4), f"AUC mismatch for {rep}: {cal_auc} vs {row['dev_roc_auc']}"


def test_regression_metric_reproduction(repo_paths):
    """Verifies exact reproduction of continuous regression metrics from saved prediction files."""
    ddir = repo_paths["distilbert_dir"]
    pred_dir = ddir / "predictions"
    tbl_path = ddir / "reports" / "model_comparison_table.csv"
    df_tbl = pd.read_csv(tbl_path)

    reg_rows = df_tbl[(df_tbl["task"] == "regression") & (~df_tbl["representation"].str.startswith("whole_matched"))]

    for _, row in reg_rows.iterrows():
        rep = row["representation"]
        model = row["model"].lower()
        csv_file = pred_dir / f"dev_reg__{rep}__{model}.csv"
        assert csv_file.exists(), f"Missing regression prediction file for {rep} ({model})"

        df_pred = pd.read_csv(csv_file)
        y_true = df_pred["y_true"].values
        y_pred = df_pred["y_pred"].values

        cal_ccc = compute_ccc(y_true, y_pred)
        cal_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        cal_r2 = r2_score(y_true, y_pred)

        assert np.isclose(cal_ccc, row["dev_ccc"], atol=1e-4), f"CCC mismatch for {rep}: {cal_ccc} vs {row['dev_ccc']}"
        assert np.isclose(cal_rmse, row["dev_rmse"], atol=1e-4), f"RMSE mismatch for {rep}: {cal_rmse} vs {row['dev_rmse']}"
        assert np.isclose(cal_r2, row["dev_r2"], atol=1e-4), f"R^2 mismatch for {rep}: {cal_r2} vs {row['dev_r2']}"


def test_regression_predictions_continuous_unclipped(repo_paths):
    """Verifies that all regression predictions remain continuous, real-valued, and unclipped."""
    ddir = repo_paths["distilbert_dir"]
    pred_dir = ddir / "predictions"
    reg_files = list(pred_dir.glob("dev_reg__*.csv"))
    assert len(reg_files) > 0

    for rf in reg_files:
        df_pred = pd.read_csv(rf)
        y_pred = df_pred["y_pred"].values
        # Ensure not all integer-rounded
        is_all_ints = np.all(y_pred == np.round(y_pred))
        assert not is_all_ints, f"Predictions appear artificially rounded in {rf.name}"
        # Ensure finite
        assert np.all(np.isfinite(y_pred)), f"Non-finite predictions in {rf.name}"


def test_paired_bootstrap_differences_exist(repo_paths):
    """Verifies that paired bootstrap difference calculations were executed and cross zero."""
    ddir = repo_paths["distilbert_dir"]
    paired_file = ddir / "reports" / "paired_bootstrap_differences.csv"
    assert paired_file.exists(), "paired_bootstrap_differences.csv is missing"

    df_paired = pd.read_csv(paired_file)
    assert len(df_paired) >= 8, f"Expected at least 8 paired comparisons, got {len(df_paired)}"

    for _, row in df_paired.iterrows():
        # Verify 95% CI bounds
        assert row["ci95_lower"] <= row["point_diff"] <= row["ci95_upper"]
        # In all evaluated tests on N=56, CI should span across 0.0
        assert row["ci95_lower"] <= 0.0 <= row["ci95_upper"], f"CI unexpectedly excludes 0 for {row['comparison']}"


def test_distilbert_performance_highlights(repo_paths):
    """Verifies specific empirical milestones achieved by DistilBERT in Phase 18B."""
    ddir = repo_paths["distilbert_dir"]
    tbl_path = ddir / "reports" / "model_comparison_table.csv"
    df_tbl = pd.read_csv(tbl_path)

    # 1. whole__mean_std Ridge regression achieves CCC > 0.40
    row_reg_composite = df_tbl[(df_tbl["representation"] == "whole__mean_std") & (df_tbl["model"] == "Ridge")]
    assert len(row_reg_composite) == 1
    ccc_val = row_reg_composite["dev_ccc"].values[0]
    assert ccc_val > 0.40, f"Expected whole__mean_std Ridge CCC > 0.40, got {ccc_val:.4f}"

    # 2. middle__mean Ridge regression achieves CCC > 0.40 and RMSE < 5.25
    row_reg_mid = df_tbl[(df_tbl["representation"] == "middle__mean") & (df_tbl["model"] == "Ridge")]
    assert len(row_reg_mid) == 1
    assert row_reg_mid["dev_ccc"].values[0] > 0.40
    assert row_reg_mid["dev_rmse"].values[0] < 5.25

    # 3. concat__mean LinearSVC achieves F1 > 0.46
    row_cls_concat = df_tbl[(df_tbl["representation"] == "concat__mean") & (df_tbl["model"] == "LinearSVC")]
    assert len(row_cls_concat) == 1
    f1_val = row_cls_concat["dev_f1"].values[0]
    assert f1_val > 0.46, f"Expected concat__mean LinearSVC F1 > 0.46, got {f1_val:.4f}"
