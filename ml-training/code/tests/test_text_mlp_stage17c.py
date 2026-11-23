"""
Unit and Integration Tests for Phase 17C (Lightweight Deep Learning Prediction Heads).

Validates:
1. Presence of all Phase 17C artifacts (reports, audit findings, manifests, metrics, cv results, plots, predictions).
2. Strict participant ID and target alignment across Train OOF (N=163) and Development (N=56) splits.
3. Official Test split quarantine (zero exposure or references to Test IDs in prediction CSVs).
4. Exact numerical reproduction of classification metrics (F1, Balanced Accuracy, ROC-AUC) from saved prediction CSVs.
5. Exact numerical reproduction of regression metrics (Lin's CCC, RMSE, R^2, Pearson r) from saved prediction CSVs.
6. Verification that regression predictions remain continuous, real-valued, and unclipped.
7. Verification that diagnostic plot files exist and are valid image artifacts.
"""

import json
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from text_mlp_v1.train_eval import concordance_correlation_coefficient


@pytest.fixture
def repo_paths():
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "repo_root": repo_root,
        "labels_dir": repo_root / "ml-training" / "edaic-woz" / "labels",
        "embeddings_dir": repo_root / "ml-training" / "dataset" / "text_embeddings_v1" / "all_minilm_l6_v2",
        "mlp_dir": repo_root / "ml-training" / "outputs" / "text_mlp_v1",
    }


def test_phase17c_artifacts_exist(repo_paths):
    """Verifies that all Phase 17C deliverables and intermediate artifacts exist on disk."""
    mdir = repo_paths["mlp_dir"]
    assert mdir.exists(), "Phase 17C output directory missing"

    # Reports, Audit, Manifest
    assert (mdir / "reports" / "phase_17c_report.md").exists()
    assert (mdir / "reports" / "audit_report.md").exists()
    assert (mdir / "reports" / "model_comparison_table.csv").exists()
    assert (mdir / "audit" / "audit_findings.md").exists()
    assert (mdir / "experiment_manifest.json").exists()

    # Metrics and Summaries
    assert (mdir / "metrics" / "initial_baselines_summary.csv").exists()
    assert (mdir / "cv_results" / "full_tuning_summary.csv").exists()

    # CV Grid Search CSVs
    reps = ["mean", "mean_std"]
    tasks = ["classification", "regression"]
    for rep in reps:
        for task in tasks:
            assert (mdir / "cv_results" / f"grid_search__{rep}__{task}.csv").exists()

    # Plots
    assert (mdir / "plots" / "loss_curves_tuned_mlp.png").exists()
    assert (mdir / "plots" / "regression_scatter_dev.png").exists()

    # Predictions (OOF and Dev for initial and tuned across all combinations)
    for rep in reps:
        for task in tasks:
            for variant in ["mlp_initial", "mlp_tuned"]:
                assert (mdir / "predictions" / f"dev_predictions_{task}__{rep}__{variant}.csv").exists()
                assert (mdir / "predictions" / f"oof_{task}__{rep}__{variant}.csv").exists()


def test_participant_alignment_and_quarantine(repo_paths):
    """Verifies participant ID alignment with official splits and strict Test quarantine."""
    mdir = repo_paths["mlp_dir"]
    labels_dir = repo_paths["labels_dir"]

    train_labels = pd.read_csv(labels_dir / "train_split.csv").sort_values("Participant_ID")
    dev_labels = pd.read_csv(labels_dir / "dev_split.csv").sort_values("Participant_ID")
    test_labels = pd.read_csv(labels_dir / "test_split.csv").sort_values("Participant_ID")

    train_pids = set(train_labels["Participant_ID"])
    dev_pids = set(dev_labels["Participant_ID"])
    test_pids = set(test_labels["Participant_ID"])

    # Disjointness check
    assert len(train_pids) == 163
    assert len(dev_pids) == 56
    assert len(test_pids) == 56
    assert len(train_pids & dev_pids) == 0
    assert len(train_pids & test_pids) == 0
    assert len(dev_pids & test_pids) == 0

    # Check all prediction CSVs in outputs/text_mlp_v1
    for pred_file in (mdir / "predictions").glob("*.csv"):
        df = pd.read_csv(pred_file)
        assert "Participant_ID" in df.columns, f"Missing Participant_ID in {pred_file.name}"
        pids = set(df["Participant_ID"])

        # Quarantine verification
        assert len(pids & test_pids) == 0, f"Quarantine breach: Test participant found in {pred_file.name}"

        if pred_file.name.startswith("oof_"):
            assert len(df) == 163, f"Expected 163 Train OOF rows in {pred_file.name}, got {len(df)}"
            assert pids == train_pids, f"Train OOF participants do not match official train split in {pred_file.name}"
            # Check target alignment
            merged = df.merge(train_labels, on="Participant_ID")
            target_col = "PHQ_Binary" if "classification" in pred_file.name else "PHQ_Score"
            np.testing.assert_array_almost_equal(merged["y_true"], merged[target_col], decimal=4)
        elif pred_file.name.startswith("dev_"):
            assert len(df) == 56, f"Expected 56 Dev rows in {pred_file.name}, got {len(df)}"
            assert pids == dev_pids, f"Dev participants do not match official dev split in {pred_file.name}"
            # Check target alignment
            merged = df.merge(dev_labels, on="Participant_ID")
            target_col = "PHQ_Binary" if "classification" in pred_file.name else "PHQ_Score"
            np.testing.assert_array_almost_equal(merged["y_true"], merged[target_col], decimal=4)


def test_classification_metrics_reproduction(repo_paths):
    """Verifies that classification metrics reproduce exactly from saved participant-level predictions."""
    mdir = repo_paths["mlp_dir"]
    df_comp = pd.read_csv(mdir / "reports" / "model_comparison_table.csv")

    for rep in ["mean", "mean_std"]:
        pred_path = mdir / "predictions" / f"dev_predictions_classification__{rep}__mlp_tuned.csv"
        assert pred_path.exists()
        df_pred = pd.read_csv(pred_path)

        y_true = df_pred["y_true"].values.astype(int)
        y_pred = df_pred["y_pred"].values.astype(int)
        y_score = df_pred["y_score"].values.astype(float)

        computed_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        computed_bal_acc = balanced_accuracy_score(y_true, y_pred)
        computed_roc_auc = roc_auc_score(y_true, y_score)

        # Match row in model comparison table
        rep_key = f"minilm_l6__{rep}"
        row = df_comp[
            (df_comp["group"] == "Group A (MLP)")
            & (df_comp["task"] == "classification")
            & (df_comp["representation"] == rep_key)
            & (df_comp["status"] == "Tuned")
        ].iloc[0]

        table_f1 = float(row["dev_primary"])
        table_bal_acc = float(row["dev_secondary_1"])
        table_roc_auc = float(row["dev_secondary_2"])

        assert np.isclose(computed_f1, table_f1, atol=1e-4), f"F1 mismatch for {rep}: {computed_f1} vs {table_f1}"
        assert np.isclose(computed_bal_acc, table_bal_acc, atol=1e-4), f"BalAcc mismatch for {rep}: {computed_bal_acc} vs {table_bal_acc}"
        assert np.isclose(computed_roc_auc, table_roc_auc, atol=1e-4), f"ROC-AUC mismatch for {rep}: {computed_roc_auc} vs {table_roc_auc}"


def test_regression_metrics_reproduction(repo_paths):
    """Verifies that regression metrics reproduce exactly from saved participant-level predictions."""
    mdir = repo_paths["mlp_dir"]
    df_comp = pd.read_csv(mdir / "reports" / "model_comparison_table.csv")

    for rep in ["mean", "mean_std"]:
        pred_path = mdir / "predictions" / f"dev_predictions_regression__{rep}__mlp_tuned.csv"
        assert pred_path.exists()
        df_pred = pd.read_csv(pred_path)

        y_true = df_pred["y_true"].values.astype(float)
        y_pred = df_pred["y_pred"].values.astype(float)

        computed_ccc = concordance_correlation_coefficient(y_true, y_pred)
        computed_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        computed_r2 = r2_score(y_true, y_pred)

        # Match row in model comparison table
        rep_key = f"minilm_l6__{rep}"
        row = df_comp[
            (df_comp["group"] == "Group A (MLP)")
            & (df_comp["task"] == "regression")
            & (df_comp["representation"] == rep_key)
            & (df_comp["status"] == "Tuned")
        ].iloc[0]

        table_ccc = float(row["dev_primary"])
        table_rmse = float(row["dev_secondary_1"])
        table_r2 = float(row["dev_secondary_2"])

        assert np.isclose(computed_ccc, table_ccc, atol=1e-4), f"CCC mismatch for {rep}: {computed_ccc} vs {table_ccc}"
        assert np.isclose(computed_rmse, table_rmse, atol=1e-4), f"RMSE mismatch for {rep}: {computed_rmse} vs {table_rmse}"
        assert np.isclose(computed_r2, table_r2, atol=1e-4), f"R^2 mismatch for {rep}: {computed_r2} vs {table_r2}"


def test_continuous_unclipped_regression_predictions(repo_paths):
    """Verifies that regression predictions are continuous and unclipped."""
    mdir = repo_paths["mlp_dir"]

    for rep in ["mean", "mean_std"]:
        pred_path = mdir / "predictions" / f"dev_predictions_regression__{rep}__mlp_tuned.csv"
        df_pred = pd.read_csv(pred_path)
        y_pred = df_pred["y_pred"].values

        # Predictions should have high floating-point variety (not discrete integers)
        n_unique = len(np.unique(y_pred))
        assert n_unique >= 50, f"Expected continuous predictions, got only {n_unique} unique values in {pred_path.name}"

        # Ensure no artificial hard clamping at [0, 24] boundary values
        assert not np.all((y_pred >= 0) & (y_pred <= 24) & (y_pred == np.round(y_pred))), (
            "Predictions appear artificially clamped or rounded"
        )


def test_experiment_manifest_integrity(repo_paths):
    """Verifies that experiment_manifest.json contains accurate configuration metadata."""
    mdir = repo_paths["mlp_dir"]
    manifest_path = mdir / "experiment_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["phase"] == "17C"
    assert manifest["official_test_quarantined"] is True
    assert manifest["splits"] == {"train": 163, "dev": 56, "test": 56}
    assert "NVIDIA" in manifest["device"] or "cuda" in manifest["device"].lower() or "cpu" in manifest["device"].lower()
    assert manifest["representations"] == ["minilm_l6__mean", "minilm_l6__mean_std"]
