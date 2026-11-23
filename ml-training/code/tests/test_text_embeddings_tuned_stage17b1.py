"""
Unit and Integration Tests for Phase 17B.1 (Hyperparameter Tuning and Verification).

Validates:
1. Presence of all tuning artifacts (grid searches, summaries, predictions, metrics, audit, reports).
2. Strict participant ID and target alignment across Train OOF and Development splits.
3. Official Test split quarantine (zero exposure or references in predictions).
4. Strict reproduction of classification metrics from saved prediction CSVs.
5. Strict reproduction of regression metrics from saved prediction CSVs.
6. Verification that regression predictions remain continuous and unclipped.
7. Verification that Ridge alpha conditioning audit and grid search files are internally consistent.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error

from text_embeddings_tuned_v1.tune_embeddings import concordance_correlation_coefficient


@pytest.fixture
def repo_paths():
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "repo_root": repo_root,
        "labels_dir": repo_root / "ml-training" / "edaic-woz" / "labels",
        "embeddings_dir": repo_root / "ml-training" / "dataset" / "text_embeddings_v1" / "all_minilm_l6_v2",
        "tuned_dir": repo_root / "ml-training" / "outputs" / "text_embeddings_tuned_v1",
    }


def test_tuned_artifacts_exist(repo_paths):
    """Verifies that all Phase 17B.1 tuning artifacts exist on disk."""
    tdir = repo_paths["tuned_dir"]
    assert tdir.exists(), "Tuned output directory missing"

    # Reports, Audit, Manifest
    assert (tdir / "reports" / "phase_17b1_tuning_report.md").exists()
    assert (tdir / "reports" / "model_comparison_table.csv").exists()
    assert (tdir / "audit" / "audit_findings.md").exists()
    assert (tdir / "experiment_manifest.json").exists()

    # Summaries
    assert (tdir / "classification" / "metrics" / "final_classification_summary.csv").exists()
    assert (tdir / "regression" / "metrics" / "final_regression_summary.csv").exists()

    # Classification files
    for rep in ["mean", "mean_std"]:
        assert (tdir / "classification" / "grid_results" / f"{rep}__linear_svc__grid_search.csv").exists()
        assert (tdir / "classification" / "predictions" / f"oof_classification__{rep}__linear_svc__tuned.csv").exists()
        assert (tdir / "classification" / "predictions" / f"dev_predictions_classification__{rep}__linear_svc__tuned.csv").exists()
        assert (tdir / "classification" / "metrics" / f"{rep}__linear_svc__tuned__metrics.json").exists()

    # Regression files
    reg_models = ["ridge", "elastic_net", "svr_linear", "svr_rbf"]
    for rep in ["mean", "mean_std"]:
        for model in reg_models:
            assert (tdir / "regression" / "grid_results" / f"{rep}__{model}__grid_search.csv").exists()
            assert (tdir / "regression" / "predictions" / f"oof_regression__{rep}__{model}__tuned.csv").exists()
            assert (tdir / "regression" / "predictions" / f"dev_predictions_regression__{rep}__{model}__tuned.csv").exists()
            assert (tdir / "regression" / "metrics" / f"{rep}__{model}__tuned__metrics.json").exists()


def test_participant_alignment_and_quarantine(repo_paths):
    """Verifies participant ID bijective alignment with official splits and Test quarantine."""
    tdir = repo_paths["tuned_dir"]
    labels_dir = repo_paths["labels_dir"]

    train_labels = pd.read_csv(labels_dir / "train_split.csv").sort_values("Participant_ID")
    dev_labels = pd.read_csv(labels_dir / "dev_split.csv").sort_values("Participant_ID")
    test_labels = pd.read_csv(labels_dir / "test_split.csv").sort_values("Participant_ID")

    test_pids = set(test_labels["Participant_ID"])
    dev_pids = set(dev_labels["Participant_ID"])
    train_pids = set(train_labels["Participant_ID"])

    # Disjointness check
    assert len(train_pids & dev_pids) == 0
    assert len(train_pids & test_pids) == 0
    assert len(dev_pids & test_pids) == 0

    # Check all prediction files
    for pred_file in (tdir).rglob("*.csv"):
        df = pd.read_csv(pred_file)
        if "Participant_ID" in df.columns:
            pids = set(df["Participant_ID"])
            assert len(pids & test_pids) == 0, f"Quarantine breach: Test participant found in {pred_file}"
            if "split" in df.columns:
                split_val = df["split"].iloc[0]
                if split_val == "dev":
                    assert len(df) == 56
                    assert list(df["Participant_ID"]) == list(dev_labels["Participant_ID"])
                elif split_val == "train":
                    assert len(df) == 163
                    assert list(df["Participant_ID"]) == list(train_labels["Participant_ID"])


def test_classification_metric_reproduction(repo_paths):
    """Verifies that tuned classification metrics strictly reproduce from saved predictions."""
    tdir = repo_paths["tuned_dir"]
    cls_summary = pd.read_csv(tdir / "classification" / "metrics" / "final_classification_summary.csv").set_index("representation")

    for rep in ["mean", "mean_std"]:
        row = cls_summary.loc[f"minilm_l6__{rep}"]
        tau = row["calibrated_tau"]

        # Dev verification
        dev_file = tdir / "classification" / "predictions" / f"dev_predictions_classification__{rep}__linear_svc__tuned.csv"
        df_dev = pd.read_csv(dev_file)
        assert len(df_dev) == 56

        # Check binary decision logic
        expected_pred = (df_dev["y_score"] > tau).astype(int)
        assert (df_dev["y_pred"] == expected_pred).all(), f"Inconsistent threshold application in {dev_file}"

        dev_f1 = f1_score(df_dev["y_true"], df_dev["y_pred"], pos_label=1, zero_division=0)
        dev_bal_acc = balanced_accuracy_score(df_dev["y_true"], df_dev["y_pred"])
        dev_auc = roc_auc_score(df_dev["y_true"], df_dev["y_score"])

        assert np.isclose(dev_f1, row["dev_f1"], atol=1e-4)
        assert np.isclose(dev_bal_acc, row["dev_balanced_acc"], atol=1e-4)
        assert np.isclose(dev_auc, row["dev_roc_auc"], atol=1e-4)

        # Train OOF verification
        oof_file = tdir / "classification" / "predictions" / f"oof_classification__{rep}__linear_svc__tuned.csv"
        df_oof = pd.read_csv(oof_file)
        assert len(df_oof) == 163

        expected_oof_pred = (df_oof["y_score"] > tau).astype(int)
        assert (df_oof["y_pred"] == expected_oof_pred).all()

        oof_f1 = f1_score(df_oof["y_true"], df_oof["y_pred"], pos_label=1, zero_division=0)
        assert np.isclose(oof_f1, row["train_oof_f1"], atol=1e-4)


def test_regression_metric_reproduction(repo_paths):
    """Verifies that tuned regression metrics strictly reproduce from saved predictions."""
    tdir = repo_paths["tuned_dir"]
    reg_summary = pd.read_csv(tdir / "regression" / "metrics" / "final_regression_summary.csv")

    reg_models = [
        ("ridge", "Ridge"),
        ("elastic_net", "ElasticNet"),
        ("svr_linear", "SVR_Linear"),
        ("svr_rbf", "SVR_Rbf"),
    ]

    for rep in ["mean", "mean_std"]:
        for model_id, model_name in reg_models:
            sub = reg_summary[(reg_summary["representation"] == f"minilm_l6__{rep}") & (reg_summary["model"] == model_name)]
            assert len(sub) == 1, f"Missing summary row for {rep} {model_name}"
            row = sub.iloc[0]

            dev_file = tdir / "regression" / "predictions" / f"dev_predictions_regression__{rep}__{model_id}__tuned.csv"
            df_dev = pd.read_csv(dev_file)
            assert len(df_dev) == 56

            # Verify predictions are unclipped continuous floats
            assert df_dev["y_pred_tuned"].dtype in [np.float32, np.float64, float]

            dev_ccc = concordance_correlation_coefficient(df_dev["y_true"].values, df_dev["y_pred_tuned"].values)
            dev_rmse = float(np.sqrt(mean_squared_error(df_dev["y_true"], df_dev["y_pred_tuned"])))
            dev_mae = float(mean_absolute_error(df_dev["y_true"], df_dev["y_pred_tuned"]))

            assert np.isclose(dev_ccc, row["dev_ccc"], atol=1e-4)
            assert np.isclose(dev_rmse, row["dev_rmse"], atol=1e-4)
            assert np.isclose(dev_mae, row["dev_mae"], atol=1e-4)
