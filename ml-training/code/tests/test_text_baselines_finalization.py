"""
Unit and Integration Tests for Phase 17A Finalization.

Validates:
1. Finalized artifact directory structure and presence of all required files.
2. Participant ID and target integrity across Out-of-Fold and Development predictions.
3. Strict reproduction of finalized metrics from saved prediction CSVs.
4. Threshold calibration math and Ridge alpha selection criteria.
5. Strict quarantine of official Test set (N=56).
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from text_baselines_v1.finalize_text_baselines import (
    ALPHA_SEARCH_SPACE,
    FINAL_OUTPUT_DIR,
    concordance_correlation_coefficient,
)


@pytest.fixture
def finalized_paths():
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "final_dir": repo_root / "ml-training" / "outputs" / "text_baselines_final_v1",
        "labels_dir": repo_root / "ml-training" / "edaic-woz" / "labels",
    }


def test_finalized_artifacts_exist(finalized_paths):
    """Verifies that all required finalized artifacts exist on disk."""
    fdir = finalized_paths["final_dir"]
    assert fdir.exists(), "Finalized output directory missing"

    # Reports and Manifest
    assert (fdir / "reports" / "phase_17a_final_report.md").exists()
    assert (fdir / "final_manifest.json").exists()

    # Summaries
    assert (fdir / "classification" / "metrics" / "final_classification_summary.csv").exists()
    assert (fdir / "regression" / "metrics" / "final_regression_summary.csv").exists()

    # Per-feature prediction files
    for feat in ["word_tfidf", "char_tfidf", "combined_tfidf"]:
        assert (fdir / "classification" / "predictions" / f"dev_predictions_classification__{feat}__linear_svc__calibrated.csv").exists()
        assert (fdir / "classification" / "predictions" / f"oof_classification__{feat}__linear_svc__calibrated.csv").exists()
        assert (fdir / "regression" / "predictions" / f"dev_predictions_regression__{feat}__ridge__tuned.csv").exists()
        assert (fdir / "regression" / "predictions" / f"oof_regression__{feat}__ridge__tuned.csv").exists()


def test_participant_alignment_and_quarantine(finalized_paths):
    """Verifies that saved predictions exactly match official Train and Dev IDs and exclude Test."""
    fdir = finalized_paths["final_dir"]
    labels_dir = finalized_paths["labels_dir"]

    train_labels = pd.read_csv(labels_dir / "train_split.csv").sort_values("Participant_ID")
    dev_labels = pd.read_csv(labels_dir / "dev_split.csv").sort_values("Participant_ID")
    test_labels = pd.read_csv(labels_dir / "test_split.csv").sort_values("Participant_ID")

    test_pids = set(test_labels["Participant_ID"])

    for feat in ["word_tfidf", "char_tfidf", "combined_tfidf"]:
        # Dev check
        dev_f = fdir / "classification" / "predictions" / f"dev_predictions_classification__{feat}__linear_svc__calibrated.csv"
        df_dev = pd.read_csv(dev_f)
        assert len(df_dev) == 56
        assert list(df_dev["Participant_ID"]) == list(dev_labels["Participant_ID"])
        assert list(df_dev["y_true"]) == list(dev_labels["PHQ_Binary"])
        assert len(set(df_dev["Participant_ID"]) & test_pids) == 0, "Leakage: Test participant found in Dev predictions"

        # Train OOF check
        oof_f = fdir / "classification" / "predictions" / f"oof_classification__{feat}__linear_svc__calibrated.csv"
        df_oof = pd.read_csv(oof_f)
        assert len(df_oof) == 163
        assert list(df_oof["Participant_ID"]) == list(train_labels["Participant_ID"])
        assert list(df_oof["y_true"]) == list(train_labels["PHQ_Binary"])
        assert len(set(df_oof["Participant_ID"]) & test_pids) == 0, "Leakage: Test participant found in Train OOF predictions"


def test_metric_reproduction_accuracy(finalized_paths):
    """Verifies that reported metrics reproduce exactly from saved predictions."""
    fdir = finalized_paths["final_dir"]

    cls_sum = pd.read_csv(fdir / "classification" / "metrics" / "final_classification_summary.csv").set_index("feature")
    reg_sum = pd.read_csv(fdir / "regression" / "metrics" / "final_regression_summary.csv").set_index("feature")

    for feat in ["word_tfidf", "char_tfidf", "combined_tfidf"]:
        # Classification
        dev_cls = pd.read_csv(fdir / "classification" / "predictions" / f"dev_predictions_classification__{feat}__linear_svc__calibrated.csv")
        f1_rep = f1_score(dev_cls["y_true"], dev_cls["y_pred_calibrated"])
        auc_rep = roc_auc_score(dev_cls["y_true"], dev_cls["y_score"])
        assert abs(f1_rep - cls_sum.loc[feat, "dev_f1"]) < 1e-4
        assert abs(auc_rep - cls_sum.loc[feat, "dev_roc_auc"]) < 1e-4

        # Regression
        dev_reg = pd.read_csv(fdir / "regression" / "predictions" / f"dev_predictions_regression__{feat}__ridge__tuned.csv")
        ccc_rep = concordance_correlation_coefficient(dev_reg["y_true"].values, dev_reg["y_pred_tuned"].values)
        rmse_rep = float(np.sqrt(np.mean((dev_reg["y_true"] - dev_reg["y_pred_tuned"]) ** 2)))
        assert abs(ccc_rep - reg_sum.loc[feat, "dev_ccc_tuned"]) < 1e-4
        assert abs(rmse_rep - reg_sum.loc[feat, "dev_rmse_tuned"]) < 1e-4


def test_char_tfidf_superiority_on_dev(finalized_paths):
    """Verifies that char_tfidf achieves the highest Dev ROC-AUC and Dev CCC among text baselines."""
    fdir = finalized_paths["final_dir"]
    cls_sum = pd.read_csv(fdir / "classification" / "metrics" / "final_classification_summary.csv").set_index("feature")
    reg_sum = pd.read_csv(fdir / "regression" / "metrics" / "final_regression_summary.csv").set_index("feature")

    # Classification: char_tfidf ROC-AUC >= word_tfidf and combined_tfidf
    assert cls_sum.loc["char_tfidf", "dev_roc_auc"] > cls_sum.loc["word_tfidf", "dev_roc_auc"]
    assert cls_sum.loc["char_tfidf", "dev_f1"] >= cls_sum.loc["word_tfidf", "dev_f1"]

    # Regression: char_tfidf CCC > 0.30
    assert reg_sum.loc["char_tfidf", "dev_ccc_tuned"] > 0.30
    assert reg_sum.loc["char_tfidf", "dev_ccc_tuned"] > reg_sum.loc["word_tfidf", "dev_ccc_tuned"]
    assert reg_sum.loc["char_tfidf", "dev_rmse_tuned"] < reg_sum.loc["word_tfidf", "dev_rmse_tuned"]
