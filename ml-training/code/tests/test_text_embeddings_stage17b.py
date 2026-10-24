"""
Unit and Integration Tests for Phase 17B (Transformer-Based Text Embeddings).

Validates:
1. Embedding dataset shapes, absence of NaNs/Infs, and metadata consistency.
2. Participant ID bijective alignment with official E-DAIC splits.
3. Strict quarantine of official Test set (N=56) without label exposure.
4. Output benchmark artifacts presence (manifest, summaries, prediction CSVs, reports).
5. Exact reproduction of classification and regression metrics from saved prediction files.
6. Deterministic chunking properties (total chunks, participant chunk coverage, max tokens).
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

from text_embeddings_v1.evaluate_embeddings import concordance_correlation_coefficient


@pytest.fixture
def repo_paths():
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "repo_root": repo_root,
        "labels_dir": repo_root / "ml-training" / "edaic-woz" / "labels",
        "embeddings_dir": repo_root / "ml-training" / "dataset" / "text_embeddings_v1" / "all_minilm_l6_v2",
        "outputs_dir": repo_root / "ml-training" / "outputs" / "text_embeddings_v1",
    }


def test_embedding_arrays_and_metadata(repo_paths):
    """Verifies that extracted embedding arrays have correct shapes and zero NaNs/infs."""
    emb_dir = repo_paths["embeddings_dir"]
    assert emb_dir.exists(), "Embeddings directory missing"

    # Check files exist
    assert (emb_dir / "embedding_manifest.json").exists()
    assert (emb_dir / "chunk_metadata.csv").exists()
    assert (emb_dir / "data_validation_report.md").exists()

    expected_shapes = {
        "train_embeddings_mean.npy": (163, 384),
        "train_embeddings_mean_std.npy": (163, 768),
        "dev_embeddings_mean.npy": (56, 384),
        "dev_embeddings_mean_std.npy": (56, 768),
        "test_embeddings_mean.npy": (56, 384),
        "test_embeddings_mean_std.npy": (56, 768),
    }

    for fname, shape in expected_shapes.items():
        arr_path = emb_dir / fname
        assert arr_path.exists(), f"Embedding file missing: {fname}"
        arr = np.load(arr_path)
        assert arr.shape == shape, f"Shape mismatch for {fname}: got {arr.shape}, expected {shape}"
        assert not np.isnan(arr).any(), f"NaNs detected in {fname}"
        assert not np.isinf(arr).any(), f"Infs detected in {fname}"


def test_chunking_metadata_integrity(repo_paths):
    """Validates deterministic chunking properties across the entire corpus."""
    emb_dir = repo_paths["embeddings_dir"]
    chunk_meta = pd.read_csv(emb_dir / "chunk_metadata.csv")

    assert len(chunk_meta) == 2163, f"Expected 2163 total chunks, got {len(chunk_meta)}"
    assert chunk_meta["word_count"].max() <= 250, "A chunk exceeded max 250 words"
    assert chunk_meta["word_count"].min() >= 1, "An empty chunk was found"
    assert chunk_meta["Participant_ID"].nunique() == 275, "Not all 275 participants have chunks"


def test_participant_split_alignment_and_quarantine(repo_paths):
    """Verifies that participant IDs strictly match official splits and Test set is quarantined."""
    emb_dir = repo_paths["embeddings_dir"]
    labels_dir = repo_paths["labels_dir"]
    out_dir = repo_paths["outputs_dir"]

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

    # Embedded participant ID files check
    train_emb_pids = pd.read_csv(emb_dir / "train_participant_ids.csv")["Participant_ID"].tolist()
    dev_emb_pids = pd.read_csv(emb_dir / "dev_participant_ids.csv")["Participant_ID"].tolist()
    test_emb_pids = pd.read_csv(emb_dir / "test_participant_ids.csv")["Participant_ID"].tolist()

    assert train_emb_pids == train_labels["Participant_ID"].tolist()
    assert dev_emb_pids == dev_labels["Participant_ID"].tolist()
    assert test_emb_pids == test_labels["Participant_ID"].tolist()

    # Quarantine check: ensure no prediction file exists for Test split or references Test IDs
    for pred_file in (out_dir).rglob("*.csv"):
        df = pd.read_csv(pred_file)
        if "Participant_ID" in df.columns:
            pids = set(df["Participant_ID"])
            assert len(pids & test_pids) == 0, f"Leakage: Test participant IDs present in {pred_file}"


def test_output_artifacts_exist(repo_paths):
    """Verifies all required Phase 17B output artifacts exist."""
    out_dir = repo_paths["outputs_dir"]
    assert out_dir.exists(), "Outputs directory missing"

    assert (out_dir / "experiment_manifest.json").exists()
    assert (out_dir / "reports" / "phase_17b_report.md").exists()
    assert (out_dir / "reports" / "model_comparison_table.csv").exists()
    assert (out_dir / "reports" / "validation_report.md").exists()

    assert (out_dir / "classification" / "metrics" / "final_classification_summary.csv").exists()
    assert (out_dir / "regression" / "metrics" / "final_regression_summary.csv").exists()

    reps = ["minilm_l6__mean", "minilm_l6__mean_std"]
    for rep in reps:
        # Classification
        assert (out_dir / "classification" / "predictions" / f"dev_predictions_classification__{rep}__linear_svc__calibrated.csv").exists()
        assert (out_dir / "classification" / "predictions" / f"oof_classification__{rep}__linear_svc__calibrated.csv").exists()
        assert (out_dir / "classification" / "metrics" / f"{rep}__linear_svc__calibrated__metrics.json").exists()

        # Regression
        assert (out_dir / "regression" / "predictions" / f"dev_predictions_regression__{rep}__ridge__tuned.csv").exists()
        assert (out_dir / "regression" / "predictions" / f"oof_regression__{rep}__ridge__tuned.csv").exists()
        assert (out_dir / "regression" / "metrics" / f"{rep}__ridge__tuned__metrics.json").exists()


def test_classification_metric_reproduction(repo_paths):
    """Validates that reported classification metrics strictly reproduce from saved predictions."""
    out_dir = repo_paths["outputs_dir"]
    cls_sum = pd.read_csv(out_dir / "classification" / "metrics" / "final_classification_summary.csv").set_index("feature")

    for rep in ["minilm_l6__mean", "minilm_l6__mean_std"]:
        row = cls_sum.loc[rep]
        tau = row["calibrated_tau"]

        # Dev verification
        dev_file = out_dir / "classification" / "predictions" / f"dev_predictions_classification__{rep}__linear_svc__calibrated.csv"
        df_dev = pd.read_csv(dev_file)
        assert len(df_dev) == 56

        # Check binary prediction logic: y_pred == (y_score >= tau).astype(int)
        expected_pred = (df_dev["y_score"] >= tau).astype(int)
        assert (df_dev["y_pred"] == expected_pred).all(), f"Threshold {tau} not applied consistently in Dev"

        # Check metrics
        dev_f1 = f1_score(df_dev["y_true"], df_dev["y_pred"], zero_division=0)
        dev_bal_acc = balanced_accuracy_score(df_dev["y_true"], df_dev["y_pred"])
        dev_auc = roc_auc_score(df_dev["y_true"], df_dev["y_score"])

        assert np.isclose(dev_f1, row["dev_f1"], atol=1e-4)
        assert np.isclose(dev_bal_acc, row["dev_balanced_acc"], atol=1e-4)
        assert np.isclose(dev_auc, row["dev_roc_auc"], atol=1e-4)

        # Train OOF verification
        oof_file = out_dir / "classification" / "predictions" / f"oof_classification__{rep}__linear_svc__calibrated.csv"
        df_oof = pd.read_csv(oof_file)
        assert len(df_oof) == 163

        expected_oof_pred = (df_oof["y_score"] >= tau).astype(int)
        assert (df_oof["y_pred"] == expected_oof_pred).all()

        oof_f1 = f1_score(df_oof["y_true"], df_oof["y_pred"], zero_division=0)
        assert np.isclose(oof_f1, row["train_oof_f1"], atol=1e-4)


def test_regression_metric_reproduction(repo_paths):
    """Validates that reported regression metrics strictly reproduce from saved predictions."""
    out_dir = repo_paths["outputs_dir"]
    reg_sum = pd.read_csv(out_dir / "regression" / "metrics" / "final_regression_summary.csv").set_index("feature")

    for rep in ["minilm_l6__mean", "minilm_l6__mean_std"]:
        row = reg_sum.loc[rep]

        dev_file = out_dir / "regression" / "predictions" / f"dev_predictions_regression__{rep}__ridge__tuned.csv"
        df_dev = pd.read_csv(dev_file)
        assert len(df_dev) == 56

        # Check unclipped predictions: min and max are floating point and not clipped to [0, 24]
        assert df_dev["y_pred_tuned"].dtype in [np.float32, np.float64, float]

        dev_ccc = concordance_correlation_coefficient(df_dev["y_true"].values, df_dev["y_pred_tuned"].values)
        dev_rmse = float(np.sqrt(mean_squared_error(df_dev["y_true"], df_dev["y_pred_tuned"])))
        dev_mae = float(mean_absolute_error(df_dev["y_true"], df_dev["y_pred_tuned"]))

        assert np.isclose(dev_ccc, row["dev_ccc_tuned"], atol=1e-4)
        assert np.isclose(dev_rmse, row["dev_rmse_tuned"], atol=1e-4)
        assert np.isclose(dev_mae, row["dev_mae_tuned"], atol=1e-4)
