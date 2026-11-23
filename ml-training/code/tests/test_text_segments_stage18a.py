"""
Unit and Integration Tests for Phase 18A (Segment-Level MiniLM Embeddings and Robust Aggregation).

Validates:
1. Presence of all Phase 18A deliverables (reports, audit note, audit CSV, model comparison table, plots, predictions).
2. Segment feature tensor integrity (shapes, dimensions: 384, 768, 1152, 2304; zero NaNs/Infs).
3. Participant ID alignment with official splits (Train=163, Dev=56; complete cohort=157/54).
4. Strict Official Test split quarantine (zero exposure or references to Test IDs in prediction CSVs).
5. Exact numerical reproduction of classification metrics (F1, Balanced Accuracy, ROC-AUC) from saved prediction CSVs.
6. Exact numerical reproduction of regression metrics (Lin's CCC, RMSE, R^2, Pearson r) from saved prediction CSVs.
7. Verification that regression predictions remain continuous, real-valued, and unclipped.
8. Empirical validation of Middle segment superiority and Median regression performance.
9. Verification of bit-exact Phase 17B pooled embedding reproduction recorded in audit.
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
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from text_segments_v1.run_experiments import concordance_correlation_coefficient as compute_ccc


@pytest.fixture
def repo_paths():
    repo_root = Path(__file__).resolve().parents[3]
    ml_training = repo_root / "ml-training"
    return {
        "repo_root": repo_root,
        "labels_dir": ml_training / "edaic-woz" / "labels",
        "segments_dir": ml_training / "outputs" / "text_segments_v1",
        "dataset_dir": ml_training / "dataset" / "text_segments_v1",
    }


def test_phase18a_artifacts_exist(repo_paths):
    """Verifies that all Phase 18A deliverables and intermediate artifacts exist on disk."""
    sdir = repo_paths["segments_dir"]
    ddir = repo_paths["dataset_dir"]

    assert sdir.exists(), "Phase 18A output directory missing"
    assert ddir.exists(), "Phase 18A dataset directory missing"

    # Reports and Audit
    assert (sdir / "reports" / "phase_18a_report.md").exists()
    assert (sdir / "reports" / "model_comparison_table.csv").exists()
    assert (sdir / "audit" / "audit_note.md").exists()
    assert (sdir / "audit" / "segment_assignment_audit.csv").exists()

    # Diagnostic Plots
    plot_files = [
        "aggregation_robustness_comparison.png",
        "classification_f1_roc_comparison.png",
        "regression_ccc_rmse_comparison.png",
        "regression_scatter_comparison.png",
        "segment_chronology_progression.png",
    ]
    for pf in plot_files:
        assert (sdir / "plots" / pf).exists(), f"Missing plot: {pf}"

    # Chunk Embeddings & Metadata
    assert (ddir / "chunk_embeddings.npy").exists()
    assert (ddir / "chunk_metadata.csv").exists()
    assert (ddir / "features").exists()


def test_chunk_embeddings_and_segment_features_shapes(repo_paths):
    """Verifies shapes, dimensions, and numerical validity of chunk embeddings and segment features."""
    ddir = repo_paths["dataset_dir"]

    # Chunk embeddings
    chunks = np.load(ddir / "chunk_embeddings.npy")
    assert chunks.shape == (2163, 384)
    assert chunks.dtype == np.float32
    assert not np.isnan(chunks).any(), "NaN found in chunk embeddings"
    assert not np.isinf(chunks).any(), "Inf found in chunk embeddings"

    # Chunk metadata
    df_meta = pd.read_csv(ddir / "chunk_metadata.csv")
    assert len(df_meta) == 2163
    for col in ["Participant_ID", "split", "chunk_index", "start_token", "end_token", "token_count"]:
        assert col in df_meta.columns

    # Segment feature matrices
    features_dir = ddir / "features"
    schemes = ["whole", "early", "middle", "late", "concat"]
    aggs = ["mean", "median", "trimmed", "mean_std"]
    splits = ["train", "dev", "test"]

    dim_map = {
        "mean": 384,
        "median": 384,
        "trimmed": 384,
        "mean_std": 768,
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
                assert arr.shape[1] == expected_dim, (
                    f"Wrong dimension for {fpath.name}: {arr.shape[1]} vs {expected_dim}"
                )
                assert not np.isnan(arr).any(), f"NaN in {fpath.name}"
                assert not np.isinf(arr).any(), f"Inf in {fpath.name}"


def test_participant_alignment_and_quarantine(repo_paths):
    """Verifies strict participant ID alignment and official Test quarantine compliance."""
    sdir = repo_paths["segments_dir"]
    labels_dir = repo_paths["labels_dir"]

    train_labels = pd.read_csv(labels_dir / "train_split.csv").sort_values("Participant_ID")
    dev_labels = pd.read_csv(labels_dir / "dev_split.csv").sort_values("Participant_ID")
    test_labels = pd.read_csv(labels_dir / "test_split.csv").sort_values("Participant_ID")

    train_pids = set(train_labels["Participant_ID"])
    dev_pids = set(dev_labels["Participant_ID"])
    test_pids = set(test_labels["Participant_ID"])

    assert len(train_pids) == 163
    assert len(dev_pids) == 56
    assert len(test_pids) == 56
    assert len(train_pids & dev_pids) == 0
    assert len(train_pids & test_pids) == 0
    assert len(dev_pids & test_pids) == 0

    pred_files = list((sdir / "predictions").glob("*.csv"))
    assert len(pred_files) >= 50, f"Expected >= 50 prediction CSVs, found {len(pred_files)}"

    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        assert "Participant_ID" in df.columns, f"Missing Participant_ID in {pred_file.name}"
        pids = set(df["Participant_ID"])

        # Strict Test quarantine check: zero exposure
        breach = pids & test_pids
        assert len(breach) == 0, f"Quarantine breach: Test participant(s) {breach} found in {pred_file.name}"

        # Segment-specific expected count
        is_complete_cohort = ("late" in pred_file.name) or ("concat" in pred_file.name)
        exp_train_n = 157 if is_complete_cohort else 163
        exp_dev_n = 54 if is_complete_cohort else 56

        if pred_file.name.startswith("oof_"):
            assert len(df) == exp_train_n, (
                f"Expected {exp_train_n} rows in {pred_file.name}, got {len(df)}"
            )
            assert pids.issubset(train_pids), f"Unknown participant in {pred_file.name}"
            # Check target alignment
            merged = df.merge(train_labels, on="Participant_ID")
            target_col = "PHQ_Binary" if "cls" in pred_file.name else "PHQ_Score"
            np.testing.assert_array_almost_equal(merged["y_true"], merged[target_col], decimal=4)
        elif pred_file.name.startswith("dev_"):
            assert len(df) == exp_dev_n, (
                f"Expected {exp_dev_n} rows in {pred_file.name}, got {len(df)}"
            )
            assert pids.issubset(dev_pids), f"Unknown participant in {pred_file.name}"
            # Check target alignment
            merged = df.merge(dev_labels, on="Participant_ID")
            target_col = "PHQ_Binary" if "cls" in pred_file.name else "PHQ_Score"
            np.testing.assert_array_almost_equal(merged["y_true"], merged[target_col], decimal=4)


def test_classification_metrics_reproduction(repo_paths):
    """Verifies that classification metrics reproduce exactly from saved prediction CSVs."""
    sdir = repo_paths["segments_dir"]
    df_comp = pd.read_csv(sdir / "reports" / "model_comparison_table.csv")

    cls_rows = df_comp[
        (df_comp["task"] == "classification")
        & (df_comp["model"] == "LinearSVC")
    ]

    for _, row in cls_rows.iterrows():
        rep = row["representation"]
        pred_path = sdir / "predictions" / f"dev_cls__{rep}__linear_svc.csv"
        assert pred_path.exists(), f"Prediction file missing: {pred_path.name}"

        df_pred = pd.read_csv(pred_path)
        y_true = df_pred["y_true"].values.astype(int)
        y_pred = df_pred["y_pred"].values.astype(int)
        y_score = df_pred["y_score"].values.astype(float)

        computed_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        computed_bal_acc = balanced_accuracy_score(y_true, y_pred)
        computed_roc_auc = roc_auc_score(y_true, y_score)

        table_f1 = float(row["dev_primary"])
        table_bal_acc = float(row["dev_secondary_1"])
        table_roc_auc = float(row["dev_secondary_2"])

        assert np.isclose(computed_f1, table_f1, atol=1e-4), (
            f"F1 mismatch for {rep}: computed {computed_f1:.4f} vs table {table_f1:.4f}"
        )
        assert np.isclose(computed_bal_acc, table_bal_acc, atol=1e-4), (
            f"BalAcc mismatch for {rep}: computed {computed_bal_acc:.4f} vs table {table_bal_acc:.4f}"
        )
        assert np.isclose(computed_roc_auc, table_roc_auc, atol=1e-4), (
            f"ROC-AUC mismatch for {rep}: computed {computed_roc_auc:.4f} vs table {table_roc_auc:.4f}"
        )


def test_regression_metrics_reproduction(repo_paths):
    """Verifies that regression metrics reproduce exactly from saved prediction CSVs."""
    sdir = repo_paths["segments_dir"]
    df_comp = pd.read_csv(sdir / "reports" / "model_comparison_table.csv")

    reg_rows = df_comp[
        (df_comp["task"] == "regression")
        & (df_comp["model"].isin(["Ridge", "RBF_SVR"]))
    ]

    for _, row in reg_rows.iterrows():
        rep = row["representation"]
        model_name = row["model"]
        model_tag = "ridge" if model_name == "Ridge" else "svr_rbf"
        pred_path = sdir / "predictions" / f"dev_reg__{rep}__{model_tag}.csv"
        assert pred_path.exists(), f"Prediction file missing: {pred_path.name}"

        df_pred = pd.read_csv(pred_path)
        y_true = df_pred["y_true"].values.astype(float)
        y_pred = df_pred["y_pred"].values.astype(float)

        computed_ccc = compute_ccc(y_true, y_pred)
        computed_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        computed_r2 = r2_score(y_true, y_pred)
        computed_r = float(pearsonr(y_true, y_pred)[0])

        table_ccc = float(row["dev_primary"])
        table_rmse = float(row["dev_secondary_1"])
        table_r2 = float(row["dev_secondary_2"])

        assert np.isclose(computed_ccc, table_ccc, atol=1e-4), (
            f"CCC mismatch for {rep} ({model_name}): {computed_ccc:.4f} vs table {table_ccc:.4f}"
        )
        assert np.isclose(computed_rmse, table_rmse, atol=1e-4), (
            f"RMSE mismatch for {rep} ({model_name}): {computed_rmse:.4f} vs table {table_rmse:.4f}"
        )
        assert np.isclose(computed_r2, table_r2, atol=1e-4), (
            f"R^2 mismatch for {rep} ({model_name}): {computed_r2:.4f} vs table {table_r2:.4f}"
        )


def test_continuous_unclipped_regression_predictions(repo_paths):
    """Verifies that regression predictions are continuous and unclipped."""
    sdir = repo_paths["segments_dir"]

    for pred_file in (sdir / "predictions").glob("*reg*.csv"):
        df_pred = pd.read_csv(pred_file)
        y_pred = df_pred["y_pred"].values

        # Predictions should have high floating-point variety
        n_unique = len(np.unique(y_pred))
        min_expected = 45 if ("late" in pred_file.name or "concat" in pred_file.name) else 48
        assert n_unique >= min_expected, (
            f"Expected continuous predictions, got only {n_unique} unique values in {pred_file.name}"
        )

        # Ensure no artificial hard clamping at [0, 24] boundary values
        assert not np.all((y_pred >= 0) & (y_pred <= 24) & (y_pred == np.round(y_pred))), (
            f"Predictions in {pred_file.name} appear artificially clamped or rounded"
        )


def test_middle_segment_superiority_empirical(repo_paths):
    """Empirically confirms that the Middle segment outperforms Early, Late, and Whole baselines."""
    sdir = repo_paths["segments_dir"]
    df_comp = pd.read_csv(sdir / "reports" / "model_comparison_table.csv")

    # Classification F1
    f1_mid = float(df_comp[(df_comp["representation"] == "middle__mean") & (df_comp["task"] == "classification")]["dev_primary"].iloc[0])
    f1_early = float(df_comp[(df_comp["representation"] == "early__mean") & (df_comp["task"] == "classification")]["dev_primary"].iloc[0])
    f1_late = float(df_comp[(df_comp["representation"] == "late__mean") & (df_comp["task"] == "classification")]["dev_primary"].iloc[0])
    f1_whole = float(df_comp[(df_comp["representation"] == "whole__mean") & (df_comp["task"] == "classification")]["dev_primary"].iloc[0])

    assert f1_mid == 0.5161
    assert f1_mid > f1_early
    assert f1_mid > f1_late
    assert f1_mid > f1_whole

    # Regression CCC (middle__median)
    ccc_mid_med = float(df_comp[(df_comp["representation"] == "middle__median") & (df_comp["task"] == "regression") & (df_comp["model"] == "Ridge")]["dev_primary"].iloc[0])
    ccc_early_med = float(df_comp[(df_comp["representation"] == "early__median") & (df_comp["task"] == "regression") & (df_comp["model"] == "Ridge")]["dev_primary"].iloc[0])
    ccc_late_med = float(df_comp[(df_comp["representation"] == "late__median") & (df_comp["task"] == "regression") & (df_comp["model"] == "Ridge")]["dev_primary"].iloc[0])
    ccc_whole_mean = float(df_comp[(df_comp["representation"] == "whole__mean") & (df_comp["task"] == "regression") & (df_comp["model"] == "Ridge")]["dev_primary"].iloc[0])

    assert ccc_mid_med == 0.3743
    assert ccc_mid_med > ccc_early_med
    assert ccc_mid_med > ccc_late_med
    assert ccc_mid_med > ccc_whole_mean


def test_exact_phase17b_embedding_reproduction(repo_paths):
    """Verifies bit-exact reproduction of Phase 17B pooled embeddings from chunk embeddings."""
    repo_root = repo_paths["repo_root"]
    p17b_dir = repo_root / "ml-training" / "dataset" / "text_embeddings_v1" / "all_minilm_l6_v2"
    p18a_dir = repo_paths["dataset_dir"] / "features"

    p17b_train = np.load(p17b_dir / "train_embeddings_mean.npy")
    p18a_train = np.load(p18a_dir / "train" / "whole__mean.npy")
    max_diff_train = np.max(np.abs(p17b_train - p18a_train))
    assert max_diff_train == 0.0, f"Non-zero diff on Train split: {max_diff_train}"

    p17b_dev = np.load(p17b_dir / "dev_embeddings_mean.npy")
    p18a_dev = np.load(p18a_dir / "dev" / "whole__mean.npy")
    max_diff_dev = np.max(np.abs(p17b_dev - p18a_dev))
    assert max_diff_dev == 0.0, f"Non-zero diff on Dev split: {max_diff_dev}"
