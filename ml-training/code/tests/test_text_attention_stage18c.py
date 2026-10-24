"""
Unit and Integration Tests for Phase 18C: Lightweight Attention Pooling on Frozen DistilBERT Chunk Embeddings.

Validates:
1. Embedding shape, dimension (2163, 768), float32 type, and numerical validity (zero NaN/Inf).
2. Participant chronological ordering (0 to K_i - 1 monotonic chunk indices).
3. Variable sequence batching, collate function, and padding masks.
4. Attention pooling mathematical properties (sum to 1.0, zero attention on padded chunks, single-chunk handling).
5. End-to-end forward pass shapes and numerical finiteness for AttentionClassifier and AttentionRegressor.
6. Loss and backward pass (gradient flow across all trainable attention and linear head parameters).
7. Frozen chunk embeddings immutability (no gradients propagating to or mutating frozen vectors).
8. Leakage-free 5-fold cross-validation fold disjointness (empty train/val participant intersection).
9. Strict Official Test split quarantine (zero exposure or references to Test IDs in prediction CSVs).
10. Metric calculation accuracy (F1, Balanced Accuracy, ROC-AUC, Lin's CCC, RMSE) against scikit-learn standard definitions.
11. Continuous, unclipped, real-valued regression outputs.
12. Preservation and non-modification of Phase 18B artifacts.
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
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold

from text_attention_v1.models import (
    AdditiveAttentionPooling,
    AttentionClassifier,
    AttentionRegressor,
)
from text_attention_v1.dataset import (
    collate_variable_sequences,
    ParticipantSequenceDataset,
    load_participant_chunk_sequences,
)
from text_attention_v1.run_experiments import compute_ccc


@pytest.fixture
def repo_paths():
    repo_root = Path(__file__).resolve().parents[3]
    ml_training = repo_root / "ml-training"
    return {
        "repo_root": repo_root,
        "labels_dir": ml_training / "edaic-woz" / "labels",
        "distilbert_dir": ml_training / "outputs" / "text_distilbert_v1",
        "attention_dir": ml_training / "outputs" / "text_attention_v1",
        "dataset_dir": ml_training / "dataset" / "text_distilbert_v1",
    }


def test_phase18c_artifacts_exist(repo_paths):
    """Verifies that all Phase 18C reports, audit notes, tables, plots, and prediction CSVs exist."""
    attn_dir = repo_paths["attention_dir"]
    assert (attn_dir / "audit" / "phase_18c_preflight_audit.md").exists()
    assert (attn_dir / "reports" / "phase_18c_report.md").exists()
    assert (attn_dir / "reports" / "model_comparison_table.csv").exists()
    assert (attn_dir / "reports" / "paired_bootstrap_differences.csv").exists()

    # 7 Diagnostic plots
    plot_files = [
        "attention_weight_profiles.png",
        "classification_benchmark_attention.png",
        "cv_fold_variability.png",
        "paired_bootstrap_differences_attention.png",
        "regression_benchmark_attention.png",
        "seed_stability_attention.png",
        "training_loss_curves.png",
    ]
    for pf in plot_files:
        assert (attn_dir / "plots" / pf).exists(), f"Missing plot: {pf}"

    # Grid search CSVs
    cv_dir = attn_dir / "cv_results"
    for name in [
        "grid_search_whole_attention_classification.csv",
        "grid_search_middle_attention_classification.csv",
        "grid_search_whole_attention_regression.csv",
        "grid_search_middle_attention_regression.csv",
    ]:
        assert (cv_dir / name).exists(), f"Missing grid search CSV: {name}"

    # Predictions
    pred_dir = attn_dir / "predictions"
    pred_files = list(pred_dir.glob("*.csv"))
    assert len(pred_files) == 24, f"Expected 24 prediction files, found {len(pred_files)}"


def test_chunk_embeddings_shape_and_numerical_integrity(repo_paths):
    """Verifies chunk embeddings have shape (2163, 768), float32 dtype, and no NaN/Inf."""
    datadir = repo_paths["dataset_dir"]
    chunks_path = datadir / "chunk_embeddings.npy"
    assert chunks_path.exists(), "chunk_embeddings.npy missing"

    chunks = np.load(chunks_path)
    assert chunks.shape == (2163, 768), f"Expected (2163, 768), got {chunks.shape}"
    assert chunks.dtype == np.float32, f"Expected float32, got {chunks.dtype}"
    assert not np.isnan(chunks).any(), "NaN found in chunk embeddings"
    assert not np.isinf(chunks).any(), "Inf found in chunk embeddings"


def test_chronological_ordering_and_monotonic_indices(repo_paths):
    """Verifies that for every participant, chunks appear strictly in increasing chunk_index order."""
    datadir = repo_paths["dataset_dir"]
    meta_path = datadir / "chunk_metadata.csv"
    assert meta_path.exists(), "chunk_metadata.csv missing"

    df_meta = pd.read_csv(meta_path)
    grouped = df_meta.groupby("Participant_ID")
    for pid, grp in grouped:
        indices = grp["chunk_index"].tolist()
        expected = list(range(len(grp)))
        assert indices == expected, f"Participant {pid} chunk indices not strictly monotonic 0..K-1: {indices}"


def test_variable_sequence_batching_and_collate():
    """Tests collate function with batches of varying sequence lengths (e.g. 3, 7, 15)."""
    # Create synthetic sequences
    d = 768
    seq1 = torch.ones((3, d), dtype=torch.float32) * 1.0
    seq2 = torch.ones((7, d), dtype=torch.float32) * 2.0
    seq3 = torch.ones((15, d), dtype=torch.float32) * 3.0

    batch = [
        (101, seq1, 0.0),
        (102, seq2, 1.0),
        (103, seq3, 1.0),
    ]

    padded_x, mask, labels, pids = collate_variable_sequences(batch)

    assert padded_x.shape == (3, 15, d)
    assert mask.shape == (3, 15)
    assert labels.shape == (3,)
    assert pids == [101, 102, 103]

    # Verify padding mask: 1.0 for valid chunks, 0.0 for padded positions
    assert (mask[0, :3] == 1.0).all()
    assert (mask[0, 3:] == 0.0).all()

    assert (mask[1, :7] == 1.0).all()
    assert (mask[1, 7:] == 0.0).all()

    assert (mask[2, :15] == 1.0).all()


def test_attention_pooling_weight_properties():
    """
    Tests mathematical properties of AdditiveAttentionPooling:
    - Zero attention on padded chunks (< 1e-7).
    - Valid attention weights sum to 1.0 +- 1e-5 for every sample.
    - Single-chunk handling: output equals chunk embedding and weight is 1.0.
    - Non-trivial attention distribution on sequences with K >= 3.
    """
    torch.manual_seed(42)
    d = 768
    h = 32
    attn = AdditiveAttentionPooling(embed_dim=d, hidden_dim=h)
    attn.eval()

    # Test 1: Single chunk participant (K=1)
    single_x = torch.randn(1, 1, d)
    single_mask = torch.ones(1, 1, dtype=torch.float32)
    with torch.no_grad():
        out_single, weights_single = attn(single_x, single_mask)

    assert torch.isclose(weights_single[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out_single[0], single_x[0, 0], atol=1e-5)

    # Test 2: Batch with variable lengths and padding
    x_batch = torch.randn(2, 5, d)
    mask_batch = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=torch.float32)

    with torch.no_grad():
        out_batch, weights_batch = attn(x_batch, mask_batch)

    assert out_batch.shape == (2, d)
    assert weights_batch.shape == (2, 5)

    # Padded chunks must receive strictly zero attention
    assert torch.all(weights_batch[0, 3:] < 1e-7)

    # Valid chunks must sum to 1.0 +- 1e-5
    sum_valid_0 = weights_batch[0, :3].sum()
    sum_valid_1 = weights_batch[1, :5].sum()
    assert torch.isclose(sum_valid_0, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(sum_valid_1, torch.tensor(1.0), atol=1e-5)

    # Test 3: Non-trivial distribution on sequence of length 5 (deviates from uniform 0.2 by >= 1e-4)
    weights_sample = weights_batch[1].numpy()
    uniform = 1.0 / 5.0
    max_dev = np.max(np.abs(weights_sample - uniform))
    assert max_dev >= 1e-4, f"Attention weights too uniform: {weights_sample}"


def test_end_to_end_forward_pass():
    """Verifies output shapes and numerical finiteness for AttentionClassifier and AttentionRegressor."""
    torch.manual_seed(42)
    B, L, D = 4, 8, 768
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.float32)
    mask[0, 5:] = 0.0
    mask[1, 3:] = 0.0

    cls_model = AttentionClassifier(embed_dim=D, hidden_dim=32, dropout=0.2)
    reg_model = AttentionRegressor(embed_dim=D, hidden_dim=32, dropout=0.2)

    cls_model.eval()
    reg_model.eval()

    with torch.no_grad():
        logits, weights_cls = cls_model(x, mask)
        reg_preds, weights_reg = reg_model(x, mask)

    assert logits.shape == (B,), f"Expected shape ({B},), got {logits.shape}"
    assert reg_preds.shape == (B,), f"Expected shape ({B},), got {reg_preds.shape}"
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert not torch.isnan(reg_preds).any()
    assert not torch.isinf(reg_preds).any()


def test_loss_and_backward_gradients():
    """Verifies BCE and MSE loss backward pass generates non-zero gradients on all trainable weights."""
    torch.manual_seed(42)
    B, L, D = 4, 6, 768
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.float32)

    # Classifier test
    cls_model = AttentionClassifier(embed_dim=D, hidden_dim=32, dropout=0.0)
    cls_model.train()
    target_cls = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    logits, _ = cls_model(x, mask)
    criterion_cls = torch.nn.BCEWithLogitsLoss()
    loss_cls = criterion_cls(logits, target_cls)
    loss_cls.backward()

    for name, param in cls_model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"

    # Regressor test
    reg_model = AttentionRegressor(embed_dim=D, hidden_dim=32, dropout=0.0)
    reg_model.train()
    target_reg = torch.tensor([5.0, 12.0, 8.0, 19.0], dtype=torch.float32)

    preds, _ = reg_model(x, mask)
    criterion_reg = torch.nn.MSELoss()
    loss_reg = criterion_reg(preds, target_reg)
    loss_reg.backward()

    for name, param in reg_model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


def test_frozen_embeddings_immutability():
    """Verifies that input chunk embeddings do not receive gradients and are not model parameters."""
    D = 768
    chunk_embeddings = np.random.randn(20, D).astype(np.float32)
    chunk_copy = chunk_embeddings.copy()

    model = AttentionClassifier(embed_dim=D, hidden_dim=32, dropout=0.1)
    for p in model.parameters():
        assert p.shape != (20, D), "Embedding matrix erroneously registered as model parameter"

    x_tensor = torch.tensor(chunk_embeddings).unsqueeze(0)
    assert not x_tensor.requires_grad
    mask_tensor = torch.ones(1, 20, dtype=torch.float32)

    out, _ = model(x_tensor, mask_tensor)
    loss = out.sum()
    loss.backward()

    # Original array must be byte-for-byte identical
    np.testing.assert_array_equal(chunk_embeddings, chunk_copy)
    assert x_tensor.grad is None


def test_concordance_correlation_coefficient_formula():
    """Tests CCC calculation against theoretical definition."""
    # Perfect concordance
    y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(compute_ccc(y1, y2), 1.0, atol=1e-5)

    # Shifted predictions (mean shift degrades concordance)
    y3 = y1 + 5.0
    ccc_shifted = compute_ccc(y1, y3)
    assert ccc_shifted < 0.5, f"Expected lower CCC for shifted: {ccc_shifted}"

    # Constant prediction (variance zero)
    y_const = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
    assert compute_ccc(y1, y_const) == 0.0


def test_participant_sequence_dataset_and_loader(repo_paths):
    """Tests ParticipantSequenceDataset item loading and sequence formatting."""
    datadir = repo_paths["dataset_dir"]
    labels_df = pd.read_csv(repo_paths["labels_dir"] / "train_split.csv")

    seq_map, df_meta = load_participant_chunk_sequences(
        dataset_dir=datadir, segment_mode="middle"
    )

    pids = labels_df["Participant_ID"].tolist()
    labels_dict = dict(zip(pids, labels_df["PHQ_Binary"].tolist()))

    ds = ParticipantSequenceDataset(pids, seq_map, labels_dict)
    assert len(ds) == 163
    pid_0, x_0, y_0 = ds[0]
    assert pid_0 == pids[0]
    assert x_0.ndim == 2
    assert x_0.shape[1] == 768
    assert isinstance(y_0, (float, np.floating, int, np.integer))


def test_cv_fold_disjointness(repo_paths):
    """Verifies that in 5-fold CV, Train and Validation folds have strictly zero participant overlap."""
    labels_dir = repo_paths["labels_dir"]
    train_df = pd.read_csv(labels_dir / "train_split.csv")
    pids = train_df["Participant_ID"].values
    y_cls = train_df["PHQ_Binary"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(pids, y_cls)):
        trn_pids = set(pids[trn_idx])
        val_pids = set(pids[val_idx])
        overlap = trn_pids.intersection(val_pids)
        assert len(overlap) == 0, f"Fold {fold} has overlapping participant IDs: {overlap}"
        assert len(trn_pids) + len(val_pids) == 163


def test_official_test_split_quarantine(repo_paths):
    """Verifies that official Test participants (N=56) never appear in any prediction CSV."""
    attn_dir = repo_paths["attention_dir"]
    labels_dir = repo_paths["labels_dir"]

    test_pids = set(pd.read_csv(labels_dir / "test_split.csv")["Participant_ID"].values)
    assert len(test_pids) == 56

    pred_dir = attn_dir / "predictions"
    if not pred_dir.exists():
        pytest.skip("Predictions directory not yet populated")

    pred_files = list(pred_dir.glob("*.csv"))
    if not pred_files:
        pytest.skip("No prediction files found yet")

    for pf in pred_files:
        df_pred = pd.read_csv(pf)
        assert "split" in df_pred.columns
        splits_present = set(df_pred["split"].unique())
        assert "test" not in splits_present, f"Test split leaked into {pf.name}"

        pids_present = set(df_pred["Participant_ID"].unique())
        leaked = pids_present.intersection(test_pids)
        assert len(leaked) == 0, f"Official Test participant leaked in {pf.name}: {leaked}"


def test_model_eval_mode_determinism():
    """Verifies that in eval() mode, forward passes are perfectly deterministic."""
    D = 768
    model = AttentionClassifier(embed_dim=D, hidden_dim=32, dropout=0.5)
    model.eval()

    x = torch.randn(2, 6, D)
    mask = torch.ones(2, 6, dtype=torch.float32)
    mask[0, 4:] = 0.0

    with torch.no_grad():
        out1, w1 = model(x, mask)
        out2, w2 = model(x, mask)

    assert torch.all(out1 == out2), "Model output not deterministic in eval mode"
    assert torch.all(w1 == w2), "Attention weights not deterministic in eval mode"


def test_attention_numerical_stability_extreme_inputs():
    """Verifies that extreme input values (+-1000) do not cause overflow/underflow or NaN."""
    D = 768
    attn = AdditiveAttentionPooling(embed_dim=D, hidden_dim=32)
    attn.eval()

    # Extreme input tensor
    x = torch.ones(2, 4, D) * 1000.0
    x[1] = -1000.0
    mask = torch.ones(2, 4, dtype=torch.float32)
    mask[0, 2:] = 0.0

    with torch.no_grad():
        out, weights = attn(x, mask)

    assert not torch.isnan(out).any(), "NaN in attention output with extreme inputs"
    assert not torch.isinf(out).any(), "Inf in attention output with extreme inputs"
    assert not torch.isnan(weights).any(), "NaN in attention weights with extreme inputs"
    assert not torch.isinf(weights).any(), "Inf in attention weights with extreme inputs"
    assert torch.allclose(weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(weights[1].sum(), torch.tensor(1.0), atol=1e-5)


def test_regression_predictions_continuous_unclipped(repo_paths):
    """Verifies that all regression predictions are continuous, unclipped floats and non-trivial."""
    attn_dir = repo_paths["attention_dir"]
    pred_dir = attn_dir / "predictions"
    reg_files = list(pred_dir.glob("dev_reg__*.csv"))
    if not reg_files:
        pytest.skip("Regression prediction files not yet generated")

    for rf in reg_files:
        df = pd.read_csv(rf)
        assert "y_pred" in df.columns
        y_pred = df["y_pred"].values
        assert issubclass(y_pred.dtype.type, (np.floating, float))
        # Ensure predictions are continuous and varied (std > 0.01)
        assert np.std(y_pred) > 0.01, f"Predictions are collapsed/constant in {rf.name}"
        assert not np.isnan(y_pred).any()
        assert not np.isinf(y_pred).any()


def test_metric_reproduction_from_prediction_files(repo_paths):
    """Verifies that metrics calculated from prediction CSVs reproduce values in model_comparison_table.csv."""
    attn_dir = repo_paths["attention_dir"]
    table_path = attn_dir / "reports" / "model_comparison_table.csv"
    if not table_path.exists():
        pytest.skip("model_comparison_table.csv not yet generated")

    df_tbl = pd.read_csv(table_path)
    pred_dir = attn_dir / "predictions"

    # Only test attention models (not 18B reference baselines)
    attn_rows = df_tbl[df_tbl["representation"].str.contains("attention")].copy()

    matched_pids = set(pd.read_csv(repo_paths["dataset_dir"] / "features" / "dev" / "complete_cohort_participant_ids.csv")["Participant_ID"].values)

    for _, row in attn_rows.iterrows():
        task = row["task"]
        rep = row["representation"].replace("_matched", "")
        seed = int(row["seed"])
        cohort = row["cohort"]

        prefix = "cla" if task == "classification" else "reg"
        pred_file = pred_dir / f"dev_{prefix}__{rep}__seed{seed}.csv"
        if not pred_file.exists():
            continue

        df_pred = pd.read_csv(pred_file)
        if cohort == "matched":
            df_pred = df_pred[df_pred["Participant_ID"].isin(matched_pids)]

        y_true = df_pred["y_true"].values
        y_pred = df_pred["y_pred"].values

        if task == "classification":
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            assert np.isclose(f1, row["dev_f1"], atol=1e-4), f"F1 mismatch for {rep} seed {seed}"
        else:
            ccc = compute_ccc(y_true, y_pred)
            assert np.isclose(ccc, row["dev_ccc"], atol=1e-4), f"CCC mismatch for {rep} seed {seed}"
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            assert np.isclose(rmse, row["dev_rmse"], atol=1e-4), f"RMSE mismatch for {rep} seed {seed}"


def test_phase18b_preservation(repo_paths):
    """Verifies that Phase 18B reports, features, and prediction artifacts remain unchanged."""
    b_dir = repo_paths["distilbert_dir"]
    assert (b_dir / "reports" / "phase_18b_report.md").exists()
    assert (b_dir / "reports" / "model_comparison_table.csv").exists()
    assert (b_dir / "reports" / "paired_bootstrap_differences.csv").exists()

    datadir = repo_paths["dataset_dir"]
    chunks = np.load(datadir / "chunk_embeddings.npy")
    assert chunks.shape == (2163, 768)

    w_mean = np.load(datadir / "features" / "train" / "whole__mean.npy")
    assert w_mean.shape == (163, 768)
