"""
Regression and validation test suite for Phase 13B: Temporal Sequence Modeling.

Tests:
1. Architecture and shape contracts for TemporalGRUAttentionModel.
2. Masked attention integrity (padded windows receive zero attention weight).
3. Participant-balanced normalization scaling (equal 1/N weighting).
4. Variable-length batch collation and masking.
5. Strict quarantine of official Test split.
6. Complete self-contained checkpoint serialization and exact prediction reproduction.
"""

import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core.config import LABELS_DIR, ROOT_DIR
from models.temporal_model import (
    TemporalGRUAttentionModel,
    save_temporal_checkpoint,
    load_temporal_checkpoint,
    predict_temporal_sequence,
)
from data.temporal_loader import (
    compute_participant_balanced_scaler,
    TemporalDataset,
    pad_collate_fn,
    load_temporal_dataset,
)


def test_temporal_model_forward_and_shapes():
    """Verify forward pass output dimensions and attention mask compliance."""
    model = TemporalGRUAttentionModel(
        in_features=4096,
        proj_dim=32,
        gru_hidden=32,
        dropout=0.30,
    )
    model.eval()

    batch_size = 2
    max_len = 50
    x = torch.randn(batch_size, max_len, 4096)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    
    # Participant 1 has only 30 valid windows
    mask[1, 30:] = False

    with torch.no_grad():
        out, attn = model(x, mask)

    assert out.shape == (batch_size, 1), f"Expected (2, 1), got {out.shape}"
    assert attn.shape == (batch_size, max_len), f"Expected (2, 50), got {attn.shape}"
    assert not torch.isnan(out).any(), "NaN detected in model outputs"
    assert not torch.isnan(attn).any(), "NaN detected in attention weights"

    # Verify attention sums to 1.0
    attn_sums = attn.sum(dim=1)
    assert torch.allclose(attn_sums, torch.ones(batch_size), atol=1e-5), "Attention weights do not sum to 1"

    # Verify padded positions receive 0 attention weight
    padded_weights = attn[1, 30:]
    assert torch.allclose(padded_weights, torch.zeros_like(padded_weights), atol=1e-6), (
        "Padded timesteps received non-zero attention weight!"
    )


def test_participant_balanced_scaler():
    """Verify that scaler weights participants equally (1/N) regardless of sequence length."""
    # Seq 1: length 1000, all values 10.0
    # Seq 2: length 10, all values 2.0
    # Under participant-balanced scaling, mean must be (10.0 + 2.0) / 2 = 6.0
    # (Under window-pooled scaling, it would be heavily biased towards 10.0: ~9.92)
    s1 = np.full((1000, 4), 10.0, dtype=np.float32)
    s2 = np.full((10, 4), 2.0, dtype=np.float32)

    mean, scale = compute_participant_balanced_scaler([s1, s2])

    assert np.allclose(mean, np.full(4, 6.0), atol=1e-5), f"Expected mean 6.0, got {mean}"
    assert (scale > 0).all(), "Scales must be strictly positive"


def test_pad_collate_fn():
    """Verify dynamic sequence batch collation and mask generation."""
    item1 = {
        "seq": torch.randn(15, 64),
        "target": torch.tensor(1.0),
        "pid": 101,
        "length": 15,
    }
    item2 = {
        "seq": torch.randn(25, 64),
        "target": torch.tensor(0.0),
        "pid": 102,
        "length": 25,
    }

    batch_x, batch_mask, batch_y, batch_pids = pad_collate_fn([item1, item2])

    assert batch_x.shape == (2, 25, 64)
    assert batch_mask.shape == (2, 25)
    assert batch_y.shape == (2, 1)
    assert list(batch_pids) == [101, 102]

    # Verify mask integrity
    assert batch_mask[0, :15].all()
    assert not batch_mask[0, 15:].any()
    assert batch_mask[1].all()


def test_official_test_quarantine():
    """Verify that calling load_temporal_dataset for test split raises PermissionError."""
    with pytest.raises(PermissionError) as exc_info:
        load_temporal_dataset("vgg16", "test", labels_dir=LABELS_DIR)
    assert "quarantined" in str(exc_info.value).lower()


def test_checkpoint_roundtrip_fidelity():
    """Verify serialization and deserialization reproduce identical predictions."""
    model = TemporalGRUAttentionModel(in_features=1920, proj_dim=32, gru_hidden=32, dropout=0.30)
    model.eval()

    dummy_mean = np.zeros(1920, dtype=np.float32)
    dummy_scale = np.ones(1920, dtype=np.float32)
    dummy_config = {"proj_dim": 32, "gru_hidden": 32, "dropout": 0.30}

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pt"
        save_temporal_checkpoint(
            path=ckpt_path,
            model=model,
            scaler_mean=dummy_mean,
            scaler_scale=dummy_scale,
            task="classification",
            config=dummy_config,
            best_epoch=12,
            best_metric_name="roc_auc",
            best_metric_value=0.825,
        )

        loaded_model, loaded_mean, loaded_scale, payload = load_temporal_checkpoint(ckpt_path)

        assert payload["task"] == "classification"
        assert payload["best_epoch"] == 12
        assert np.allclose(dummy_mean, loaded_mean)
        assert np.allclose(dummy_scale, loaded_scale)

        # Compare inference on a random raw sequence
        raw_seq = np.random.randn(45, 1920).astype(np.float32)
        p1, prob1, a1 = predict_temporal_sequence(model, raw_seq, dummy_mean, dummy_scale, "classification")
        p2, prob2, a2 = predict_temporal_sequence(loaded_model, raw_seq, loaded_mean, loaded_scale, "classification")

        assert np.isclose(p1, p2, atol=1e-5), f"Class prediction mismatch: {p1} vs {p2}"
        assert np.isclose(prob1, prob2, atol=1e-5), f"Probability mismatch: {prob1} vs {prob2}"
        assert np.allclose(a1, a2, atol=1e-5), "Attention distribution mismatch"
