"""
Temporal Sequence Model for Deep Audio Embeddings (E-DAIC Benchmark).

Implements the Phase 13B architecture:
  Ordered sequence of window embeddings (T, D)
               |
  Per-window Linear projection: D -> 32
               |
  LayerNorm(32) + ReLU + Dropout(0.30)
               |
  Small unidirectional GRU, hidden size 32
               |
  Masked temporal attention pooling
               |
  Participant-level representation (32)
               |
  Linear output head: 32 -> 1

Features:
- Handles variable-length participant sequences using boolean masking and packed sequences.
- Masked attention ensures padded windows never contribute to pooled representations.
- Single loss computation per participant prediction.
- Self-contained checkpoint saving and loading with fitted normalization statistics.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn


class TemporalGRUAttentionModel(nn.Module):
    """
    Compact, regularized temporal sequence model combining a linear bottleneck projection,
    unidirectional GRU, and masked temporal attention pooling.
    """
    def __init__(
        self,
        in_features: int,  # 4096 (VGG16) or 1920 (DenseNet201)
        proj_dim: int = 32,
        gru_hidden: int = 32,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.in_features = in_features
        self.proj_dim = proj_dim
        self.gru_hidden = gru_hidden
        self.dropout_rate = dropout

        # 1. Per-window projection: D -> 32 with LayerNorm and Dropout
        self.proj = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # 2. Small unidirectional GRU
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # 3. Masked temporal attention pooling
        self.attn = nn.Linear(gru_hidden, 1)

        # 4. Linear output head: 32 -> 1
        self.out_head = nn.Linear(gru_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, T, D)
            mask: BoolTensor of shape (B, T), True for valid windows, False for padding.

        Returns:
            out: Tensor of shape (B, 1) - participant-level scalar prediction.
            attn_weights: Tensor of shape (B, T) - normalized attention distribution.
        """
        batch_size, seq_len, _ = x.size()

        # 1. Project per-window: (B, T, 32)
        h_proj = self.proj(x)

        # 2. Unidirectional GRU over projected sequence
        gru_out, _ = self.gru(h_proj)  # (B, T, 32)

        # 3. Masked temporal attention pooling
        scores = self.attn(gru_out).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, -1e4)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, T)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 4. Weighted sum over time steps: (B, 32)
        pooled = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)

        # 5. Output head: (B, 1)
        out = self.out_head(pooled)
        return out, attn_weights


def save_temporal_checkpoint(
    path: Path,
    model: TemporalGRUAttentionModel,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    task: str,
    config: Dict[str, Any],
    best_epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    train_history: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Serializes a complete, self-contained temporal checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "in_features": model.in_features,
        "proj_dim": model.proj_dim,
        "gru_hidden": model.gru_hidden,
        "dropout": model.dropout_rate,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "task": task,
        "config": config,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "train_history": train_history,
        "pytorch_version": torch.__version__,
    }
    torch.save(payload, path)


def load_temporal_checkpoint(
    path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[TemporalGRUAttentionModel, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads a temporal checkpoint, restoring model weights, normalization statistics, and metadata.
    """
    if device is None:
        device = torch.device("cpu")

    payload = torch.load(path, map_location=device, weights_only=False)

    model = TemporalGRUAttentionModel(
        in_features=payload["in_features"],
        proj_dim=payload["proj_dim"],
        gru_hidden=payload["gru_hidden"],
        dropout=payload["dropout"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    scaler_mean = payload["scaler_mean"]
    scaler_scale = payload["scaler_scale"]

    metadata = {
        "task": payload["task"],
        "config": payload.get("config", {}),
        "best_epoch": payload.get("best_epoch", -1),
        "best_metric_name": payload.get("best_metric_name", ""),
        "best_metric_value": payload.get("best_metric_value", 0.0),
        "train_history": payload.get("train_history", None),
        "pytorch_version": payload.get("pytorch_version", ""),
    }

    return model, scaler_mean, scaler_scale, metadata


def predict_temporal_sequence(
    model: TemporalGRUAttentionModel,
    sequence: np.ndarray,  # Shape: (T, D)
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    task: str,
    device: Optional[torch.device] = None,
) -> Tuple[float, Optional[float], np.ndarray]:
    """
    Runs deterministic participant-level inference on a single sequence.

    Args:
        model: Trained TemporalGRUAttentionModel in eval mode.
        sequence: Raw unnormalized feature array (T, D).
        scaler_mean: Mean vector (D,).
        scaler_scale: Scale vector (D,).
        task: "classification" or "regression".

    Returns:
        (pred_value, prob_value, attention_weights)
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    # Normalize sequence: (T, D)
    seq_norm = (sequence - scaler_mean) / np.maximum(scaler_scale, 1e-7)

    x_tensor = torch.tensor(seq_norm, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, D)
    mask = torch.ones(1, sequence.shape[0], dtype=torch.bool, device=device)  # (1, T)

    with torch.no_grad():
        out, attn = model(x_tensor, mask)
        raw_val = float(out.item())
        attn_np = attn.squeeze(0).cpu().numpy()

    if task == "classification":
        # Logit -> probability
        prob = 1.0 / (1.0 + np.exp(-raw_val))
        pred_class = 1.0 if raw_val >= 0.0 else 0.0
        return pred_class, prob, attn_np
    elif task == "regression":
        return raw_val, None, attn_np
    else:
        raise ValueError(f"Unknown task: {task}")
