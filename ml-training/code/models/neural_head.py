"""
Neural Prediction Head for Deep Audio Embeddings (E-DAIC Benchmark).

Defines a small, regularized Multi-Layer Perceptron (MLP) prediction head:
  Input embedding (D)
      |
  Linear(D, 32)
      |
  LayerNorm(32)
      |
  ReLU
      |
  Dropout(p=0.30)
      |
  Linear(32, 1)

Features training-fitted input standardization, self-contained serialization,
and deterministic inference for both classification (PHQ_Binary) and regression (PHQ_Score).
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class MLPHead(nn.Module):
    """
    Two-layer regularized MLP head with LayerNorm and Dropout.
    Suitable for small sample sizes (N ~ 163) with high input dimensionality (D in 1920..8192).
    """
    def __init__(self, in_features: int, hidden_dim: int = 32, dropout: float = 0.30):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns:
            Tensor of shape (batch_size, 1).
        """
        return self.net(x)


def save_checkpoint(
    path: Path,
    model: MLPHead,
    scaler: StandardScaler,
    task: str,
    config: Dict[str, Any],
    best_epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    train_history: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves a self-contained checkpoint including model weights, fitted scaler statistics,
    and experiment configuration.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "in_features": model.in_features,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout_rate,
        "task": task,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_var": scaler.var_,
        "scaler_n_samples_seen": scaler.n_samples_seen_,
        "config": config,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "train_history": train_history,
        "pytorch_version": torch.__version__,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[MLPHead, StandardScaler, Dict[str, Any]]:
    """
    Loads a saved checkpoint, reinstantiates the MLPHead, restores the fitted StandardScaler,
    and returns (model, scaler, metadata_dict).
    """
    if device is None:
        device = torch.device("cpu")

    payload = torch.load(path, map_location=device, weights_only=False)

    model = MLPHead(
        in_features=payload["in_features"],
        hidden_dim=payload["hidden_dim"],
        dropout=payload["dropout"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = payload["scaler_mean"]
    scaler.scale_ = payload["scaler_scale"]
    scaler.var_ = payload["scaler_var"]
    scaler.n_samples_seen_ = payload["scaler_n_samples_seen"]

    metadata = {
        "task": payload["task"],
        "config": payload.get("config", {}),
        "best_epoch": payload.get("best_epoch", -1),
        "best_metric_name": payload.get("best_metric_name", ""),
        "best_metric_value": payload.get("best_metric_value", 0.0),
        "train_history": payload.get("train_history", None),
        "pytorch_version": payload.get("pytorch_version", ""),
    }

    return model, scaler, metadata


def predict_with_checkpoint(
    model: MLPHead,
    scaler: StandardScaler,
    X: np.ndarray,
    task: str,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Runs deterministic inference using a reloaded model and scaler.
    
    Returns:
        For classification: (preds_binary, probs_continuous)
        For regression: (preds_continuous, None)
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        raw_out = model(X_tensor).squeeze(-1).cpu().numpy()

    if task == "classification":
        # Logits -> probs via sigmoid
        probs = 1.0 / (1.0 + np.exp(-raw_out))
        # Threshold at logit 0.0 <-> prob 0.5
        preds = (raw_out >= 0.0).astype(np.int64)
        return preds, probs
    elif task == "regression":
        # Raw unclipped predictions
        return raw_out, None
    else:
        raise ValueError(f"Unknown task: {task}")
