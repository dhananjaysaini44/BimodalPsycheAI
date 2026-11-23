"""
MLP Prediction Head Architecture for Phase 17C.

Defines configurable 1-layer and 2-layer Multi-Layer Perceptrons for:
- Binary Classification (PHQ_Binary logits with BCEWithLogitsLoss)
- Continuous Severity Regression (PHQ_Score scalar with MSELoss)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class TextMLPHead(nn.Module):
    """
    Configurable MLP prediction head for frozen transformer representations.
    Supports 1 or 2 hidden layers with LayerNorm, Dropout, and nonlinear activations.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Union[int, List[int]] = 64,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        self.activation_name = activation

        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning shape (batch_size,)."""
        out = self.net(x)
        return out.squeeze(-1)


def save_checkpoint(
    path: Path,
    model: TextMLPHead,
    scaler: StandardScaler,
    task: str,
    config: Dict[str, Any],
    best_epoch: int,
    best_val_metric: float,
    train_history: Optional[Dict[str, Any]] = None,
) -> None:
    """Serializes model weights, fitted scaler, and training metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "in_features": model.in_features,
        "hidden_dims": model.hidden_dims,
        "dropout": model.dropout_rate,
        "use_layer_norm": model.use_layer_norm,
        "activation": model.activation_name,
        "task": task,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_var": scaler.var_,
        "scaler_n_samples_seen": scaler.n_samples_seen_,
        "config": config,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "train_history": train_history,
        "pytorch_version": torch.__version__,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[TextMLPHead, StandardScaler, Dict[str, Any]]:
    """Restores model, fitted scaler, and experiment metadata from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(path, map_location=device, weights_only=False)

    model = TextMLPHead(
        in_features=payload["in_features"],
        hidden_dims=payload["hidden_dims"],
        dropout=payload["dropout"],
        use_layer_norm=payload.get("use_layer_norm", True),
        activation=payload.get("activation", "relu"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = payload["scaler_mean"]
    scaler.scale_ = payload["scaler_scale"]
    scaler.var_ = payload["scaler_var"]
    scaler.n_samples_seen_ = payload["scaler_n_samples_seen"]

    return model, scaler, payload
