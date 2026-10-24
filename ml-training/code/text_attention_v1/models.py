"""
Neural Attention Pooling Models for Phase 18C.

Implements lightweight additive attention pooling over variable-length sequences
of frozen DistilBERT chunk embeddings (D=768) for participant-level depression estimation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttentionPooling(nn.Module):
    """
    Additive Attention Pooling Module.
    
    Given chunk sequence E_i = [e_i1, ..., e_iK] where e_ij in R^768:
        u_ij = tanh(W_a e_ij + b_a) in R^hidden_dim
        s_ij = w_a^T u_ij in R
        alpha_ij = exp(s_ij + mask_ij) / sum_k exp(s_ik + mask_ik)
        z_i = sum_j alpha_ij e_ij in R^768
    """
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, max_seq_len, embed_dim)
            mask: Tensor of shape (batch_size, max_seq_len), 1 for valid, 0 for pad.
        Returns:
            z: Pooled participant vectors of shape (batch_size, embed_dim)
            alpha: Attention weights of shape (batch_size, max_seq_len), strictly sum to 1 over valid chunks.
        """
        # (batch_size, max_seq_len, hidden_dim)
        u = torch.tanh(self.proj(x))
        # (batch_size, max_seq_len)
        s = self.score(u).squeeze(-1)

        # Masking: set padded positions to -1e9 so exp(-1e9) -> 0
        bool_mask = mask.bool()
        s_masked = s.masked_fill(~bool_mask, -1e9)

        # Softmax over chunk sequence
        alpha = F.softmax(s_masked, dim=-1)

        # Guarantee exact 0 for padded positions and re-normalize for numerical precision
        alpha = alpha * mask.float()
        alpha_sum = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        alpha = alpha / alpha_sum

        # Pooled vector: sum_j alpha_j * e_j -> (batch_size, embed_dim)
        z = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)

        return z, alpha


class AttentionClassifier(nn.Module):
    """
    Participant-level Attention Classifier for PHQ_Binary.
    """
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.pooling = AdditiveAttentionPooling(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: Unnormalized logits of shape (batch_size,)
            alpha: Attention weights of shape (batch_size, max_seq_len)
        """
        z, alpha = self.pooling(x, mask)
        z = self.dropout(z)
        logits = self.classifier(z).squeeze(-1)
        return logits, alpha


class AttentionRegressor(nn.Module):
    """
    Participant-level Attention Regressor for continuous PHQ_Score.
    """
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.pooling = AdditiveAttentionPooling(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            preds: Continuous predictions of shape (batch_size,)
            alpha: Attention weights of shape (batch_size, max_seq_len)
        """
        z, alpha = self.pooling(x, mask)
        z = self.dropout(z)
        preds = self.regressor(z).squeeze(-1)
        return preds, alpha
