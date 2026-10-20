"""
Temporal Sequence Data Loading and Caching Module (E-DAIC Benchmark).

Loads unpooled window-level CNN embeddings (VGG16: 4,096 dims, DenseNet201: 1,920 dims)
preserving exact sequential order, timestamps, and participant-level targets.

Guarantees:
- Validates monotonic timestamp ordering for every participant.
- Fast binary caching (.npz) under dataset/deep_features_v1/temporal_cache/.
- Participant-balanced input normalization (each participant has equal weight).
- Variable-length batch collation with boolean padding masks.
- Official Test split quarantine enforcement.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from core.config import (
        ROOT_DIR,
        DATASET_DIR,
        DEEP_FEATURES_DIR,
        LABELS_DIR,
        FEATURE_REGISTRY,
        RANDOM_SEED,
    )
except ImportError:
    from config import (
        ROOT_DIR,
        DATASET_DIR,
        DEEP_FEATURES_DIR,
        LABELS_DIR,
        FEATURE_REGISTRY,
        RANDOM_SEED,
    )

try:
    from data.split_loader import load_official_splits
except ImportError:
    from split_loader import load_official_splits

TEMPORAL_CACHE_DIR = DEEP_FEATURES_DIR / "temporal_cache"
TEMPORAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_participant_sequence(
    pid: int,
    representation: str,
    participant_inventory: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the raw unpooled window-level embedding sequence and timestamps for a participant.
    Uses cached .npz if available, otherwise parses source CSV once and caches.

    Returns:
        (features_array, timestamps_array)
        where features_array has shape (T, D) and timestamps_array has shape (T,).
    """
    cache_path = TEMPORAL_CACHE_DIR / f"{pid}_{representation}.npz"
    if cache_path.exists():
        with np.load(cache_path) as data:
            return data["X"], data["timestamps"]

    # Locate source CSV from participant inventory
    match = participant_inventory[participant_inventory["participant_id"] == pid]
    if match.empty:
        raise FileNotFoundError(f"Participant {pid} not found in participant inventory.")

    path_col = f"{representation}_path"
    raw_csv_path = Path(match[path_col].values[0])
    if not raw_csv_path.exists():
        # Fallback relative to ROOT_DIR
        raw_csv_path = ROOT_DIR / raw_csv_path

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found for participant {pid}: {raw_csv_path}")

    # Read CSV
    df = pd.read_csv(raw_csv_path)

    # Validate timestamp ordering
    if "timeStamp" not in df.columns:
        raise ValueError(f"Missing timeStamp column in {raw_csv_path}")

    timestamps = df["timeStamp"].to_numpy(dtype=np.int32)
    # Check monotonic increasing
    if not np.all(timestamps[1:] >= timestamps[:-1]):
        raise ValueError(f"Timestamps are not monotonically increasing in {raw_csv_path}")

    # Extract feature columns
    feature_cols = [c for c in df.columns if c.startswith("neuron_")]
    expected_dim = FEATURE_REGISTRY[representation].expected_numeric_dim
    if len(feature_cols) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} features for {representation}, got {len(feature_cols)} in {raw_csv_path}"
        )

    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Cache for subsequent runs
    np.savez_compressed(cache_path, X=X, timestamps=timestamps)
    return X, timestamps


def load_temporal_dataset(
    representation: str,
    split: str,
    labels_dir: Path = LABELS_DIR,
) -> Dict[str, Any]:
    """
    Loads all participant sequences and aligned targets for the requested split.

    Args:
        representation: "vgg16" or "densenet201".
        split: "train" or "dev". (Test split is quarantined!)

    Returns:
        Dict with keys:
            "pids": np.ndarray of shape (N,)
            "sequences": List[np.ndarray] of length N, each (T_i, D)
            "y_binary": np.ndarray of shape (N,)
            "y_score": np.ndarray of shape (N,)
    """
    if split == "test":
        raise PermissionError(
            "Access to the official Test split is strictly quarantined and forbidden in Phase 13B."
        )
    if split not in ["train", "dev"]:
        raise ValueError(f"Invalid split '{split}'. Allowed: 'train', 'dev'.")

    official = load_official_splits(labels_dir=labels_dir)
    p_inv = pd.read_csv(DEEP_FEATURES_DIR / "metadata" / "participant_inventory.csv")

    target_pids = official.train_pids if split == "train" else official.dev_pids
    labels_df = official.labels_df

    pids_list = []
    seqs_list = []
    y_binary_list = []
    y_score_list = []

    for pid in target_pids:
        X, _ = get_participant_sequence(pid, representation, p_inv)
        y_b = int(labels_df.loc[pid, "PHQ_Binary"])
        y_s = float(labels_df.loc[pid, "PHQ_Score"])

        pids_list.append(pid)
        seqs_list.append(X)
        y_binary_list.append(y_b)
        y_score_list.append(y_s)

    return {
        "pids": np.array(pids_list, dtype=np.int64),
        "sequences": seqs_list,
        "y_binary": np.array(y_binary_list, dtype=np.int64),
        "y_score": np.array(y_score_list, dtype=np.float32),
    }


def compute_participant_balanced_scaler(
    sequences: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes participant-balanced mean and standard deviation vectors across a list of sequences.
    Each participant contributes with equal weight 1/N to prevent longer interviews
    from dominating the normalization statistics.

    Args:
        sequences: List of N arrays, each shape (T_i, D).

    Returns:
        (mean_vector, scale_vector), each shape (D,).
    """
    num_participants = len(sequences)
    if num_participants == 0:
        raise ValueError("Cannot compute scaler on empty sequence list.")

    feat_dim = sequences[0].shape[1]

    # 1. Accumulate participant-level means and second moments
    part_means = np.zeros((num_participants, feat_dim), dtype=np.float64)
    part_second_moments = np.zeros((num_participants, feat_dim), dtype=np.float64)

    for i, seq in enumerate(sequences):
        # seq shape: (T_i, D)
        part_means[i] = np.mean(seq, axis=0, dtype=np.float64)
        part_second_moments[i] = np.mean(seq ** 2, axis=0, dtype=np.float64)

    # 2. Balanced global mean across participants (each participant has weight 1/N)
    global_mean = np.mean(part_means, axis=0, dtype=np.float64)

    # 3. Balanced global variance: E[X^2] - (E[X])^2
    global_second_moment = np.mean(part_second_moments, axis=0, dtype=np.float64)
    global_var = np.maximum(global_second_moment - (global_mean ** 2), 1e-8)
    global_scale = np.sqrt(global_var)

    return global_mean.astype(np.float32), global_scale.astype(np.float32)


class TemporalDataset(Dataset):
    """
    PyTorch Dataset providing normalized sequences, participant IDs, and scalar targets.
    """
    def __init__(
        self,
        pids: np.ndarray,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ):
        self.pids = pids
        self.sequences = sequences
        self.targets = targets
        self.scaler_mean = mean if mean is not None else scaler_mean
        self.scaler_scale = scale if scale is not None else scaler_scale

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.sequences[idx]  # (T_i, D)
        if self.scaler_mean is not None and self.scaler_scale is not None:
            seq = (seq - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-7)

        return {
            "seq": torch.tensor(seq, dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "pid": int(self.pids[idx]),
            "length": seq.shape[0],
        }


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Collates a list of variable-length participant sequences into a padded batch.

    Returns:
        padded_x: (B, T_max, D)
        mask: (B, T_max) BoolTensor, True for valid positions, False for padding.
        targets: (B, 1) FloatTensor.
        pids: np.ndarray of shape (B,).
    """
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)
    feat_dim = batch[0]["seq"].shape[1]

    padded_x = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    targets = torch.zeros(batch_size, 1, dtype=torch.float32)
    pids = []

    for i, item in enumerate(batch):
        l = item["length"]
        padded_x[i, :l, :] = item["seq"]
        mask[i, :l] = True
        targets[i, 0] = item["target"]
        pids.append(item["pid"])

    return padded_x, mask, targets, np.array(pids, dtype=np.int64)
