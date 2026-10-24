"""
Dataset loading and batch collation for Phase 18C.

Extracts participant-level chunk sequences from cached DistilBERT chunk embeddings
and prepares variable-length padded batches for additive attention pooling.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ParticipantSequenceDataset(Dataset):
    """
    Dataset wrapping participant-level chunk sequences.
    """
    def __init__(
        self,
        participant_ids: List[int],
        chunk_dict: Dict[int, np.ndarray],
        labels_dict: Dict[int, float],
    ):
        self.participant_ids = [pid for pid in participant_ids if pid in chunk_dict]
        self.chunk_dict = chunk_dict
        self.labels_dict = labels_dict

    def __len__(self) -> int:
        return len(self.participant_ids)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, float]:
        pid = self.participant_ids[idx]
        chunks = self.chunk_dict[pid] # (K_i, 768)
        label = self.labels_dict[pid]
        return pid, torch.from_numpy(chunks).float(), float(label)


def collate_variable_sequences(
    batch: List[Tuple[int, torch.Tensor, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Collate function for variable-length chunk sequences.
    Pads sequences with zeros to max length in the batch.
    Creates attention mask (1 for valid chunks, 0 for padded positions).
    
    Returns:
        padded_x: (B, max_len, D) float32 tensor
        mask: (B, max_len) float32 tensor
        labels: (B,) float32 tensor
        pids: List[int]
    """
    pids = [item[0] for item in batch]
    tensors = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)

    lengths = [t.shape[0] for t in tensors]
    max_len = max(lengths)
    embed_dim = tensors[0].shape[1]
    batch_size = len(batch)

    padded_x = torch.zeros(batch_size, max_len, embed_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, t in enumerate(tensors):
        seq_len = t.shape[0]
        padded_x[i, :seq_len] = t
        mask[i, :seq_len] = 1.0

    return padded_x, mask, labels, pids


def load_participant_chunk_sequences(
    dataset_dir: Path,
    segment_mode: str = "whole",
) -> Tuple[Dict[int, np.ndarray], pd.DataFrame]:
    """
    Loads raw chunk embeddings and organizes them into chronologically ordered
    arrays for each participant.
    
    Args:
        dataset_dir: Directory containing chunk_embeddings.npy and chunk_metadata.csv.
        segment_mode: "whole" (all chunks) or "middle" (middle tertile only).
        
    Returns:
        participant_chunks: Dict mapping participant_id -> (K_i, 768) float32 array
        df_meta: Chunk metadata DataFrame
    """
    embeddings = np.load(dataset_dir / "chunk_embeddings.npy")
    df_meta = pd.read_csv(dataset_dir / "chunk_metadata.csv")

    assert len(embeddings) == len(df_meta), "Chunk embeddings and metadata row count mismatch"

    participant_chunks = {}

    for pid, group in df_meta.groupby("Participant_ID", sort=False):
        group_sorted = group.sort_values("chunk_index")
        indices = group_sorted.index.to_numpy()
        K_i = len(indices)

        if segment_mode == "whole":
            selected_indices = indices
        elif segment_mode == "middle":
            # Rule: K_i / 3 <= k < 2 * K_i / 3
            mid_ranks = [k for k in range(K_i) if (K_i / 3.0) <= k < (2.0 * K_i / 3.0)]
            selected_indices = indices[mid_ranks]
        else:
            raise ValueError(f"Unsupported segment_mode: {segment_mode}")

        if len(selected_indices) > 0:
            participant_chunks[pid] = embeddings[selected_indices].astype(np.float32)

    return participant_chunks, df_meta


def load_targets(
    labels_dir: Path,
    task: str = "classification",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads Train and Dev target labels.
    
    Args:
        labels_dir: Path to edaic-woz/labels
        task: "classification" (PHQ_Binary) or "regression" (PHQ_Score)
        
    Returns:
        df_train: DataFrame with Participant_ID and target
        df_dev: DataFrame with Participant_ID and target
    """
    df_train = pd.read_csv(labels_dir / "train_split.csv")
    df_dev = pd.read_csv(labels_dir / "dev_split.csv")

    target_col = "PHQ_Binary" if task == "classification" else "PHQ_Score"

    df_train = df_train[["Participant_ID", target_col]].rename(columns={target_col: "target"})
    df_dev = df_dev[["Participant_ID", target_col]].rename(columns={target_col: "target"})

    return df_train, df_dev
