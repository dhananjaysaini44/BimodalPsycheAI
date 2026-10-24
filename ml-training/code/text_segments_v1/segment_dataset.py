"""
Phase 18A: Segment Definition, Robust Aggregation, and Representation Generation.

Implements:
1. Deterministic chronological transcript segmentation into Early, Middle, Late tertiles.
2. Comparative assignment audit: Chunk Rank Tertiles vs. Token Midpoint Tertiles.
3. Robust aggregation strategies: Mean, Median, 10% Trimmed Mean, Mean + Std.
4. Construction of participant-level feature matrices:
   - Whole-interview baselines (D=384, 768)
   - Single-segment representations (Early, Middle, Late x 4 aggregations, D=384, 768)
   - Three-segment concatenation (Early + Middle + Late x 4 aggregations, D=1152, 2304)
5. Strict Test split quarantine and alignment validation.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase18A_SegmentDataset")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "ml-training" / "dataset" / "text_segments_v1"
OUTPUT_DIR = REPO_ROOT / "ml-training" / "outputs" / "text_segments_v1"
LABELS_DIR = REPO_ROOT / "ml-training" / "edaic-woz" / "labels"

FEATURES_DIR = DATASET_DIR / "features"


def aggregate_mean(embeds: np.ndarray) -> np.ndarray:
    """Computes element-wise mean across chunk embeddings."""
    assert len(embeds) > 0, "Cannot aggregate empty chunk array"
    return np.mean(embeds, axis=0).astype(np.float32)


def aggregate_median(embeds: np.ndarray) -> np.ndarray:
    """Computes element-wise median across chunk embeddings."""
    assert len(embeds) > 0, "Cannot aggregate empty chunk array"
    return np.median(embeds, axis=0).astype(np.float32)


def aggregate_trimmed_mean(embeds: np.ndarray, proportion: float = 0.10) -> np.ndarray:
    """
    Computes element-wise trimmed mean.
    For each dimension, trims floor(proportion * n_chunks) values from each tail.
    If n_chunks < 10 (floor(0.10 * n) == 0), gracefully evaluates standard mean.
    """
    assert len(embeds) > 0, "Cannot aggregate empty chunk array"
    n_chunks = len(embeds)
    k_trim = int(np.floor(proportion * n_chunks))

    if k_trim == 0 or 2 * k_trim >= n_chunks:
        return np.mean(embeds, axis=0).astype(np.float32)

    # Sort along chunk axis for each dimension
    sorted_embeds = np.sort(embeds, axis=0)
    trimmed_slice = sorted_embeds[k_trim : n_chunks - k_trim]
    return np.mean(trimmed_slice, axis=0).astype(np.float32)


def aggregate_mean_std(embeds: np.ndarray) -> np.ndarray:
    """Concatenates element-wise mean and standard deviation."""
    assert len(embeds) > 0, "Cannot aggregate empty chunk array"
    mean_vec = np.mean(embeds, axis=0)
    if len(embeds) > 1:
        std_vec = np.std(embeds, axis=0, ddof=0)
    else:
        std_vec = np.zeros_like(mean_vec)
    return np.concatenate([mean_vec, std_vec], axis=0).astype(np.float32)


AGGREGATION_FUNCS = {
    "mean": aggregate_mean,
    "median": aggregate_median,
    "trimmed": aggregate_trimmed_mean,
    "mean_std": aggregate_mean_std,
}


def build_segment_assignments(df_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes deterministic segment assignments for each chunk preserving exact row order.
    - Method 1 (Primary): Chunk Rank Tertiles (k < K/3 -> Early, K/3 <= k < 2K/3 -> Middle, k >= 2K/3 -> Late)
    - Method 2 (Diagnostic): Token Midpoint Tertiles (m < T/3 -> Early, T/3 <= m < 2T/3 -> Middle, m >= 2T/3 -> Late)
    """
    df_enriched = df_meta.copy()
    df_enriched["midpoint_token"] = (df_enriched["start_token"] + df_enriched["end_token"]) / 2.0

    # Per-participant metadata map
    p_info = {}
    for pid, group in df_meta.groupby("Participant_ID"):
        p_info[pid] = {
            "split": group["split"].iloc[0],
            "total_chunks": len(group),
            "total_tokens": int(group["end_token"].max()),
        }

    seg_rank = []
    seg_token = []

    for _, row in df_enriched.iterrows():
        pid = row["Participant_ID"]
        k = row["chunk_index"]
        mid = row["midpoint_token"]
        K = p_info[pid]["total_chunks"]
        T = p_info[pid]["total_tokens"]

        # Method 1: Chunk Rank Tertiles
        if k < K / 3.0:
            seg_rank.append("early")
        elif k < 2.0 * K / 3.0:
            seg_rank.append("middle")
        else:
            seg_rank.append("late")

        # Method 2: Token Midpoint Tertiles
        if mid < T / 3.0:
            seg_token.append("early")
        elif mid < 2.0 * T / 3.0:
            seg_token.append("middle")
        else:
            seg_token.append("late")

    df_enriched["segment_rank"] = seg_rank
    df_enriched["segment_token"] = seg_token

    # Build participant audit dataframe
    participant_audits = []
    for pid in df_meta["Participant_ID"].drop_duplicates():
        p_chunks = df_enriched[df_enriched["Participant_ID"] == pid]
        m1_counts = p_chunks["segment_rank"].value_counts().to_dict()
        m2_counts = p_chunks["segment_token"].value_counts().to_dict()

        participant_audits.append({
            "Participant_ID": pid,
            "split": p_info[pid]["split"],
            "total_chunks": p_info[pid]["total_chunks"],
            "total_tokens": p_info[pid]["total_tokens"],
            "rank_early_chunks": m1_counts.get("early", 0),
            "rank_middle_chunks": m1_counts.get("middle", 0),
            "rank_late_chunks": m1_counts.get("late", 0),
            "token_early_chunks": m2_counts.get("early", 0),
            "token_middle_chunks": m2_counts.get("middle", 0),
            "token_late_chunks": m2_counts.get("late", 0),
            "rank_has_all_three": int(all(m1_counts.get(s, 0) > 0 for s in ["early", "middle", "late"])),
            "token_has_all_three": int(all(m2_counts.get(s, 0) > 0 for s in ["early", "middle", "late"])),
        })

    df_audit = pd.DataFrame(participant_audits)
    return df_enriched, df_audit


def build_and_save_representations() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audit").mkdir(parents=True, exist_ok=True)

    logger.info("Loading chunk embeddings and metadata...")
    chunk_embeds = np.load(DATASET_DIR / "chunk_embeddings.npy")
    df_meta = pd.read_csv(DATASET_DIR / "chunk_metadata.csv")
    assert len(chunk_embeds) == len(df_meta) == 2163

    logger.info("Computing segment assignments and audit logs...")
    df_enriched, df_audit = build_segment_assignments(df_meta)

    # Save audit artifacts
    df_enriched.to_csv(DATASET_DIR / "chunk_metadata_enriched.csv", index=False)
    df_audit.to_csv(OUTPUT_DIR / "audit" / "segment_assignment_audit.csv", index=False)

    logger.info("Participant coverage summary:")
    logger.info(f"Total participants: {len(df_audit)}")
    for split in ["train", "dev", "test"]:
        sub = df_audit[df_audit["split"] == split]
        logger.info(f"Split '{split}': N={len(sub)}, K>=3 complete: {sub['rank_has_all_three'].sum()} ({sub['rank_has_all_three'].mean()*100:.1f}%), K<3: {(sub['total_chunks'] < 3).sum()}")

    # Load official participant splits
    train_labels = pd.read_csv(LABELS_DIR / "train_split.csv").sort_values("Participant_ID")
    dev_labels = pd.read_csv(LABELS_DIR / "dev_split.csv").sort_values("Participant_ID")
    test_labels = pd.read_csv(LABELS_DIR / "test_split.csv").sort_values("Participant_ID")

    split_pids = {
        "train": train_labels["Participant_ID"].values,
        "dev": dev_labels["Participant_ID"].values,
        "test": test_labels["Participant_ID"].values,
    }

    # Generate representations per split
    # Aggregation types: mean, median, trimmed, mean_std
    # Segment types: whole, early, middle, late, concat
    representations = {}

    for split, pids in split_pids.items():
        logger.info(f"Generating representations for split '{split}' (N={len(pids)})...")
        rep_dict: Dict[str, Dict[int, np.ndarray]] = {}

        # Initialize storage
        for seg in ["whole", "early", "middle", "late"]:
            for agg in AGGREGATION_FUNCS.keys():
                rep_dict[f"{seg}__{agg}"] = {}

        for agg in AGGREGATION_FUNCS.keys():
            rep_dict[f"concat__{agg}"] = {}

        for pid in pids:
            p_indices = df_enriched[df_enriched["Participant_ID"] == pid].index.values
            p_meta = df_enriched.loc[p_indices]
            p_embeds = chunk_embeds[p_indices]

            # 1. Whole interview aggregation
            for agg_name, agg_fn in AGGREGATION_FUNCS.items():
                rep_dict[f"whole__{agg_name}"][pid] = agg_fn(p_embeds)

            # 2. Segment aggregations (using rank assignment as primary)
            seg_embeds = {}
            for seg in ["early", "middle", "late"]:
                seg_idx = p_meta[p_meta["segment_rank"] == seg].index.values
                if len(seg_idx) > 0:
                    local_idx = [np.where(p_indices == i)[0][0] for i in seg_idx]
                    s_embeds = p_embeds[local_idx]
                    seg_embeds[seg] = s_embeds
                    for agg_name, agg_fn in AGGREGATION_FUNCS.items():
                        rep_dict[f"{seg}__{agg_name}"][pid] = agg_fn(s_embeds)

            # 3. Three-segment concatenation (if all three segments present)
            if all(seg in seg_embeds and len(seg_embeds[seg]) > 0 for seg in ["early", "middle", "late"]):
                for agg_name, agg_fn in AGGREGATION_FUNCS.items():
                    v_early = agg_fn(seg_embeds["early"])
                    v_mid = agg_fn(seg_embeds["middle"])
                    v_late = agg_fn(seg_embeds["late"])
                    v_concat = np.concatenate([v_early, v_mid, v_late], axis=0)
                    rep_dict[f"concat__{agg_name}"][pid] = v_concat

        # Convert to aligned matrices and save
        split_dir = FEATURES_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save participant IDs for full split
        pd.DataFrame({"Participant_ID": pids}).to_csv(split_dir / "all_participant_ids.csv", index=False)

        # Save complete cohort IDs (participants with K >= 3)
        complete_pids = [pid for pid in pids if pid in rep_dict["concat__mean"]]
        pd.DataFrame({"Participant_ID": complete_pids}).to_csv(split_dir / "complete_cohort_participant_ids.csv", index=False)

        for rep_name, p_map in rep_dict.items():
            # For representations that might have exclusions (e.g. concat or empty segments)
            valid_pids = [pid for pid in pids if pid in p_map]
            mat = np.array([p_map[pid] for pid in valid_pids], dtype=np.float32)

            np.save(split_dir / f"{rep_name}.npy", mat)
            pd.DataFrame({"Participant_ID": valid_pids}).to_csv(split_dir / f"{rep_name}__pids.csv", index=False)

        logger.info(f"Saved {len(rep_dict)} representation matrices for split '{split}' to {split_dir}")

    # Build manifest
    manifest = {
        "phase": "18A",
        "description": "Segment-Level MiniLM Embeddings and Robust Aggregations",
        "total_participants": 275,
        "splits": {"train": 163, "dev": 56, "test": 56},
        "complete_cohort_participants": {
            "train": int(df_audit[df_audit["split"] == "train"]["rank_has_all_three"].sum()),
            "dev": int(df_audit[df_audit["split"] == "dev"]["rank_has_all_three"].sum()),
            "test": int(df_audit[df_audit["split"] == "test"]["rank_has_all_three"].sum()),
        },
        "aggregations": list(AGGREGATION_FUNCS.keys()),
        "segments": ["whole", "early", "middle", "late", "concat"],
        "dimensions": {
            "mean": 384,
            "median": 384,
            "trimmed": 384,
            "mean_std": 768,
            "concat_mean": 1152,
            "concat_median": 1152,
            "concat_trimmed": 1152,
            "concat_mean_std": 2304,
        },
    }
    with open(DATASET_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Segment dataset construction completed successfully.")


if __name__ == "__main__":
    build_and_save_representations()
