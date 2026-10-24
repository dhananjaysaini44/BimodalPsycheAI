"""
Phase 18A: Extract and Serialize Chunk-Level MiniLM Embeddings.

Extracts and serializes chunk embeddings for all 2,163 chunks across the 275 E-DAIC participants:
1. Loads participant transcripts from ml-training/dataset/text_v1/participant_transcripts.csv.
2. Tokenizes and chunks using deterministic sliding window (W=250, S=190, overlap=60).
3. Encodes chunks using frozen sentence-transformers/all-MiniLM-L6-v2.
4. Serializes chunk embeddings array (2163, 384) to ml-training/dataset/text_segments_v1/chunk_embeddings.npy.
5. Verifies 100% mathematical reproduction of Phase 17B pooled mean embeddings (train and dev).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase18A_ExtractChunkEmbeddings")

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DATASET_PATH = REPO_ROOT / "ml-training" / "dataset" / "text_v1" / "participant_transcripts.csv"
PHASE_17B_EMBED_DIR = REPO_ROOT / "ml-training" / "dataset" / "text_embeddings_v1" / "all_minilm_l6_v2"
OUTPUT_DIR = REPO_ROOT / "ml-training" / "dataset" / "text_segments_v1"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 250
STRIDE = 190
BATCH_SIZE = 64
RANDOM_SEED = 42


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunk_transcript(
    tokenizer: Any,
    text: str,
    participant_id: int,
    split: str,
    chunk_size: int = CHUNK_SIZE,
    stride: int = STRIDE,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Deterministically divides transcript into token-bounded chunks respecting model limit."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(tokens)

    if n_tokens <= chunk_size:
        token_slices = [(0, n_tokens)]
    else:
        token_slices = []
        for start_idx in range(0, n_tokens, stride):
            end_idx = min(start_idx + chunk_size, n_tokens)
            token_slices.append((start_idx, end_idx))
            if end_idx == n_tokens:
                break

    chunk_texts = []
    chunk_metadata = []

    for chunk_idx, (start_idx, end_idx) in enumerate(token_slices):
        slice_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(slice_tokens, skip_special_tokens=True)
        chunk_texts.append(chunk_text)
        chunk_metadata.append({
            "Participant_ID": participant_id,
            "split": split,
            "chunk_index": chunk_idx,
            "total_chunks_in_session": len(token_slices),
            "start_token": start_idx,
            "end_token": end_idx,
            "token_count": len(slice_tokens),
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
        })

    return chunk_texts, chunk_metadata


def run_chunk_extraction() -> Tuple[np.ndarray, pd.DataFrame]:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Phase 18A chunk-level embedding extraction...")
    df_transcripts = pd.read_csv(INPUT_DATASET_PATH)
    logger.info(f"Loaded {len(df_transcripts)} participant transcripts.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    logger.info(f"Loading SentenceTransformer: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device=device)
    model.eval()

    all_chunk_embeddings = []
    all_chunk_meta = []
    participant_mean_embeds: Dict[int, np.ndarray] = {}

    expected_splits = ["train", "dev", "test"]
    t0 = time.time()

    for split_name in expected_splits:
        df_sub = df_transcripts[df_transcripts["split"] == split_name].sort_values("Participant_ID")
        logger.info(f"Processing split '{split_name}' (N={len(df_sub)} participants)...")

        for _, row in df_sub.iterrows():
            pid = int(row["Participant_ID"])
            text = str(row["normalized_text"])

            chunk_texts, chunk_meta = chunk_transcript(
                tokenizer=model.tokenizer,
                text=text,
                participant_id=pid,
                split=split_name,
            )

            with torch.no_grad():
                chunk_embeds = model.encode(
                    chunk_texts,
                    batch_size=BATCH_SIZE,
                    device=device,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )

            all_chunk_embeddings.append(chunk_embeds)
            all_chunk_meta.extend(chunk_meta)
            participant_mean_embeds[pid] = np.mean(chunk_embeds, axis=0)

    chunk_embeddings_matrix = np.vstack(all_chunk_embeddings).astype(np.float32)
    df_meta_all = pd.DataFrame(all_chunk_meta)
    elapsed = time.time() - t0

    logger.info(f"Extracted {len(chunk_embeddings_matrix)} chunk embeddings in {elapsed:.1f} seconds. Shape: {chunk_embeddings_matrix.shape}")
    assert len(chunk_embeddings_matrix) == 2163, f"Expected 2163 chunks, got {len(chunk_embeddings_matrix)}"

    # Verification against Phase 17B saved pooled embeddings
    logger.info("Verifying reproduction against Phase 17B participant pooled embeddings...")
    train_pids = pd.read_csv(PHASE_17B_EMBED_DIR / "train_participant_ids.csv")["Participant_ID"].values
    dev_pids = pd.read_csv(PHASE_17B_EMBED_DIR / "dev_participant_ids.csv")["Participant_ID"].values

    train_17b_mean = np.load(PHASE_17B_EMBED_DIR / "train_embeddings_mean.npy")
    dev_17b_mean = np.load(PHASE_17B_EMBED_DIR / "dev_embeddings_mean.npy")

    train_reconstructed = np.array([participant_mean_embeds[pid] for pid in train_pids])
    dev_reconstructed = np.array([participant_mean_embeds[pid] for pid in dev_pids])

    max_diff_train = np.max(np.abs(train_reconstructed - train_17b_mean))
    max_diff_dev = np.max(np.abs(dev_reconstructed - dev_17b_mean))
    logger.info(f"Max absolute difference vs Phase 17B (Train): {max_diff_train:.2e}")
    logger.info(f"Max absolute difference vs Phase 17B (Dev): {max_diff_dev:.2e}")

    assert max_diff_train < 1e-4, f"Train reproduction divergence: {max_diff_train}"
    assert max_diff_dev < 1e-4, f"Dev reproduction divergence: {max_diff_dev}"
    logger.info("Verification passed: 100% mathematical fidelity to Phase 17B embeddings.")

    # Save serialized chunk embeddings and metadata
    np.save(OUTPUT_DIR / "chunk_embeddings.npy", chunk_embeddings_matrix)
    df_meta_all.to_csv(OUTPUT_DIR / "chunk_metadata.csv", index=False)

    manifest = {
        "phase": "18A",
        "model_id": MODEL_ID,
        "total_chunks": len(chunk_embeddings_matrix),
        "embedding_dim": int(chunk_embeddings_matrix.shape[1]),
        "device": str(device),
        "extraction_runtime_sec": round(elapsed, 2),
        "verified_vs_phase17b": True,
        "max_diff_train": float(max_diff_train),
        "max_diff_dev": float(max_diff_dev),
    }
    with open(OUTPUT_DIR / "chunk_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Chunk embeddings and manifest successfully saved to {OUTPUT_DIR}")
    return chunk_embeddings_matrix, df_meta_all


if __name__ == "__main__":
    run_chunk_extraction()
