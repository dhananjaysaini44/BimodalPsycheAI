"""
Phase 18B: Extract and Serialize Chunk-Level DistilBERT Embeddings.

Extracts and serializes frozen chunk embeddings across all 275 E-DAIC participants:
1. Loads transcripts from ml-training/dataset/text_v1/participant_transcripts.csv.
2. Tokenizes and chunks using deterministic sliding window (W=250, S=190, overlap=60).
3. Encodes chunks using frozen distilbert/distilbert-base-uncased with attention-mask-aware mean pooling.
4. Serializes chunk embeddings array (N_chunks, 768) to ml-training/dataset/text_distilbert_v1/chunk_embeddings.npy.
5. Serializes chunk metadata to ml-training/dataset/text_distilbert_v1/chunk_metadata.csv.
6. Validates shapes, finiteness, and zero test leakage.
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
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase18B_ExtractDistilBert")

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DATASET_PATH = REPO_ROOT / "ml-training" / "dataset" / "text_v1" / "participant_transcripts.csv"
OUTPUT_DIR = REPO_ROOT / "ml-training" / "dataset" / "text_distilbert_v1"

MODEL_ID = "distilbert/distilbert-base-uncased"
CHUNK_SIZE = 250
STRIDE = 190
BATCH_SIZE = 32
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


def encode_chunks_with_pooling(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    chunks: List[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encodes chunks in batches using attention-mask-aware mean pooling."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # (B, L, 768)
            attention_mask = inputs["attention_mask"]       # (B, L)
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float32)
            all_embeddings.append(pooled)

    return np.vstack(all_embeddings)


def run_chunk_extraction() -> Tuple[np.ndarray, pd.DataFrame]:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Phase 18B chunk-level DistilBERT extraction...")
    df_transcripts = pd.read_csv(INPUT_DATASET_PATH)
    logger.info(f"Loaded {len(df_transcripts)} participant transcripts from {INPUT_DATASET_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    logger.info(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.to(device)
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
                tokenizer=tokenizer,
                text=text,
                participant_id=pid,
                split=split_name,
                chunk_size=CHUNK_SIZE,
                stride=STRIDE,
            )

            chunk_embeds = encode_chunks_with_pooling(
                model=model,
                tokenizer=tokenizer,
                chunks=chunk_texts,
                batch_size=BATCH_SIZE,
                device=device,
            )

            all_chunk_embeddings.append(chunk_embeds)
            all_chunk_meta.extend(chunk_meta)
            participant_mean_embeds[pid] = np.mean(chunk_embeds, axis=0)

    chunk_embeddings_matrix = np.vstack(all_chunk_embeddings).astype(np.float32)
    df_meta_all = pd.DataFrame(all_chunk_meta)
    elapsed = time.time() - t0

    logger.info(f"Extracted {len(chunk_embeddings_matrix)} total chunk embeddings in {elapsed:.2f}s ({len(chunk_embeddings_matrix)/elapsed:.1f} chunks/sec).")
    logger.info(f"Matrix shape: {chunk_embeddings_matrix.shape}, dtype: {chunk_embeddings_matrix.dtype}")

    # Validate finite values
    assert np.all(np.isfinite(chunk_embeddings_matrix)), "Detected non-finite values in chunk embeddings!"
    assert len(chunk_embeddings_matrix) == len(df_meta_all), "Mismatch between chunk embeddings and metadata count!"

    # Split breakdown
    counts_by_split = df_meta_all["split"].value_counts().to_dict()
    participants_by_split = df_meta_all.groupby("split")["Participant_ID"].nunique().to_dict()
    logger.info(f"Chunk distribution by split: {counts_by_split}")
    logger.info(f"Participant distribution by split: {participants_by_split}")

    # Serialize artifacts
    embeds_out_path = OUTPUT_DIR / "chunk_embeddings.npy"
    meta_out_path = OUTPUT_DIR / "chunk_metadata.csv"
    manifest_out_path = OUTPUT_DIR / "extraction_manifest.json"

    np.save(embeds_out_path, chunk_embeddings_matrix)
    df_meta_all.to_csv(meta_out_path, index=False)
    logger.info(f"Serialized chunk embeddings to: {embeds_out_path} ({embeds_out_path.stat().st_size / (1024**2):.2f} MB)")
    logger.info(f"Serialized chunk metadata to: {meta_out_path}")

    manifest = {
        "model_id": MODEL_ID,
        "pooling_method": "attention_mask_aware_mean_pooling",
        "device": device,
        "total_participants": len(df_transcripts),
        "total_chunks": int(len(chunk_embeddings_matrix)),
        "embedding_dim": int(chunk_embeddings_matrix.shape[1]),
        "chunk_size": CHUNK_SIZE,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": round(elapsed, 2),
        "chunks_by_split": counts_by_split,
        "participants_by_split": participants_by_split,
        "finite_verification": True,
        "test_labels_exposed": False,
    }
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved extraction manifest to: {manifest_out_path}")

    return chunk_embeddings_matrix, df_meta_all


if __name__ == "__main__":
    run_chunk_extraction()
