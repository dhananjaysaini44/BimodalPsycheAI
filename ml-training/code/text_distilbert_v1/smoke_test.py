"""
Phase 18B: DistilBERT Extraction Smoke Test.

Validates:
1. Download / caching of distilbert/distilbert-base-uncased.
2. Tokenizer sliding-window behavior.
3. Attention-mask-aware mean pooling.
4. Output tensor shapes (N, 768), finite values, and absence of NaNs/Infs.
5. GPU memory consumption and extraction speed on 3 participants.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("Phase18B_SmokeTest")

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DATASET_PATH = REPO_ROOT / "ml-training" / "dataset" / "text_v1" / "participant_transcripts.csv"
OUTPUT_DIR = REPO_ROOT / "ml-training" / "outputs" / "text_distilbert_v1" / "audit"

MODEL_ID = "distilbert/distilbert-base-uncased"
CHUNK_SIZE = 250
STRIDE = 190
BATCH_SIZE = 32


def run_smoke_test():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Smoke test running on device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated() / (1024**2)
        logger.info(f"Initial VRAM allocated: {vram_start:.2f} MB")

    logger.info(f"Loading tokenizer and model for: {MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.2f}s")

    if torch.cuda.is_available():
        vram_model = torch.cuda.memory_allocated() / (1024**2)
        logger.info(f"VRAM after model load: {vram_model:.2f} MB")

    df_transcripts = pd.read_csv(INPUT_DATASET_PATH)
    sample_df = df_transcripts[df_transcripts["split"] == "train"].head(3)
    logger.info(f"Testing on 3 sample Train participants: {sample_df['Participant_ID'].tolist()}")

    all_chunks = []
    chunk_meta = []
    for _, row in sample_df.iterrows():
        pid = int(row["Participant_ID"])
        text = str(row["normalized_text"])
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        
        if n_tokens <= CHUNK_SIZE:
            slices = [(0, n_tokens)]
        else:
            slices = []
            for s in range(0, n_tokens, STRIDE):
                e = min(s + CHUNK_SIZE, n_tokens)
                slices.append((s, e))
                if e == n_tokens:
                    break
        
        for c_idx, (s, e) in enumerate(slices):
            c_tokens = tokens[s:e]
            c_text = tokenizer.decode(c_tokens, skip_special_tokens=True)
            all_chunks.append(c_text)
            chunk_meta.append({
                "pid": pid,
                "c_idx": c_idx,
                "tokens": len(c_tokens)
            })

    logger.info(f"Total chunks extracted for 3 participants: {len(all_chunks)}")

    # Encode with attention-mask-aware mean pooling
    t_start_enc = time.time()
    embeddings = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_texts = all_chunks[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            batch_embeds = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float32)
            embeddings.append(batch_embeds)
            
    embeddings = np.vstack(embeddings)
    enc_time = time.time() - t_start_enc
    logger.info(f"Encoded {len(embeddings)} chunks in {enc_time:.4f}s ({len(embeddings)/enc_time:.1f} chunks/sec)")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(f"Peak VRAM during extraction: {peak_vram:.2f} MB")
    else:
        peak_vram = 0.0

    # Verification checks
    shape_ok = embeddings.shape == (len(all_chunks), 768)
    finite_ok = bool(np.all(np.isfinite(embeddings)))
    non_zero = bool(np.all(np.linalg.norm(embeddings, axis=1) > 0.1))

    logger.info(f"Shape OK: {shape_ok} (Shape: {embeddings.shape})")
    logger.info(f"Finite OK: {finite_ok}")
    logger.info(f"Non-zero norm OK: {non_zero}")

    smoke_results = {
        "model_id": MODEL_ID,
        "device": device,
        "sample_participants": sample_df["Participant_ID"].tolist(),
        "total_chunks_tested": len(all_chunks),
        "embedding_shape": list(embeddings.shape),
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "non_zero_norm_ok": non_zero,
        "load_time_sec": round(load_time, 2),
        "encode_time_sec": round(enc_time, 4),
        "chunks_per_sec": round(len(embeddings) / enc_time, 1),
        "peak_vram_mb": round(peak_vram, 2),
        "smoke_test_passed": bool(shape_ok and finite_ok and non_zero),
    }

    out_file = OUTPUT_DIR / "extraction_smoke_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(smoke_results, f, indent=2)
    logger.info(f"Saved smoke test results to: {out_file}")

    assert smoke_results["smoke_test_passed"], "Smoke test failed verification checks!"
    logger.info("SMOKE TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_smoke_test()
