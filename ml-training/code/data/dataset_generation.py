"""
Dataset Generation and Pooling Pipeline for E-DAIC Benchmark.

Generates 24 benchmark feature datasets across 4 representations and 6 pooling strategies:
Representations:
- raw_egemaps
- raw_mfcc
- boaw_egemaps
- boaw_mfcc

Pooling Strategies:
- mean
- std
- min
- max
- mean_std
- mean_std_min_max

Features:
- Resumable generation with force_rebuild option.
- Explicit standard deviation convention (ddof=1 sample standard deviation).
- Zero label pollution: target labels are stored strictly separately in dataset/labels.csv.
- Full diagnostic validation: checks for duplicate IDs, non-finite values, constant features,
  and inconsistent dimensions.
- Comprehensive manifest generation saved to dataset/manifest.csv.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import numpy as np
import pandas as pd

try:
    from core.config import (
        DATA_DIR,
        DATASET_DIR,
        LABELS_DIR,
        SPLIT_FILES,
        FEATURE_REGISTRY,
        FeatureSpec,
    )
except ImportError:
    from config import (
        DATA_DIR,
        DATASET_DIR,
        LABELS_DIR,
        SPLIT_FILES,
        FEATURE_REGISTRY,
        FeatureSpec,
    )

try:
    from data.feature_extraction import (
        extract_features,
        ExtractionResult,
        FeatureExtractionError,
    )
except ImportError:
    from feature_extraction import (
        extract_features,
        ExtractionResult,
        FeatureExtractionError,
    )

try:
    from data.pooling import (
        VALID_POOLING_STRATEGIES,
        STD_DDOF,
        compute_all_pooled_variants,
        get_pooled_feature_names,
    )
except ImportError:
    from pooling import (
        VALID_POOLING_STRATEGIES,
        STD_DDOF,
        compute_all_pooled_variants,
        get_pooled_feature_names,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_official_labels_and_splits(labels_dir: Path = LABELS_DIR) -> pd.DataFrame:
    """
    Loads and standardizes participant labels and split memberships.
    Strips whitespace from text columns.
    Returns DataFrame with columns:
    ['Participant_ID', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity', 'split']
    """
    dfs = []
    for split_name in ["train", "dev", "test"]:
        split_path = labels_dir / f"{split_name}_split.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        
        df_split = pd.read_csv(split_path)
        # Strip whitespace from string columns (e.g., 'female ', 'male ')
        for col in df_split.select_dtypes(include="object").columns:
            df_split[col] = df_split[col].astype(str).str.strip()
        
        df_split["split"] = split_name
        dfs.append(df_split)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Participant_ID"] = combined["Participant_ID"].astype(int)
    
    # Sort deterministically by participant ID
    combined = combined.sort_values("Participant_ID").reset_index(drop=True)
    return combined


def validate_dataset_integrity(
    df: pd.DataFrame,
    expected_pids: List[int],
    expected_dim: int,
    representation: str,
    strategy: str,
) -> Dict[str, Any]:
    """
    Runs integrity checks on a generated feature dataset DataFrame.
    Returns diagnostic statistics dictionary.
    """
    pids = df["Participant_ID"].tolist()
    feature_cols = [c for c in df.columns if c != "Participant_ID"]
    num_features = len(feature_cols)
    num_rows = len(df)
    
    # 1. Duplicate IDs
    num_duplicates = len(pids) - len(set(pids))
    if num_duplicates > 0:
        raise ValueError(f"Duplicate Participant_IDs found in {representation}/{strategy}: {num_duplicates}")
    
    # 2. Participant alignment
    missing_pids = set(expected_pids) - set(pids)
    if missing_pids:
        raise ValueError(f"Missing expected participants in {representation}/{strategy}: {missing_pids}")
    
    # 3. Column count check
    if num_features != expected_dim:
        raise ValueError(
            f"Dimension mismatch in {representation}/{strategy}: expected {expected_dim}, got {num_features}"
        )

    # 4. Non-finite values check
    feature_data = df[feature_cols].to_numpy()
    nan_count = int(np.isnan(feature_data).sum())
    inf_count = int(np.isinf(feature_data).sum())
    total_non_finite = nan_count + inf_count

    # 5. Constant features check (features with std == 0 across all participants)
    feature_stds = np.nanstd(feature_data, axis=0)
    constant_features = [feature_cols[i] for i, s in enumerate(feature_stds) if s == 0.0]
    
    return {
        "participant_count": num_rows,
        "feature_count": num_features,
        "missing_count": nan_count,
        "non_finite_count": total_non_finite,
        "constant_feature_count": len(constant_features),
        "constant_features": constant_features,
    }


def generate_representation_datasets(
    representation: str,
    participants_df: pd.DataFrame,
    dataset_dir: Path = DATASET_DIR,
    force_rebuild: bool = False,
    use_extraction_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generates all 6 pooling strategy datasets for a single representation.
    Returns list of manifest records for each strategy.
    """
    if representation not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown representation '{representation}'")
    
    spec = FEATURE_REGISTRY[representation]
    rep_dir = dataset_dir / representation
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if all 6 already exist and are valid (for resumability)
    all_exist = True
    manifest_records = []
    pids_list = participants_df["Participant_ID"].tolist()

    expected_dims = {
        "mean": spec.expected_numeric_dim,
        "std": spec.expected_numeric_dim,
        "min": spec.expected_numeric_dim,
        "max": spec.expected_numeric_dim,
        "mean_std": spec.expected_numeric_dim * 2,
        "mean_std_min_max": spec.expected_numeric_dim * 4,
    }

    if not force_rebuild:
        for strat in VALID_POOLING_STRATEGIES:
            out_file = rep_dir / f"{strat}.csv"
            if not out_file.exists():
                all_exist = False
                break
            try:
                df_existing = pd.read_csv(out_file)
                stats = validate_dataset_integrity(
                    df_existing,
                    expected_pids=pids_list,
                    expected_dim=expected_dims[strat],
                    representation=representation,
                    strategy=strat,
                )
                manifest_records.append({
                    "representation": representation,
                    "pooling_strategy": strat,
                    "participant_count": stats["participant_count"],
                    "feature_count": stats["feature_count"],
                    "missing_count": stats["missing_count"],
                    "non_finite_count": stats["non_finite_count"],
                    "constant_feature_count": stats["constant_feature_count"],
                    "source_files": spec.file_pattern,
                    "output_path": str(out_file.relative_to(dataset_dir.parent)),
                    "status": "CACHED_VALID",
                })
            except Exception as e:
                logger.warning(f"Existing dataset {out_file} failed validation ({e}). Rebuilding.")
                all_exist = False
                manifest_records = []
                break
        
        if all_exist and len(manifest_records) == 6:
            logger.info(f"[{representation}] All 6 pooled datasets already exist and are valid. Skipping.")
            return manifest_records

    # Rebuild pooled datasets
    logger.info(f"[{representation}] Generating pooled datasets for {len(pids_list)} participants...")
    t0 = time.time()
    
    # Dictionary to hold pooled vectors per strategy
    # strategy -> list of dicts with 'Participant_ID' and pooled feature values
    pooled_data: Dict[str, List[Dict[str, Any]]] = {strat: [] for strat in VALID_POOLING_STRATEGIES}
    base_feature_names: Optional[List[str]] = None
    failed_pids: Dict[int, str] = {}

    for idx, pid in enumerate(pids_list):
        try:
            # Extract sequence (utilizing feature_extraction cache if available)
            res = extract_features(
                participant_id=pid,
                feature_name=representation,
                use_cache=use_extraction_cache,
            )
            
            if base_feature_names is None:
                base_feature_names = res.feature_names
            
            # Compute all 6 pooled variants in a single pass
            all_pooled = compute_all_pooled_variants(res.features, ddof=STD_DDOF)
            
            for strat in VALID_POOLING_STRATEGIES:
                vec = all_pooled[strat]
                pooled_data[strat].append({
                    "pid": pid,
                    "vector": vec,
                })
        except Exception as e:
            logger.error(f"Failed extraction for PID {pid} ({representation}): {e}")
            failed_pids[pid] = str(e)

    if failed_pids:
        raise RuntimeError(
            f"Extraction failed for {len(failed_pids)} participants in {representation}: {failed_pids}"
        )

    # Save each pooled strategy dataset to CSV
    records = []
    for strat in VALID_POOLING_STRATEGIES:
        out_file = rep_dir / f"{strat}.csv"
        pooled_cols = get_pooled_feature_names(base_feature_names, strat)
        
        # Build DataFrame
        rows = []
        for item in pooled_data[strat]:
            row_dict = {"Participant_ID": item["pid"]}
            row_dict.update(dict(zip(pooled_cols, item["vector"])))
            rows.append(row_dict)
        
        df_out = pd.DataFrame(rows)
        # Ensure Participant_ID is first
        df_out = df_out[["Participant_ID"] + pooled_cols]
        df_out.to_csv(out_file, index=False)
        
        # Validate saved dataset
        stats = validate_dataset_integrity(
            df_out,
            expected_pids=pids_list,
            expected_dim=expected_dims[strat],
            representation=representation,
            strategy=strat,
        )
        
        records.append({
            "representation": representation,
            "pooling_strategy": strat,
            "participant_count": stats["participant_count"],
            "feature_count": stats["feature_count"],
            "missing_count": stats["missing_count"],
            "non_finite_count": stats["non_finite_count"],
            "constant_feature_count": stats["constant_feature_count"],
            "source_files": spec.file_pattern,
            "output_path": str(out_file.relative_to(dataset_dir.parent)),
            "status": "SUCCESS",
        })

    logger.info(f"[{representation}] Finished generating 6 datasets in {time.time() - t0:.2f}s.")
    return records


def generate_all_datasets(
    dataset_dir: Path = DATASET_DIR,
    force_rebuild: bool = False,
    use_extraction_cache: bool = True,
) -> pd.DataFrame:
    """
    Main orchestrator function:
    1. Standardizes and saves isolated target labels to dataset/labels.csv.
    2. Generates all 24 representation-pooling combinations under dataset/.
    3. Produces and saves dataset/manifest.csv.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and isolate labels
    logger.info("Loading official participant labels and splits...")
    participants_df = load_official_labels_and_splits()
    labels_out_path = dataset_dir / "labels.csv"
    participants_df.to_csv(labels_out_path, index=False)
    logger.info(f"Saved isolated labels ({len(participants_df)} participants) to {labels_out_path}")

    # 2. Iterate representations and generate datasets
    all_manifest_records: List[Dict[str, Any]] = []
    
    for rep in ["raw_egemaps", "raw_mfcc", "boaw_egemaps", "boaw_mfcc"]:
        rep_records = generate_representation_datasets(
            representation=rep,
            participants_df=participants_df,
            dataset_dir=dataset_dir,
            force_rebuild=force_rebuild,
            use_extraction_cache=use_extraction_cache,
        )
        all_manifest_records.extend(rep_records)

    # 3. Create and save Manifest DataFrame
    manifest_df = pd.DataFrame(all_manifest_records)
    manifest_path = dataset_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved dataset generation manifest ({len(manifest_df)} combinations) to {manifest_path}")

    return manifest_df


if __name__ == "__main__":
    generate_all_datasets()
