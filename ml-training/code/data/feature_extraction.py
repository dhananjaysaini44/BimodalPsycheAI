"""
Feature Extraction Layer for E-DAIC Depression Detection Benchmark.

Provides reusable, robust extraction functions for:
- Raw eGeMAPS (23 acoustic LLDs at 100 Hz)
- Raw MFCC (39 features: static + deltas + delta-deltas at 100 Hz)
- BoAW eGeMAPS (100 codebook bins at 10 Hz)
- BoAW MFCC (100 codebook bins at 10 Hz)

Handles split-dependent delimiters, header differences, caching, dimension checks,
and explicit error reporting without silent zero-imputation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import os
import numpy as np
import pandas as pd

try:
    from core.config import (
        DATA_DIR,
        CACHE_DIR,
        FEATURE_REGISTRY,
        FeatureSpec,
        detect_delimiter,
    )
except ImportError:
    from config import (
        DATA_DIR,
        CACHE_DIR,
        FEATURE_REGISTRY,
        FeatureSpec,
        detect_delimiter,
    )

logger = logging.getLogger(__name__)


class FeatureExtractionError(Exception):
    """Raised when feature extraction fails for a participant."""
    def __init__(self, message: str, participant_id: Optional[int] = None, feature_name: Optional[str] = None):
        super().__init__(message)
        self.participant_id = participant_id
        self.feature_name = feature_name


@dataclass
class ExtractionResult:
    """Encapsulates extracted feature matrix, metadata, and diagnostic records."""
    participant_id: int
    feature_name: str
    features: np.ndarray             # shape: (num_frames, num_features), dtype: float32
    timestamps: np.ndarray           # shape: (num_frames,), dtype: float64
    feature_names: List[str]         # list of stable column names
    source_path: str                 # absolute path to source file
    input_dimensions: Tuple[int, int] # raw rows (including header if present), raw cols
    output_dimensions: Tuple[int, int]# (num_frames, num_features)
    warnings: List[str]              # diagnostic warnings
    cached: bool = False             # True if loaded from disk cache

    def summary(self) -> str:
        return (
            f"ExtractionResult(PID={self.participant_id}, Feature={self.feature_name}, "
            f"Frames={self.output_dimensions[0]}, Dims={self.output_dimensions[1]}, "
            f"Cached={self.cached}, Warnings={len(self.warnings)})"
        )


def get_feature_path(participant_id: int, feature_name: str, data_dir: Path = DATA_DIR) -> Path:
    """
    Locates the verified feature file path for a given participant and representation.
    Path structure: data/{pid}_P/features/{filename}
    """
    if feature_name not in FEATURE_REGISTRY:
        raise ValueError(
            f"Unknown feature '{feature_name}'. Valid options: {list(FEATURE_REGISTRY.keys())}"
        )
    
    spec = FEATURE_REGISTRY[feature_name]
    fname = spec.file_pattern.format(pid=participant_id)
    folder = data_dir / f"{participant_id}_P" / "features"
    return folder / fname


def _save_to_cache(result: ExtractionResult, cache_dir: Path = CACHE_DIR) -> None:
    """Serializes extracted features, timestamps, and metadata to cache."""
    target_dir = cache_dir / result.feature_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = target_dir / f"{result.participant_id}.npz"
    meta_path = target_dir / f"{result.participant_id}.json"
    
    # Save numeric arrays
    np.savez_compressed(
        npz_path,
        features=result.features,
        timestamps=result.timestamps,
    )
    
    # Save metadata
    source_file = Path(result.source_path)
    mtime = source_file.stat().st_mtime if source_file.exists() else 0.0
    size = source_file.stat().st_size if source_file.exists() else 0
    
    meta = {
        "participant_id": result.participant_id,
        "feature_name": result.feature_name,
        "feature_names": result.feature_names,
        "source_path": result.source_path,
        "source_mtime": mtime,
        "source_size": size,
        "input_dimensions": list(result.input_dimensions),
        "output_dimensions": list(result.output_dimensions),
        "warnings": result.warnings,
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_from_cache(
    participant_id: int,
    feature_name: str,
    source_path: Path,
    cache_dir: Path = CACHE_DIR,
) -> Optional[ExtractionResult]:
    """
    Attempts to load extraction result from cache if cache is valid and source is unchanged.
    Returns None if cache is missing, corrupted, or stale.
    """
    target_dir = cache_dir / feature_name
    npz_path = target_dir / f"{participant_id}.npz"
    meta_path = target_dir / f"{participant_id}.json"
    
    if not (npz_path.exists() and meta_path.exists()):
        return None
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Verify source file integrity against cache
        if source_path.exists():
            curr_stat = source_path.stat()
            if curr_stat.st_mtime != meta.get("source_mtime") or curr_stat.st_size != meta.get("source_size"):
                return None  # Source modified; cache is stale
        
        with np.load(npz_path) as data:
            features = data["features"]
            timestamps = data["timestamps"]
        
        return ExtractionResult(
            participant_id=participant_id,
            feature_name=feature_name,
            features=features,
            timestamps=timestamps,
            feature_names=meta["feature_names"],
            source_path=meta["source_path"],
            input_dimensions=tuple(meta["input_dimensions"]),
            output_dimensions=tuple(meta["output_dimensions"]),
            warnings=meta.get("warnings", []),
            cached=True,
        )
    except Exception as e:
        logger.warning(f"Failed to read cache for PID {participant_id} ({feature_name}): {e}. Will re-extract.")
        return None


def extract_features(
    participant_id: int,
    feature_name: str,
    data_dir: Path = DATA_DIR,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
    force_rebuild: bool = False,
    allow_non_finite: bool = False,
) -> ExtractionResult:
    """
    Extracts features for a single participant and feature representation.

    Parameters:
    -----------
    participant_id: int
        E-DAIC participant ID (e.g., 300, 600).
    feature_name: str
        One of 'raw_egemaps', 'raw_mfcc', 'boaw_egemaps', 'boaw_mfcc'.
    data_dir: Path
        Root directory containing participant session folders (default: DATA_DIR).
    cache_dir: Path
        Directory storing cached extracted features.
    use_cache: bool
        Whether to check and load cached features if available.
    force_rebuild: bool
        If True, ignores existing cache and forces re-extraction from source CSV.
    allow_non_finite: bool
        If False (default), raises FeatureExtractionError on NaN/Inf values.
        If True, records a warning and replaces non-finite values with 0.0.

    Returns:
    --------
    ExtractionResult containing numeric features, timestamps, column names, and metadata.
    """
    if feature_name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature '{feature_name}'. Supported: {list(FEATURE_REGISTRY.keys())}")
    
    spec = FEATURE_REGISTRY[feature_name]
    file_path = get_feature_path(participant_id, feature_name, data_dir=data_dir)
    warnings: List[str] = []
    
    # 1. Check source file existence
    if not file_path.exists():
        raise FeatureExtractionError(
            f"Source file not found for participant {participant_id} and feature '{feature_name}': {file_path}",
            participant_id=participant_id,
            feature_name=feature_name,
        )
    
    # Check if empty (0 bytes)
    if file_path.stat().st_size == 0:
        raise FeatureExtractionError(
            f"Source file is completely empty (0 bytes): {file_path}",
            participant_id=participant_id,
            feature_name=feature_name,
        )

    # 2. Check Cache
    if use_cache and not force_rebuild:
        cached_res = _load_from_cache(participant_id, feature_name, file_path, cache_dir=cache_dir)
        if cached_res is not None:
            return cached_res

    # 3. Detect delimiter dynamically (, vs ;)
    delimiter = detect_delimiter(file_path)
    if feature_name.startswith("boaw") and delimiter == ";":
        warnings.append(f"BoAW file used semicolon (';') delimiter (test split format).")
    
    # 4. Read source file with appropriate header setting
    try:
        if spec.has_header:
            df = pd.read_csv(file_path, sep=delimiter)
            raw_rows, raw_cols = df.shape
            raw_rows += 1  # count the header line as part of input lines
            all_cols = list(df.columns)
            
            # Metadata exclusion: first column is identifier 'name', second is timestamp 'frameTime'
            id_col_name = all_cols[0]
            time_col_name = all_cols[1]
            feature_cols = all_cols[spec.metadata_cols_to_drop:]
            
            timestamps = df[time_col_name].to_numpy(dtype=np.float64)
            feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
            feature_names = [str(c) for c in feature_cols]
        else:
            # Headerless (BoAW files)
            df = pd.read_csv(file_path, sep=delimiter, header=None)
            raw_rows, raw_cols = df.shape
            
            # Metadata exclusion: Col 0 is 'unknown', Col 1 is timestamp
            timestamps = df.iloc[:, 1].to_numpy(dtype=np.float64)
            feature_matrix = df.iloc[:, spec.metadata_cols_to_drop:].to_numpy(dtype=np.float32)
            # Generate stable feature names
            feature_names = [f"{feature_name}_bin_{i}" for i in range(feature_matrix.shape[1])]

    except Exception as e:
        raise FeatureExtractionError(
            f"Failed to parse source file {file_path}: {e}",
            participant_id=participant_id,
            feature_name=feature_name,
        ) from e

    # 5. Dimension integrity validation
    actual_dim = feature_matrix.shape[1]
    if actual_dim != spec.expected_numeric_dim:
        raise FeatureExtractionError(
            f"Feature dimension mismatch for PID {participant_id} ({feature_name}): "
            f"expected {spec.expected_numeric_dim}, but extracted {actual_dim} columns. "
            f"(Raw columns in file: {raw_cols}, Delimiter used: '{delimiter}')",
            participant_id=participant_id,
            feature_name=feature_name,
        )
    
    if feature_matrix.shape[0] == 0:
        raise FeatureExtractionError(
            f"Feature matrix has 0 frames for PID {participant_id}: {file_path}",
            participant_id=participant_id,
            feature_name=feature_name,
        )

    # 6. Check for non-finite values (NaN, Inf)
    nan_mask = np.isnan(feature_matrix)
    inf_mask = np.isinf(feature_matrix)
    nan_count = int(np.sum(nan_mask))
    inf_count = int(np.sum(inf_mask))
    
    if nan_count > 0 or inf_count > 0:
        msg = f"Non-finite values detected in PID {participant_id} ({feature_name}): {nan_count} NaNs, {inf_count} Infs."
        if not allow_non_finite:
            raise FeatureExtractionError(msg, participant_id=participant_id, feature_name=feature_name)
        else:
            warnings.append(msg)
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # 7. Check timestamp monotonicity
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) >= 0):
        warnings.append("Timestamps are not strictly non-decreasing.")

    result = ExtractionResult(
        participant_id=participant_id,
        feature_name=feature_name,
        features=feature_matrix,
        timestamps=timestamps,
        feature_names=feature_names,
        source_path=str(file_path.resolve()),
        input_dimensions=(raw_rows, raw_cols),
        output_dimensions=feature_matrix.shape,
        warnings=warnings,
        cached=False,
    )

    # 8. Save to cache
    if use_cache:
        try:
            _save_to_cache(result, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Could not write cache for PID {participant_id}: {e}")

    return result


# Convenience functions for each of the four representations

def extract_raw_egemaps(
    participant_id: int,
    use_cache: bool = True,
    force_rebuild: bool = False,
    data_dir: Path = DATA_DIR,
) -> ExtractionResult:
    """Extracts Raw eGeMAPS acoustic features (23 features at 100 Hz)."""
    return extract_features(
        participant_id=participant_id,
        feature_name="raw_egemaps",
        data_dir=data_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
    )


def extract_raw_mfcc(
    participant_id: int,
    use_cache: bool = True,
    force_rebuild: bool = False,
    data_dir: Path = DATA_DIR,
) -> ExtractionResult:
    """Extracts Raw MFCC acoustic features (39 features at 100 Hz)."""
    return extract_features(
        participant_id=participant_id,
        feature_name="raw_mfcc",
        data_dir=data_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
    )


def extract_boaw_egemaps(
    participant_id: int,
    use_cache: bool = True,
    force_rebuild: bool = False,
    data_dir: Path = DATA_DIR,
) -> ExtractionResult:
    """Extracts BoAW eGeMAPS bag-of-words histogram features (100 bins at 10 Hz)."""
    return extract_features(
        participant_id=participant_id,
        feature_name="boaw_egemaps",
        data_dir=data_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
    )


def extract_boaw_mfcc(
    participant_id: int,
    use_cache: bool = True,
    force_rebuild: bool = False,
    data_dir: Path = DATA_DIR,
) -> ExtractionResult:
    """Extracts BoAW MFCC bag-of-words histogram features (100 bins at 10 Hz)."""
    return extract_features(
        participant_id=participant_id,
        feature_name="boaw_mfcc",
        data_dir=data_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
    )


def extract_batch(
    participant_ids: List[int],
    feature_name: str,
    use_cache: bool = True,
    force_rebuild: bool = False,
    data_dir: Path = DATA_DIR,
) -> Tuple[Dict[int, ExtractionResult], Dict[int, str]]:
    """
    Extracts features for a list of participants.
    Returns:
      - results: Dict[pid, ExtractionResult]
      - failures: Dict[pid, error_message] (failures are reported explicitly, never silently zeroed)
    """
    results: Dict[int, ExtractionResult] = {}
    failures: Dict[int, str] = {}

    for pid in participant_ids:
        try:
            res = extract_features(
                participant_id=pid,
                feature_name=feature_name,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                data_dir=data_dir,
            )
            results[pid] = res
        except FeatureExtractionError as e:
            failures[pid] = str(e)
            logger.error(f"Extraction failed for PID {pid}: {e}")
        except Exception as e:
            failures[pid] = f"Unexpected error: {e}"
            logger.error(f"Unexpected extraction error for PID {pid}: {e}")

    return results, failures
