"""
Temporal Pooling Module for E-DAIC Benchmark Representations.

Provides pure, deterministic pooling functions over temporal sequence matrices:
- mean: Temporal arithmetic mean across frames
- std: Sample standard deviation (ddof=1, with safeguard for T <= 1)
- min: Minimum value across frames
- max: Maximum value across frames
- mean_std: Concatenation of mean and sample standard deviation
- mean_std_min_max: Concatenation of mean, sample standard deviation, min, and max

All pooling operations operate strictly on numeric feature columns along axis 0.
Participant IDs, labels, timestamps, and metadata are excluded prior to pooling.
"""

from typing import Dict, List, Tuple
import numpy as np


VALID_POOLING_STRATEGIES = [
    "mean",
    "std",
    "min",
    "max",
    "mean_std",
    "mean_std_min_max",
]

# Standard deviation convention documentation:
# ddof=1 computes the unbiased sample standard deviation: sqrt(sum((x - mean)^2) / (N - 1))
# If N <= 1, sample standard deviation is mathematically undefined / 0 degrees of freedom,
# so 0.0 is returned as a safe finite value.
STD_DDOF = 1


def pool_sequence(
    X: np.ndarray,
    strategy: str,
    ddof: int = STD_DDOF,
    allow_non_finite: bool = False,
) -> np.ndarray:
    """
    Applies temporal pooling across frames (axis 0) of sequence matrix X.

    Parameters:
    -----------
    X: np.ndarray
        2D numeric feature matrix of shape (num_frames, num_features).
    strategy: str
        One of 'mean', 'std', 'min', 'max', 'mean_std', 'mean_std_min_max'.
    ddof: int
        Delta degrees of freedom for standard deviation (default: 1 for sample std).
    allow_non_finite: bool
        If False, raises ValueError if non-finite values (NaN, Inf) are present.

    Returns:
    --------
    1D np.ndarray of pooled features for the participant.
    """
    if strategy not in VALID_POOLING_STRATEGIES:
        raise ValueError(
            f"Unknown pooling strategy '{strategy}'. Valid strategies: {VALID_POOLING_STRATEGIES}"
        )
    
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (num_frames, num_features), got ndim={X.ndim}")
    
    num_frames, num_features = X.shape
    if num_frames == 0 or num_features == 0:
        raise ValueError(f"Cannot pool empty sequence with shape {X.shape}")

    # Check non-finite values
    if not np.isfinite(X).all():
        if not allow_non_finite:
            raise ValueError(
                f"Non-finite values (NaN or Inf) detected in sequence of shape {X.shape}. "
                "Set allow_non_finite=True if intentional."
            )
        # Graceful fallback: clean non-finite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Core statistics along axis 0
    mean_val = np.mean(X, axis=0)
    
    if num_frames > 1:
        std_val = np.std(X, axis=0, ddof=ddof)
    else:
        # If single frame, sample variance with ddof=1 has zero degrees of freedom -> 0.0
        std_val = np.zeros(num_features, dtype=X.dtype)
    
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    if strategy == "mean":
        return mean_val
    elif strategy == "std":
        return std_val
    elif strategy == "min":
        return min_val
    elif strategy == "max":
        return max_val
    elif strategy == "mean_std":
        return np.concatenate([mean_val, std_val], axis=0)
    elif strategy == "mean_std_min_max":
        return np.concatenate([mean_val, std_val, min_val, max_val], axis=0)
    else:
        raise ValueError(f"Unhandled strategy: {strategy}")


def compute_all_pooled_variants(
    X: np.ndarray,
    ddof: int = STD_DDOF,
    allow_non_finite: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Efficiently computes all 6 pooling variants for sequence X in a single pass.
    Avoids recomputing basic statistics multiple times.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (num_frames, num_features), got ndim={X.ndim}")
    
    num_frames, num_features = X.shape
    if num_frames == 0 or num_features == 0:
        raise ValueError(f"Cannot pool empty sequence with shape {X.shape}")

    if not np.isfinite(X).all():
        if not allow_non_finite:
            raise ValueError(f"Non-finite values detected in sequence of shape {X.shape}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mean_val = np.mean(X, axis=0)
    std_val = np.std(X, axis=0, ddof=ddof) if num_frames > 1 else np.zeros(num_features, dtype=X.dtype)
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "mean_std": np.concatenate([mean_val, std_val], axis=0),
        "mean_std_min_max": np.concatenate([mean_val, std_val, min_val, max_val], axis=0),
    }


def get_pooled_feature_names(base_feature_names: List[str], strategy: str) -> List[str]:
    """
    Generates deterministic, unique column names for pooled features.
    
    Examples:
    - mean: ['f1_mean', 'f2_mean']
    - std: ['f1_std', 'f2_std']
    - mean_std: ['f1_mean', 'f2_mean', 'f1_std', 'f2_std']
    - mean_std_min_max: ['f1_mean', ..., 'f1_std', ..., 'f1_min', ..., 'f1_max', ...]
    """
    if strategy == "mean":
        return [f"{name}_mean" for name in base_feature_names]
    elif strategy == "std":
        return [f"{name}_std" for name in base_feature_names]
    elif strategy == "min":
        return [f"{name}_min" for name in base_feature_names]
    elif strategy == "max":
        return [f"{name}_max" for name in base_feature_names]
    elif strategy == "mean_std":
        return (
            [f"{name}_mean" for name in base_feature_names] +
            [f"{name}_std" for name in base_feature_names]
        )
    elif strategy == "mean_std_min_max":
        return (
            [f"{name}_mean" for name in base_feature_names] +
            [f"{name}_std" for name in base_feature_names] +
            [f"{name}_min" for name in base_feature_names] +
            [f"{name}_max" for name in base_feature_names]
        )
    else:
        raise ValueError(f"Unknown pooling strategy '{strategy}'")
