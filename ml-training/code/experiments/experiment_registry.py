"""
Experiment Registry and Resume State Manager for E-DAIC Benchmark.

Builds deterministic experiment IDs:
{representation}__{pooling}__{model}__{evaluation}

Manages experiment configuration saving, completed experiment lookup,
and non-destructive resuming.
"""

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import json
import logging
import pandas as pd

code_root = str(Path(__file__).resolve().parent.parent)
if code_root not in sys.path:
    sys.path.insert(0, code_root)

from config import (
    CLASSIFICATION_OUTPUT_DIR,
    CONFIGS_DIR,
    RANDOM_SEED,
    TARGET_CLASSIFICATION,
    POSITIVE_CLASS_VALUE,
)

logger = logging.getLogger(__name__)


def build_experiment_id(
    representation: str,
    pooling_strategy: str,
    model_name: str,
    evaluation_protocol: str,
) -> str:
    """
    Constructs a deterministic, unique experiment ID.
    Examples:
    - raw_egemaps__mean__logreg__cv5
    - raw_egemaps__mean__logreg__dev
    """
    return f"{representation}__{pooling_strategy}__{model_name}__{evaluation_protocol}"


def get_completed_experiment_ids(results_csv_path: Optional[Path] = None) -> Set[str]:
    """
    Reads existing results.csv and returns set of experiment IDs that completed successfully.
    Used for safe, resumable execution.
    """
    if results_csv_path is None:
        results_csv_path = CLASSIFICATION_OUTPUT_DIR / "results.csv"

    if not results_csv_path.exists():
        return set()

    try:
        df = pd.read_csv(results_csv_path)
        if "experiment_id" in df.columns:
            # Only consider SUCCESS as completed
            if "status" in df.columns:
                completed = df[df["status"] == "SUCCESS"]["experiment_id"].tolist()
            else:
                completed = df["experiment_id"].tolist()
            return set(completed)
    except Exception as e:
        logger.warning(f"Failed to read existing results from {results_csv_path}: {e}")

    return set()


def save_experiment_config(
    experiment_id: str,
    representation: str,
    pooling_strategy: str,
    model_name: str,
    evaluation_protocol: str,
    model_params: Dict[str, Any],
    feature_count: int,
    configs_dir: Path = CONFIGS_DIR,
) -> Path:
    """Saves serialized JSON experiment configuration fingerprint."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    out_path = configs_dir / f"{experiment_id}.json"

    config_data = {
        "experiment_id": experiment_id,
        "representation": representation,
        "pooling_strategy": pooling_strategy,
        "model_name": model_name,
        "evaluation_protocol": evaluation_protocol,
        "target": TARGET_CLASSIFICATION,
        "positive_class": POSITIVE_CLASS_VALUE,
        "random_seed": RANDOM_SEED,
        "feature_count": feature_count,
        "model_params": model_params,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return out_path
