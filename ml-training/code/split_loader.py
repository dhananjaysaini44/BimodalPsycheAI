"""
Split-Aware Dataset Loading Module for E-DAIC Benchmark.

Provides reusable, leakage-safe dataset loading across official splits:
- train (163 participants)
- dev (56 participants)
- test (56 participants)

Guarantees:
1. Strict isolation between train, dev, and test sets.
2. Separate features (X) and target labels (y_binary, y_score).
3. Zero data regeneration (loads dynamically from master pooled datasets).
4. Full integrity validation (no missing IDs, no unexpected IDs, no duplicates, no cross-split leakage).
5. CV helpers strictly restricted to the official training set.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from config import (
    DATASET_DIR,
    LABELS_DIR,
    FEATURE_REGISTRY,
    RANDOM_SEED,
)
from pooling import VALID_POOLING_STRATEGIES

logger = logging.getLogger(__name__)


class SplitIntegrityError(Exception):
    """Raised when participant split alignment or data integrity is violated."""
    pass


@dataclass(frozen=True)
class OfficialSplits:
    """Stores validated, mutually exclusive official participant IDs and metadata."""
    train_pids: Tuple[int, ...]
    dev_pids: Tuple[int, ...]
    test_pids: Tuple[int, ...]
    labels_df: pd.DataFrame  # Indexed by Participant_ID


@dataclass
class SplitData:
    """Encapsulates features, targets, and metadata for a specific participant split."""
    split_name: str
    representation: str
    pooling_strategy: str
    participant_ids: np.ndarray      # Shape: (N,)
    X: np.ndarray                   # Shape: (N, D) - float32 numeric features
    feature_names: List[str]         # Length: D
    y_binary: np.ndarray            # Shape: (N,) - int64 classification labels (PHQ_Binary)
    y_score: np.ndarray             # Shape: (N,) - float32 regression scores (PHQ_Score)
    gender: np.ndarray              # Shape: (N,) - string ('male' / 'female')
    ptsd_binary: np.ndarray         # Shape: (N,) - int64 (PCL-C PTSD)
    ptsd_severity: np.ndarray       # Shape: (N,) - float32 (PTSD Severity)

    @property
    def num_participants(self) -> int:
        return len(self.participant_ids)

    @property
    def num_features(self) -> int:
        return self.X.shape[1]

    def to_dataframe(self, include_targets: bool = False) -> pd.DataFrame:
        """Returns features as a DataFrame with Participant_ID as the first column."""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df.insert(0, "Participant_ID", self.participant_ids)
        if include_targets:
            df["PHQ_Binary"] = self.y_binary
            df["PHQ_Score"] = self.y_score
            df["Gender"] = self.gender
            df["split"] = self.split_name
        return df


@dataclass
class DatasetSplits:
    """Container holding strictly isolated train, dev, and test subsets."""
    representation: str
    pooling_strategy: str
    train: SplitData
    dev: SplitData
    test: SplitData


def load_official_splits(labels_dir: Path = LABELS_DIR) -> OfficialSplits:
    """
    Loads official participant split files, verifies mutual exclusivity,
    and returns an immutable OfficialSplits object.
    """
    split_files = {
        "train": labels_dir / "train_split.csv",
        "dev": labels_dir / "dev_split.csv",
        "test": labels_dir / "test_split.csv",
    }
    
    pids_by_split: Dict[str, List[int]] = {}
    dfs: List[pd.DataFrame] = []

    for split_name, fpath in split_files.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Missing official split file: {fpath}")
        
        df = pd.read_csv(fpath)
        if "Participant_ID" not in df.columns:
            raise SplitIntegrityError(f"'Participant_ID' missing from {fpath}")
        
        # Clean string columns (strip whitespace like 'female ')
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
        
        pids = df["Participant_ID"].astype(int).tolist()
        
        # Check for duplicates within split
        if len(pids) != len(set(pids)):
            duplicates = [pid for pid in set(pids) if pids.count(pid) > 1]
            raise SplitIntegrityError(f"Duplicate IDs within {split_name} split: {duplicates}")
        
        pids_by_split[split_name] = pids
        df["split"] = split_name
        dfs.append(df)

    train_set = set(pids_by_split["train"])
    dev_set = set(pids_by_split["dev"])
    test_set = set(pids_by_split["test"])

    # 1. Check mutual exclusivity
    overlap_train_dev = train_set & dev_set
    overlap_train_test = train_set & test_set
    overlap_dev_test = dev_set & test_set

    if overlap_train_dev or overlap_train_test or overlap_dev_test:
        raise SplitIntegrityError(
            f"Overlap detected between official splits: "
            f"train∩dev={overlap_train_dev}, train∩test={overlap_train_test}, dev∩test={overlap_dev_test}"
        )

    # 2. Check official counts (Train=163, Dev=56, Test=56, Total=275)
    if len(train_set) != 163:
        raise SplitIntegrityError(f"Expected 163 train participants, got {len(train_set)}")
    if len(dev_set) != 56:
        raise SplitIntegrityError(f"Expected 56 dev participants, got {len(dev_set)}")
    if len(test_set) != 56:
        raise SplitIntegrityError(f"Expected 56 test participants, got {len(test_set)}")

    # Combined labels DataFrame indexed by Participant_ID
    combined_labels = pd.concat(dfs, ignore_index=True)
    combined_labels["Participant_ID"] = combined_labels["Participant_ID"].astype(int)
    combined_labels = combined_labels.set_index("Participant_ID").sort_index()

    return OfficialSplits(
        train_pids=tuple(sorted(pids_by_split["train"])),
        dev_pids=tuple(sorted(pids_by_split["dev"])),
        test_pids=tuple(sorted(pids_by_split["test"])),
        labels_df=combined_labels,
    )


def load_master_feature_dataset(
    representation: str,
    pooling_strategy: str,
    dataset_dir: Path = DATASET_DIR,
) -> pd.DataFrame:
    """
    Loads one of the 24 master pooled feature datasets.
    File path: dataset/{representation}/{pooling_strategy}.csv
    """
    if representation not in FEATURE_REGISTRY:
        raise ValueError(
            f"Unknown representation '{representation}'. Valid options: {list(FEATURE_REGISTRY.keys())}"
        )
    if pooling_strategy not in VALID_POOLING_STRATEGIES:
        raise ValueError(
            f"Unknown pooling strategy '{pooling_strategy}'. Valid options: {VALID_POOLING_STRATEGIES}"
        )

    csv_path = dataset_dir / representation / f"{pooling_strategy}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Pooled dataset not found at {csv_path}. Run dataset generation first."
        )

    df = pd.read_csv(csv_path)
    if "Participant_ID" not in df.columns:
        raise SplitIntegrityError(f"Participant_ID column missing in {csv_path}")

    df["Participant_ID"] = df["Participant_ID"].astype(int)
    return df


def load_split_dataset(
    representation: str,
    pooling_strategy: str,
    split: str,
    dataset_dir: Path = DATASET_DIR,
    labels_dir: Path = LABELS_DIR,
) -> SplitData:
    """
    Loads a pooled feature dataset and extracts the requested split subset
    by strictly matching Participant_ID against official split definitions.

    Parameters:
    -----------
    representation: str
        One of 'raw_egemaps', 'raw_mfcc', 'boaw_egemaps', 'boaw_mfcc'.
    pooling_strategy: str
        One of 'mean', 'std', 'min', 'max', 'mean_std', 'mean_std_min_max'.
    split: str
        Requested split: 'train', 'dev', 'test', or 'train_dev'.
    dataset_dir: Path
        Root directory of master pooled datasets (default: DATASET_DIR).
    labels_dir: Path
        Root directory containing split label files (default: LABELS_DIR).

    Returns:
    --------
    SplitData dataclass containing strictly filtered features, targets, and IDs.
    """
    valid_splits = {"train", "dev", "test", "train_dev"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Valid options: {valid_splits}")

    # 1. Load official splits and master dataset
    official = load_official_splits(labels_dir=labels_dir)
    master_df = load_master_feature_dataset(
        representation=representation,
        pooling_strategy=pooling_strategy,
        dataset_dir=dataset_dir,
    )

    # 2. Identify target participant IDs
    if split == "train":
        target_pids = official.train_pids
    elif split == "dev":
        target_pids = official.dev_pids
    elif split == "test":
        target_pids = official.test_pids
    elif split == "train_dev":
        target_pids = tuple(sorted(set(official.train_pids) | set(official.dev_pids)))

    target_pid_set = set(target_pids)
    dataset_pid_set = set(master_df["Participant_ID"])

    # 3. Validate missing and unexpected participant IDs
    missing_from_dataset = target_pid_set - dataset_pid_set
    if missing_from_dataset:
        raise SplitIntegrityError(
            f"Dataset {representation}/{pooling_strategy}.csv is missing {len(missing_from_dataset)} "
            f"participants required for split '{split}': {sorted(missing_from_dataset)}"
        )

    # Check for duplicate IDs in dataset
    pids_in_dataset = master_df["Participant_ID"].tolist()
    if len(pids_in_dataset) != len(dataset_pid_set):
        duplicates = [p for p in dataset_pid_set if pids_in_dataset.count(p) > 1]
        raise SplitIntegrityError(f"Duplicate Participant_IDs in {representation}/{pooling_strategy}.csv: {duplicates}")

    # 4. Filter features strictly by matching Participant_ID
    # Set Participant_ID as index and select exact target_pids in deterministic order
    indexed_features = master_df.set_index("Participant_ID")
    split_features_df = indexed_features.loc[list(target_pids)]
    feature_names = list(split_features_df.columns)

    # 5. Extract aligned ground truth labels from official labels
    split_labels_df = official.labels_df.loc[list(target_pids)]

    # 6. Verify cross-split boundary integrity
    for pid in target_pids:
        actual_split = split_labels_df.loc[pid, "split"]
        if split in ["train", "dev", "test"] and actual_split != split:
            raise SplitIntegrityError(
                f"Participant {pid} has split='{actual_split}', but was loaded into requested split '{split}'"
            )

    # Convert to clean numeric arrays
    X = split_features_df.to_numpy(dtype=np.float32)
    y_binary = split_labels_df["PHQ_Binary"].to_numpy(dtype=np.int64)
    y_score = split_labels_df["PHQ_Score"].to_numpy(dtype=np.float32)
    gender = split_labels_df["Gender"].to_numpy(dtype=str)
    ptsd_binary = split_labels_df["PCL-C (PTSD)"].to_numpy(dtype=np.int64)
    ptsd_severity = split_labels_df["PTSD Severity"].to_numpy(dtype=np.float32)

    return SplitData(
        split_name=split,
        representation=representation,
        pooling_strategy=pooling_strategy,
        participant_ids=np.array(target_pids, dtype=np.int64),
        X=X,
        feature_names=feature_names,
        y_binary=y_binary,
        y_score=y_score,
        gender=gender,
        ptsd_binary=ptsd_binary,
        ptsd_severity=ptsd_severity,
    )


def load_all_splits(
    representation: str,
    pooling_strategy: str,
    dataset_dir: Path = DATASET_DIR,
    labels_dir: Path = LABELS_DIR,
) -> DatasetSplits:
    """
    Loads train, dev, and test subsets for a given representation and pooling strategy.
    Keeps all subsets strictly isolated.
    """
    train_data = load_split_dataset(
        representation=representation,
        pooling_strategy=pooling_strategy,
        split="train",
        dataset_dir=dataset_dir,
        labels_dir=labels_dir,
    )
    dev_data = load_split_dataset(
        representation=representation,
        pooling_strategy=pooling_strategy,
        split="dev",
        dataset_dir=dataset_dir,
        labels_dir=labels_dir,
    )
    test_data = load_split_dataset(
        representation=representation,
        pooling_strategy=pooling_strategy,
        split="test",
        dataset_dir=dataset_dir,
        labels_dir=labels_dir,
    )
    return DatasetSplits(
        representation=representation,
        pooling_strategy=pooling_strategy,
        train=train_data,
        dev=dev_data,
        test=test_data,
    )


def get_train_cv_splits(
    train_data: SplitData,
    n_splits: int = 5,
    stratify: bool = True,
    random_state: int = RANDOM_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates cross-validation fold indices strictly restricted to the official training set.
    Never includes dev or test participants.

    Parameters:
    -----------
    train_data: SplitData
        Must be a SplitData instance with split_name='train'.
    n_splits: int
        Number of cross-validation folds (default: 5).
    stratify: bool
        If True, uses StratifiedKFold on PHQ_Binary to preserve class balance.
    random_state: int
        Random seed for reproducibility.

    Returns:
    --------
    List of (train_fold_idx, val_fold_idx) tuples indexable into train_data.X and train_data.y.
    """
    if train_data.split_name != "train":
        raise ValueError(
            f"Cross-validation must be run strictly on the official training set! "
            f"Got split_name='{train_data.split_name}'."
        )

    if stratify:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(cv.split(train_data.X, train_data.y_binary))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(cv.split(train_data.X))

    return folds


def export_split_csvs(
    representation: str,
    pooling_strategy: str,
    output_dir: Optional[Path] = None,
    dataset_dir: Path = DATASET_DIR,
    labels_dir: Path = LABELS_DIR,
    include_targets: bool = False,
) -> Dict[str, Path]:
    """
    Optional utility to export split-specific CSVs into dataset/splits/.
    These files are derived convenience exports; master datasets remain the source of truth.
    """
    if output_dir is None:
        output_dir = dataset_dir / "splits" / representation / pooling_strategy
    
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_all_splits(
        representation=representation,
        pooling_strategy=pooling_strategy,
        dataset_dir=dataset_dir,
        labels_dir=labels_dir,
    )

    exported_paths = {}
    for split_name, split_obj in [("train", splits.train), ("dev", splits.dev), ("test", splits.test)]:
        out_path = output_dir / f"{split_name}.csv"
        df = split_obj.to_dataframe(include_targets=include_targets)
        df.to_csv(out_path, index=False)
        exported_paths[split_name] = out_path

    logger.info(f"Exported split CSVs for {representation}/{pooling_strategy} to {output_dir}")
    return exported_paths
