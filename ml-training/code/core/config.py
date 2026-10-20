"""
Central configuration for the E-DAIC depression-detection benchmark.
Preserves existing project structure and resolves delimiter/path ambiguities.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Base Project Root (auto-resolves relative to repo location)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Core Directories
if (ROOT_DIR / "edaic-woz" / "data").exists():
    DATA_DIR = ROOT_DIR / "edaic-woz" / "data"
elif (ROOT_DIR / "edaic_woz" / "data").exists():
    DATA_DIR = ROOT_DIR / "edaic_woz" / "data"
else:
    DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = ROOT_DIR / "dataset"
if (ROOT_DIR / "edaic-woz" / "labels").exists():
    LABELS_DIR = ROOT_DIR / "edaic-woz" / "labels"
elif (ROOT_DIR / "edaic_woz" / "labels").exists():
    LABELS_DIR = ROOT_DIR / "edaic_woz" / "labels"
else:
    LABELS_DIR = ROOT_DIR / "labels"
CODE_DIR = ROOT_DIR / "code"
CACHE_DIR = DATA_DIR / ".cache" / "features"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
CLASSIFICATION_OUTPUT_DIR = OUTPUTS_DIR / "classification"

# Classification Output Subdirectories
PREDICTIONS_DIR = CLASSIFICATION_OUTPUT_DIR / "predictions"
ROC_CURVES_DIR = CLASSIFICATION_OUTPUT_DIR / "roc_curves"
PR_CURVES_DIR = CLASSIFICATION_OUTPUT_DIR / "precision_recall_curves"
CONFUSION_MATRICES_DIR = CLASSIFICATION_OUTPUT_DIR / "confusion_matrices"
COMPARISONS_DIR = CLASSIFICATION_OUTPUT_DIR / "comparisons"
CURVE_DATA_DIR = CLASSIFICATION_OUTPUT_DIR / "curve_data"
CONFIGS_DIR = CLASSIFICATION_OUTPUT_DIR / "configs"

# Regression Output Directories
REGRESSION_OUTPUT_DIR = OUTPUTS_DIR / "regression"
REG_PREDICTIONS_DIR = REGRESSION_OUTPUT_DIR / "predictions"
REG_CONFIGS_DIR = REGRESSION_OUTPUT_DIR / "configs"
REG_COMPARISONS_DIR = REGRESSION_OUTPUT_DIR / "comparisons"
REG_AUDIT_DIR = REGRESSION_OUTPUT_DIR / "audit"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Target Variable Names
TARGET_REGRESSION = "PHQ_Score"
TARGET_CLASSIFICATION = "PHQ_Binary"
POSITIVE_CLASS_VALUE = 1  # 1 = Depressed (PHQ-8 >= 10), 0 = Non-depressed / Control
AUXILIARY_TARGETS = ["PCL-C (PTSD)", "PTSD Severity"]

# Official Split Files
SPLIT_FILES = {
    "train": LABELS_DIR / "train_split.csv",
    "dev": LABELS_DIR / "dev_split.csv",
    "test": LABELS_DIR / "test_split.csv",
}


@dataclass(frozen=True)
class FeatureSpec:
    """Specification of a feature representation schema."""
    name: str
    file_pattern: str
    has_header: bool
    expected_numeric_dim: int
    metadata_cols_to_drop: int
    sampling_rate_hz: float
    description: str


# Feature Specifications verified against actual dataset files
FEATURE_REGISTRY: Dict[str, FeatureSpec] = {
    "raw_egemaps": FeatureSpec(
        name="raw_egemaps",
        file_pattern="{pid}_OpenSMILE2.3.0_egemaps.csv",
        has_header=True,
        expected_numeric_dim=23,
        metadata_cols_to_drop=2,  # name, frameTime
        sampling_rate_hz=100.0,
        description="Raw eGeMAPS acoustic Low-Level Descriptors (23 features at 100 Hz)",
    ),
    "raw_mfcc": FeatureSpec(
        name="raw_mfcc",
        file_pattern="{pid}_OpenSMILE2.3.0_mfcc.csv",
        has_header=True,
        expected_numeric_dim=39,
        metadata_cols_to_drop=2,  # name, frameTime
        sampling_rate_hz=100.0,
        description="Raw MFCC 1-13 + deltas + delta-deltas (39 features at 100 Hz)",
    ),
    "boaw_egemaps": FeatureSpec(
        name="boaw_egemaps",
        file_pattern="{pid}_BoAW_openSMILE_2.3.0_eGeMAPS.csv",
        has_header=False,
        expected_numeric_dim=100,
        metadata_cols_to_drop=2,  # 'unknown', timestamp
        sampling_rate_hz=10.0,
        description="Bag-of-Audio-Words quantized eGeMAPS codebook histogram (100 bins at 10 Hz)",
    ),
    "boaw_mfcc": FeatureSpec(
        name="boaw_mfcc",
        file_pattern="{pid}_BoAW_openSMILE_2.3.0_MFCC.csv",
        has_header=False,
        expected_numeric_dim=100,
        metadata_cols_to_drop=2,  # 'unknown', timestamp
        sampling_rate_hz=10.0,
        description="Bag-of-Audio-Words quantized MFCC codebook histogram (100 bins at 10 Hz)",
    ),
}


def detect_delimiter(file_path: Path) -> str:
    """
    Detects whether a file uses comma or semicolon delimiter.
    Handles the split-dependent delimiter in BoAW files where:
      - Train/Dev files use comma (',')
      - Test files use semicolon (';')
      - Raw openSMILE files use semicolon (';')
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot detect delimiter for non-existent file: {file_path}")
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    
    if not first_line:
        raise ValueError(f"File is empty: {file_path}")
    
    num_semis = first_line.count(";")
    num_commas = first_line.count(",")
    
    if num_semis > num_commas:
        return ";"
    elif num_commas > num_semis:
        return ","
    else:
        # Fallback to semicolon if neither or equal, as openSMILE default is semicolon
        return ";"
