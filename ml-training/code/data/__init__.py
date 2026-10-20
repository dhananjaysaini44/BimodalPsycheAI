"""
Data management, feature extraction, pooling, and split loading package.
"""

try:
    from data.split_loader import (
        SplitIntegrityError,
        OfficialSplits,
        SplitData,
        DatasetSplits,
        load_official_splits,
        load_master_feature_dataset,
        load_split_dataset,
        load_all_splits,
        get_train_cv_splits,
        export_split_csvs,
    )
    from data.pooling import (
        VALID_POOLING_STRATEGIES,
        STD_DDOF,
        pool_sequence,
        compute_all_pooled_variants,
        get_pooled_feature_names,
    )
    from data.feature_extraction import (
        FeatureExtractionError,
        ExtractionResult,
        get_feature_path,
        extract_features,
        extract_raw_egemaps,
        extract_raw_mfcc,
        extract_boaw_egemaps,
        extract_boaw_mfcc,
        extract_batch,
    )
    from data.dataset_generation import (
        load_official_labels_and_splits,
        validate_dataset_integrity,
        generate_representation_datasets,
        generate_all_datasets,
    )
except ImportError:
    from split_loader import (
        SplitIntegrityError,
        OfficialSplits,
        SplitData,
        DatasetSplits,
        load_official_splits,
        load_master_feature_dataset,
        load_split_dataset,
        load_all_splits,
        get_train_cv_splits,
        export_split_csvs,
    )
    from pooling import (
        VALID_POOLING_STRATEGIES,
        STD_DDOF,
        pool_sequence,
        compute_all_pooled_variants,
        get_pooled_feature_names,
    )
    from feature_extraction import (
        FeatureExtractionError,
        ExtractionResult,
        get_feature_path,
        extract_features,
        extract_raw_egemaps,
        extract_raw_mfcc,
        extract_boaw_egemaps,
        extract_boaw_mfcc,
        extract_batch,
    )
    from dataset_generation import (
        load_official_labels_and_splits,
        validate_dataset_integrity,
        generate_representation_datasets,
        generate_all_datasets,
    )

__all__ = [
    # Split loading
    "SplitIntegrityError",
    "OfficialSplits",
    "SplitData",
    "DatasetSplits",
    "load_official_splits",
    "load_master_feature_dataset",
    "load_split_dataset",
    "load_all_splits",
    "get_train_cv_splits",
    "export_split_csvs",
    # Pooling
    "VALID_POOLING_STRATEGIES",
    "STD_DDOF",
    "pool_sequence",
    "compute_all_pooled_variants",
    "get_pooled_feature_names",
    # Feature extraction
    "FeatureExtractionError",
    "ExtractionResult",
    "get_feature_path",
    "extract_features",
    "extract_raw_egemaps",
    "extract_raw_mfcc",
    "extract_boaw_egemaps",
    "extract_boaw_mfcc",
    "extract_batch",
    # Dataset generation
    "load_official_labels_and_splits",
    "validate_dataset_integrity",
    "generate_representation_datasets",
    "generate_all_datasets",
]
