"""
Stage 0 Test Suite: Master Pooled VGG16 and DenseNet201 Datasets.

Validates:
1. Numerical correctness of mean and mean_std pooling (with ddof=1).
2. Physical existence and dimensional integrity of the 4 master pooled datasets.
3. Metadata, manifest, validation report, and README generation.
4. Split-aware loader integration across official Train (163), Dev (56), and Test (56) splits.
5. Strict exclusion of metadata/timestamps/IDs from numeric feature matrices.
6. Target alignment (PHQ_Binary and PHQ_Score).
7. Cross-validation fold restriction to the official training set.
8. Deterministic loading without feature re-computation.
"""

from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd

import sys
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core.config import DATASET_DIR, DEEP_FEATURES_DIR, FEATURE_REGISTRY
from data.pooling import pool_sequence, STD_DDOF
from data.split_loader import (
    load_official_splits,
    load_master_feature_dataset,
    load_split_dataset,
    load_all_splits,
    get_train_cv_splits,
    SplitIntegrityError,
)


class TestDeepFeaturesStage0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deep_dir = DEEP_FEATURES_DIR
        cls.pooled_dir = cls.deep_dir / "pooled"
        cls.meta_dir = cls.deep_dir / "metadata"
        cls.reports_dir = cls.deep_dir / "reports"

    def test_01_numerical_pooling_mean(self):
        """Validates arithmetic mean pooling on a deterministic synthetic sequence."""
        X = np.array([
            [1.0, 2.0, 10.0],
            [3.0, 4.0, 20.0],
            [5.0, 6.0, 30.0],
        ], dtype=np.float32)

        mean_pooled = pool_sequence(X, strategy="mean")
        expected_mean = np.array([3.0, 4.0, 20.0], dtype=np.float32)
        np.testing.assert_allclose(mean_pooled, expected_mean, rtol=1e-5)

    def test_02_numerical_pooling_mean_std(self):
        """Validates sample standard deviation (ddof=1) and mean_std concatenation."""
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], dtype=np.float32)

        pooled = pool_sequence(X, strategy="mean_std", ddof=STD_DDOF)
        # mean: [3.0, 4.0]
        # sample std: ddof=1 -> sqrt(((1-3)^2 + 0 + (5-3)^2)/2) = sqrt(8/2) = 2.0
        expected = np.array([3.0, 4.0, 2.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(pooled, expected, rtol=1e-5)

    def test_03_single_frame_std_is_zero(self):
        """Validates that a single temporal frame returns 0.0 for standard deviation."""
        X = np.array([[7.5, 12.0]], dtype=np.float32)
        pooled = pool_sequence(X, strategy="mean_std", ddof=STD_DDOF)
        expected = np.array([7.5, 12.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(pooled, expected, rtol=1e-5)

    def test_04_master_datasets_exist(self):
        """Verifies that all 4 master pooled CSVs exist in dataset/deep_features_v1/pooled/."""
        expected_files = [
            "vgg16__mean.csv",
            "vgg16__mean_std.csv",
            "densenet201__mean.csv",
            "densenet201__mean_std.csv",
        ]
        for fname in expected_files:
            fpath = self.pooled_dir / fname
            self.assertTrue(fpath.exists(), f"Missing master pooled dataset: {fpath}")

    def test_05_feature_dimensions_and_row_counts(self):
        """Validates row counts (275) and column counts for each representation."""
        expected_dims = {
            "vgg16__mean.csv": (275, 4097),        # 1 ID + 4096 feats
            "vgg16__mean_std.csv": (275, 8193),    # 1 ID + 8192 feats
            "densenet201__mean.csv": (275, 1921),    # 1 ID + 1920 feats
            "densenet201__mean_std.csv": (275, 3841),# 1 ID + 3840 feats
        }
        for fname, (expected_rows, expected_cols) in expected_dims.items():
            fpath = self.pooled_dir / fname
            df = pd.read_csv(fpath)
            self.assertEqual(df.shape, (expected_rows, expected_cols), f"Shape mismatch in {fname}")
            self.assertEqual(df.columns[0], "Participant_ID", f"First column in {fname} must be 'Participant_ID'")
            self.assertFalse(df.isna().any().any(), f"NaNs detected in {fname}")
            # Ensure unique IDs
            self.assertEqual(len(df["Participant_ID"]), len(df["Participant_ID"].unique()), f"Duplicate IDs in {fname}")

    def test_06_metadata_and_manifest_files(self):
        """Verifies existence and structural completeness of inventory, manifest, report, and README."""
        inv_path = self.meta_dir / "participant_inventory.csv"
        manifest_path = self.meta_dir / "dataset_manifest.json"
        report_path = self.reports_dir / "stage0_validation.json"
        readme_path = self.deep_dir / "README.md"

        self.assertTrue(inv_path.exists(), "Missing participant_inventory.csv")
        self.assertTrue(manifest_path.exists(), "Missing dataset_manifest.json")
        self.assertTrue(report_path.exists(), "Missing stage0_validation.json")
        self.assertTrue(readme_path.exists(), "Missing README.md")

        # Validate inventory
        df_inv = pd.read_csv(inv_path)
        self.assertEqual(len(df_inv), 275)
        self.assertTrue((df_inv["vgg16_status"] == "VALID").all())
        self.assertTrue((df_inv["densenet201_status"] == "VALID").all())

        # Validate manifest JSON
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertIn("datasets", manifest)
        self.assertIn("vgg16__mean", manifest["datasets"])
        self.assertIn("densenet201__mean_std", manifest["datasets"])

        # Validate validation report JSON
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["coverage_summary"]["vgg16"]["total_processed"], 275)
        self.assertEqual(report["coverage_summary"]["densenet201"]["total_processed"], 275)
        self.assertEqual(report["integrity_checks"]["status"], "PASS")

    def test_07_split_loader_training_split(self):
        """Tests loading official Training split for VGG16 and DenseNet201."""
        for rep, expected_dim in [("vgg16", 4096), ("densenet201", 1920)]:
            data = load_split_dataset(representation=rep, pooling_strategy="mean", split="train")
            self.assertEqual(data.split_name, "train")
            self.assertEqual(data.representation, rep)
            self.assertEqual(data.pooling_strategy, "mean")
            self.assertEqual(len(data.participant_ids), 163)
            self.assertEqual(data.X.shape, (163, expected_dim))
            self.assertEqual(len(data.y_binary), 163)
            self.assertEqual(len(data.y_score), 163)
            self.assertEqual(data.X.dtype, np.float32)
            self.assertTrue(np.isfinite(data.X).all())

    def test_08_split_loader_dev_and_test_splits(self):
        """Tests loading Dev (56) and Test (56) splits with mean_std pooling."""
        # VGG16 mean_std Dev
        dev_vgg = load_split_dataset(representation="vgg16", pooling_strategy="mean_std", split="dev")
        self.assertEqual(len(dev_vgg.participant_ids), 56)
        self.assertEqual(dev_vgg.X.shape, (56, 8192))

        # DenseNet201 mean_std Test
        test_dense = load_split_dataset(representation="densenet201", pooling_strategy="mean_std", split="test")
        self.assertEqual(len(test_dense.participant_ids), 56)
        self.assertEqual(test_dense.X.shape, (56, 3840))

    def test_09_exclusion_of_metadata_and_identifiers(self):
        """Verifies that no metadata columns, timestamps, or IDs exist inside feature matrix X."""
        data = load_split_dataset(representation="vgg16", pooling_strategy="mean", split="dev")
        for col in data.feature_names:
            self.assertFalse(col in ["name", "timeStamp", "Participant_ID", "split"])
            self.assertTrue(col.startswith("neuron_"))

    def test_10_target_alignment(self):
        """Verifies exact alignment between Participant_IDs, X, and targets."""
        official = load_official_splits()
        data = load_split_dataset(representation="densenet201", pooling_strategy="mean", split="train")

        for idx, pid in enumerate(data.participant_ids):
            expected_binary = official.labels_df.loc[pid, "PHQ_Binary"]
            expected_score = official.labels_df.loc[pid, "PHQ_Score"]
            self.assertEqual(data.y_binary[idx], expected_binary)
            self.assertAlmostEqual(data.y_score[idx], expected_score, places=4)

    def test_11_cross_validation_helpers_use_train_only(self):
        """Verifies CV folds are strictly partitioned over the 163 training participants."""
        train_data = load_split_dataset(representation="vgg16", pooling_strategy="mean", split="train")
        folds = get_train_cv_splits(train_data, n_splits=5, stratify=True)
        self.assertEqual(len(folds), 5)

        all_val_indices = []
        for train_idx, val_idx in folds:
            # Overlap within fold must be empty
            self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
            self.assertEqual(len(train_idx) + len(val_idx), 163)
            all_val_indices.extend(val_idx)

        # All 163 train samples must be evaluated once
        self.assertEqual(sorted(all_val_indices), list(range(163)))

    def test_12_deterministic_reproducibility(self):
        """Verifies multiple loading calls return identical data without re-pooling."""
        run1 = load_split_dataset(representation="vgg16", pooling_strategy="mean", split="dev")
        run2 = load_split_dataset(representation="vgg16", pooling_strategy="mean", split="dev")

        np.testing.assert_array_equal(run1.participant_ids, run2.participant_ids)
        np.testing.assert_array_equal(run1.X, run2.X)
        np.testing.assert_array_equal(run1.y_binary, run2.y_binary)
        np.testing.assert_array_equal(run1.y_score, run2.y_score)


if __name__ == "__main__":
    unittest.main()
